"""Panel image conversion + on-disk memoization for Figure Studio.

Source panels are not directly displayable in a browser: VessMAP/DRIVE/DCA1
inputs are TIFF, DRIVE ground truth is GIF, and neither decodes in an <img>
tag. This module is the single place that turns an arbitrary on-disk panel
(input / ground truth / prediction mask) into PNG/JPEG/WEBP bytes a browser
can render, and it exposes the same decode primitive (`load_array`) that the
grid composer imports, so gallery tiles and composed figures come from one
code path instead of two that could drift apart.

Masks (predictions, ground truth) are single-channel and binary-valued, so
they always resample with NEAREST — LANCZOS would blur 0/255 edges into grey
and quietly corrupt the data. Photographic inputs resample with LANCZOS for
quality.

Encoded bytes are memoized on disk under `paths.IMG_CACHE_DIR`, keyed on the
source file's identity (path + mtime + size) plus the requested width/format,
so a re-run inference file busts its own cache entry automatically and the
cache never needs explicit invalidation.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from . import gallery, paths

_MIME: dict[str, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}
_PIL_FORMAT: dict[str, str] = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}
_EXT: dict[str, str] = {"png": "png", "jpeg": "jpg", "webp": "webp"}

# Modes that already carry no colour information; anything else becomes RGB.
_GRAYSCALE_MODES = {"1", "L", "I", "I;16", "I;16B", "I;16L", "F"}

# Prune the cache every Nth write rather than checking on every request —
# a cheap guard, not a hot-path cost. The counter is shared across uvicorn's
# threadpool workers, hence the lock.
_PRUNE_EVERY = 32
_counter_lock = threading.Lock()
_write_count = 0


def _input_mode(img: Image.Image) -> str:
    """Target Pillow mode for a non-mask panel.

    RGB for colour sources (DRIVE's RGB fundus photos); L for single-channel
    sources (VessMAP/DCA1/OCTA2D, all stored greyscale).
    """
    return "L" if img.mode in _GRAYSCALE_MODES else "RGB"


def load_array(path: Path, kind: str, size: tuple[int, int] | None = None) -> np.ndarray:
    """Decode one panel into a numpy array, ready for compositing.

    `kind` ("pred"/"gt"/"input") picks the target colour mode via
    `gallery.is_mask`: masks become single-channel "L"; other panels keep RGB
    if the source is colour, else "L". `size`, if given, is a (width, height)
    to resample to: NEAREST for masks (a blurred binary mask is corrupted
    data), LANCZOS for photographic inputs.

    This is the shared decode path the grid composer also imports, so the
    signature is kept stable.
    """
    with Image.open(path) as img:
        img.load()
        mask = gallery.is_mask(kind)
        mode = "L" if mask else _input_mode(img)
        if img.mode != mode:
            img = img.convert(mode)
        if size is not None:
            resample = Image.NEAREST if mask else Image.LANCZOS
            img = img.resize(size, resample=resample)
        return np.array(img)


def _stat_digest(path: Path, width: int | None, fmt: str, kind: str) -> str:
    """Hash covering everything that determines the encoded bytes.

    Keys on the source file's mtime+size (not just its path) so a
    regenerated inference PNG busts its own cache entry instead of serving a
    stale image forever.
    """
    st = path.stat()
    payload = f"{path}|{st.st_mtime_ns}|{st.st_size}|{width}|{fmt}|{kind}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def etag_for(path: Path, width: int | None, fmt: str, kind: str) -> str:
    """Stable strong ETag string, built from the same inputs as the cache key."""
    return f'"{_stat_digest(path, width, fmt, kind)}"'


def _cache_path(digest: str, fmt: str) -> Path:
    return paths.IMG_CACHE_DIR / f"{digest}.{_EXT[fmt]}"


def _read_cache(cache_path: Path) -> bytes | None:
    try:
        content = cache_path.read_bytes()
    except OSError:
        return None
    # Bump mtime/atime on hit so LRU pruning reflects real recency even on
    # filesystems mounted noatime (atime alone can't be trusted there).
    with contextlib.suppress(OSError):
        os.utime(cache_path, None)
    return content


def _write_cache(cache_path: Path, content: bytes) -> None:
    """Best-effort disk memoization: write to a temp file, then atomic replace.

    Atomic rename (rather than a lock) is what makes this safe across
    uvicorn's threadpool: a partially-written file is never visible at
    `cache_path`, and concurrent writers for the same key just race to
    replace it with equal content. Any failure (disk full, read-only fs,
    permissions) is swallowed — the cache is a pure optimization, never
    load-bearing.
    """
    global _write_count
    try:
        paths.IMG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(paths.IMG_CACHE_DIR), prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(content)
            os.replace(tmp_path, cache_path)
        except OSError:
            with contextlib.suppress(OSError):
                os.remove(tmp_path)
            return
    except OSError:
        return

    with _counter_lock:
        _write_count += 1
        count = _write_count
    if count % _PRUNE_EVERY == 0:
        prune()


def prune(max_bytes: int = 512 * 1024 * 1024) -> None:
    """Evict least-recently-used cache files until the cache is under `max_bytes`.

    Called opportunistically from `_write_cache` (not on every request), so
    it never sits on the hot path. Best-effort: any filesystem error aborts
    silently, leaving the cache as-is for next time.
    """
    try:
        entries: list[tuple[float, int, Path]] = []
        total = 0
        for p in paths.IMG_CACHE_DIR.iterdir():
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            total += st.st_size
            recency = st.st_atime or st.st_mtime  # fall back if atime is unreliable
            entries.append((recency, st.st_size, p))
        if total <= max_bytes:
            return
        entries.sort(key=lambda e: e[0])  # oldest access first
        for _recency, size, p in entries:
            if total <= max_bytes:
                break
            with contextlib.suppress(OSError):
                p.unlink()
                total -= size
    except OSError:
        return


def encoded(path: Path, kind: str, width: int | None = None, fmt: str = "png") -> tuple[bytes, str]:
    """Convert one panel to web-displayable bytes, memoized on disk.

    Downscales to `width` (preserving aspect ratio) when given and smaller
    than the source width; never upscales. Masks resample NEAREST, inputs
    LANCZOS. Returns `(bytes, mime_type)`.
    """
    fmt = fmt.lower()
    if fmt not in _MIME:
        raise ValueError(f"unsupported format: {fmt!r}")

    digest = _stat_digest(path, width, fmt, kind)
    cache_path = _cache_path(digest, fmt)

    cached = _read_cache(cache_path)
    if cached is not None:
        return cached, _MIME[fmt]

    mask = gallery.is_mask(kind)
    with Image.open(path) as img:
        img.load()
        mode = "L" if mask else _input_mode(img)
        if img.mode != mode:
            img = img.convert(mode)
        if width is not None and width < img.width:
            new_height = max(1, round(img.height * width / img.width))
            resample = Image.NEAREST if mask else Image.LANCZOS
            img = img.resize((width, new_height), resample=resample)
        buf = io.BytesIO()
        save_kwargs = {"quality": 90} if fmt == "jpeg" else {}
        img.save(buf, format=_PIL_FORMAT[fmt], **save_kwargs)
    content = buf.getvalue()

    _write_cache(cache_path, content)
    return content, _MIME[fmt]
