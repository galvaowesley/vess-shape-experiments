"""Publication-figure grid composer for Figure Studio.

A qualitative-results figure for a paper is a grid of segmentation panels
(input | ground truth | model predictions) arranged by model x num_samples,
with one red ROI box drawn once per source image and repeated identically on
every panel that shows it -- even though those panels come from different
pipelines at different resolutions (predictions are saved at the model's
resize, 256/288/384; input/GT stay at native resolution, which for DRIVE is
non-square). Two things make the same box land in the same place everywhere:
ROI coordinates are normalized [0,1] (`RoiSpec`), and every panel that shares
an ROI key is resampled to one common canvas size before drawing (see
`_canvas_sizes`), so a marker/crop/inset computed from normalized coordinates
always lands on the same displayed pixel grid.

`render_grid(spec)` is the single code path for both the live composite and
the exported paper figure (svg/png/jpg/pdf), mirroring `plotting.py`'s
`render_figure`. It never raises on a single bad panel -- a missing
prediction file or an unreadable image degrades to a "missing" placeholder,
because one stale run must not blow up a 20-panel figure a researcher is
about to paste into a paper. A `kind="empty"` panel is not an error at all:
it is how ragged layouts (a header row of input/GT, mismatched grid shapes)
are built, so it renders as a quiet gap.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import gallery, imagesvc  # noqa: E402
from .schemas import CropSpec, GridFigureSpec, PanelRef, RoiSpec  # noqa: E402

_MIME: dict[str, str] = {
    "svg": "image/svg+xml",
    "png": "image/png",
    "jpg": "image/jpeg",
    "pdf": "application/pdf",
}

# File extension per export format (matches `_EXT` in routers/figuregrid.py).
_EXT_NAME: dict[str, str] = {"svg": "svg", "png": "png", "jpg": "jpg", "pdf": "pdf"}

# axes-fraction (x, y, ha, va) for each corner metric_loc can name.
_METRIC_LOC: dict[str, tuple[float, float, str, str]] = {
    "upper left": (0.03, 0.97, "left", "top"),
    "upper right": (0.97, 0.97, "right", "top"),
    "lower left": (0.03, 0.03, "left", "bottom"),
    "lower right": (0.97, 0.03, "right", "bottom"),
}

# inset_corner -> (left, bottom, width, height) in axes-fraction, given a
# scale that is a FRACTION of the panel (not a zoom factor).
_INSET_BOUNDS = {
    "upper right": lambda s: (1 - s, 1 - s, s, s),
    "upper left": lambda s: (0.0, 1 - s, s, s),
    "lower right": lambda s: (1 - s, 0.0, s, s),
    "lower left": lambda s: (0.0, 0.0, s, s),
}

_MIN_ROI = 1e-3          # clamp so w/h never hits zero (divide-by-zero / invisible axis)
_MISSING_COLOR = "#8a8a8a"


# --------------------------------------------------------------------------- #
# Panel identity helpers -- mirror lib/figureSpec.ts exactly (`panelLabel`,
# `roiKey`), since that TS module and this one must agree on what one ROI
# applies to and what a panel's caption says.
# --------------------------------------------------------------------------- #
def _panel_label(p: PanelRef) -> str:
    if p.kind == "input":
        return "Input"
    if p.kind == "gt":
        return "Ground truth"
    if p.kind == "empty":
        return ""
    shot: Optional[str]
    if p.num_samples == 0:
        shot = "Zero-shot"
    elif p.num_samples is not None:
        shot = f"{p.num_samples}-shot"
    else:
        shot = None
    if p.model_type and shot:
        return f"{p.model_type} · {shot}"
    return p.model_type or shot or "Prediction"


def _cell_label(panel: PanelRef, index: int, spec: GridFigureSpec) -> str:
    """Per-panel caption with the parts the row/column headers already say removed.

    In a model x num_samples grid the full label ("VSUNet18 · 1-shot") duplicates
    both headers, so every cell would repeat them and the text would collide
    across cells. Dropping the covered parts leaves captions only where they add
    information — in practice the input/GT column, which has no header of its own.
    """
    label = _panel_label(panel)
    if not label:
        return ""
    r, c = divmod(index, spec.cols)
    covered = {
        (spec.row_labels[r] if r < len(spec.row_labels) else "").strip().casefold(),
        (spec.col_labels[c] if c < len(spec.col_labels) else "").strip().casefold(),
    }
    covered.discard("")
    parts = [p for p in label.split(" · ") if p.strip().casefold() not in covered]
    return " · ".join(parts)


def _roi_key(scope: str, panel: PanelRef, index: int, cols: int) -> str:
    if scope == "image":
        return f"{panel.dataset}:{panel.image}"
    if scope == "column":
        return f"col:{index % cols}"
    if scope == "panel":
        return f"panel:{index}"
    return "figure"


def _unavailable_indices(panels: list[PanelRef], cols: int) -> set[int]:
    """Empty cells that stand for a row x column combination with no inference.

    Three signals, in priority order. `panel.missing` is an explicit user
    override (the toggle on an empty cell) and always wins. Otherwise a cell
    that kept its identity is unavailable — that is what `autofill` emits when
    no run exists for a (model, num_samples) pair. Failing that, position
    decides: an empty cell whose row *and* column both hold predictions is a
    hole inside the populated block, whereas the blank left over beside an
    input/GT column sits in a column with no predictions at all and stays a
    plain gap. The positional rule is what catches hand-built grids and cells
    that lost their identity to a clear or a drag.

    Mirrors `unavailableIndices` in lib/figureSpec.ts.
    """
    pred_rows = {i // cols for i, p in enumerate(panels) if p.kind == "pred"}
    pred_cols = {i % cols for i, p in enumerate(panels) if p.kind == "pred"}
    out: set[int] = set()
    for i, p in enumerate(panels):
        if p.kind != "empty":
            continue
        if p.missing is not None:
            if p.missing:
                out.add(i)
            continue
        if p.model_type or p.num_samples is not None:
            out.add(i)
        elif (i // cols) in pred_rows and (i % cols) in pred_cols:
            out.add(i)
    return out


def _draw_missing(ax, spec: GridFigureSpec, aspect: float) -> None:
    """Boxed diagonal cross marking a combination that has no inference.

    Drawn in axes-fraction coordinates so it needs no image and no data limits,
    with `clip_on=False` so the border strokes are not halved at the axes edge.
    `aspect` (height/width) keeps the box the same shape as the real panels
    around it, which for a non-square dataset like DRIVE is visible.
    """
    ps = spec.panel_style
    color = ps.missing_color or spec.label_color
    lw = ps.missing_width
    ax.set_axis_off()
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(aspect)
    ax.add_patch(
        Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                  edgecolor=color, linewidth=lw, clip_on=False, zorder=3)
    )
    for xs, ys in (((0, 1), (0, 1)), ((0, 1), (1, 0))):
        ax.plot(xs, ys, transform=ax.transAxes, color=color,
                linewidth=lw, clip_on=False, zorder=3, solid_capstyle="butt")


def _pad_panels(panels: list[PanelRef], n: int) -> list[PanelRef]:
    out = list(panels[:n])
    while len(out) < n:
        out.append(PanelRef(kind="empty"))
    return out


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
class _Loader:
    """Per-render decode cache.

    At most two `imagesvc.load_array` calls per panel: one at native size
    (to probe dimensions for the shared-canvas computation) and, only if the
    panel actually needs resampling, one at the target canvas size. Scoped to
    a single `render_grid` call -- a module-level cache would risk unbounded
    memory across a long editing session, and a bounded `lru_cache` buys
    nothing here since nothing is reused between requests.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, Optional[tuple[int, int]]], np.ndarray] = {}

    def load(self, path: Path, kind: str, size: Optional[tuple[int, int]]) -> np.ndarray:
        key = (str(path), size)
        arr = self._cache.get(key)
        if arr is None:
            arr = imagesvc.load_array(path, kind, size=size)
            self._cache[key] = arr
        return arr


def _resolve_all(
    panels: list[PanelRef], loader: _Loader
) -> tuple[dict[int, Path], dict[int, tuple[int, int]], dict[int, str]]:
    """First pass: resolve + probe every non-empty panel. Never raises.

    Returns (index -> resolved path, index -> native (width, height),
    index -> error message for panels that could not be resolved/decoded).
    """
    resolved: dict[int, Path] = {}
    native: dict[int, tuple[int, int]] = {}
    errors: dict[int, str] = {}
    for i, p in enumerate(panels):
        if p.kind == "empty":
            continue
        try:
            path = gallery.resolve_panel(
                p.kind, p.dataset, p.image, stage=p.stage, experiment=p.experiment, run_name=p.run_name
            )
            arr = loader.load(path, p.kind, None)
            h, w = arr.shape[:2]
            resolved[i] = path
            native[i] = (w, h)
        except Exception as exc:  # noqa: BLE001 - one bad panel must not kill the figure
            errors[i] = str(exc)
    return resolved, native, errors


def _canvas_sizes(
    native: dict[int, tuple[int, int]], keys: list[str]
) -> dict[str, tuple[int, int]]:
    """Largest (width, height) among the panels sharing each ROI-group key.

    Every panel in a group gets resampled up to this size before drawing, so
    normalized ROI coordinates land on identical pixels across panels that
    were decoded at different native resolutions.
    """
    canvas: dict[str, tuple[int, int]] = {}
    for i, (w, h) in native.items():
        key = keys[i]
        cw, ch = canvas.get(key, (0, 0))
        canvas[key] = (max(cw, w), max(ch, h))
    return canvas


# --------------------------------------------------------------------------- #
# ROI drawing
# --------------------------------------------------------------------------- #
def _clamped(roi: RoiSpec) -> tuple[float, float, float, float]:
    return roi.x, roi.y, max(roi.w, _MIN_ROI), max(roi.h, _MIN_ROI)


def _apply_crop(ax, arr: np.ndarray, crop: CropSpec) -> None:
    """Narrow the view to `crop` without touching the array.

    Runs before `_draw_roi`, so an ROI marker or inset lands inside the zoomed
    frame. `set_xlim`/`set_ylim` is exact where slicing would round, and the
    inverted y matches imshow's origin="upper".
    """
    h, w = arr.shape[:2]
    x, y = crop.x, crop.y
    cw, ch = max(crop.w, _MIN_ROI), max(crop.h, _MIN_ROI)
    ax.set_xlim(x * w, (x + cw) * w)
    ax.set_ylim((y + ch) * h, y * h)


def _draw_roi(ax, arr: np.ndarray, roi: RoiSpec) -> None:
    h, w = arr.shape[:2]
    x, y, rw, rh = _clamped(roi)

    if roi.mode == "crop":
        # Full array stays; just narrow the view. set_xlim/set_ylim is exact
        # (no off-by-one from slicing), and the inverted y matches imshow's
        # origin="upper" convention.
        ax.set_xlim(x * w, (x + rw) * w)
        ax.set_ylim((y + rh) * h, y * h)
        return

    if roi.mode == "inset":
        bounds = _INSET_BOUNDS[roi.inset_corner](min(max(roi.inset_scale, 0.05), 0.95))
        axins = ax.inset_axes(bounds)
        cmap = "gray" if arr.ndim == 2 else None
        axins.imshow(arr, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
        axins.set_xlim(x * w, (x + rw) * w)
        axins.set_ylim((y + rh) * h, y * h)
        axins.set_xticks([])
        axins.set_yticks([])
        for spine in axins.spines.values():
            spine.set_edgecolor(roi.color)
            spine.set_linewidth(roi.linewidth)
        if roi.inset_connectors:
            ax.indicate_inset_zoom(axins, edgecolor=roi.color)
        else:
            ax.add_patch(
                Rectangle((x * w, y * h), rw * w, rh * h, fill=False,
                          edgecolor=roi.color, linewidth=roi.linewidth)
            )
        return

    # "marker" (default)
    ax.add_patch(
        Rectangle((x * w, y * h), rw * w, rh * h, fill=False,
                  edgecolor=roi.color, linewidth=roi.linewidth)
    )


# --------------------------------------------------------------------------- #
# Per-cell rendering
# --------------------------------------------------------------------------- #
def _render_cell(
    ax,
    panel: PanelRef,
    idx: int,
    key: str,
    canvas: dict[str, tuple[int, int]],
    resolved: dict[int, Path],
    native: dict[int, tuple[int, int]],
    errors: dict[int, str],
    loader: _Loader,
    spec: GridFigureSpec,
    unavailable: set[int],
) -> None:
    # Aspect of this cell's ROI group, so a placeholder matches the panels it
    # sits among. An empty cell has no group of its own, so it borrows the
    # figure's prevailing aspect instead (1.0 when nothing resolved at all).
    cw, ch = canvas.get(key, (0, 0))
    if not cw and canvas:
        cw, ch = next(iter(canvas.values()))
    aspect = (ch / cw) if cw else 1.0

    if panel.kind == "empty":
        if spec.panel_style.missing_mark and idx in unavailable:
            _draw_missing(ax, spec, aspect)
        else:
            ax.set_axis_off()
        return

    if idx in errors:
        # Resolved to nothing on disk — same visual as an absent combination,
        # since either way the figure has no prediction to show here.
        if spec.panel_style.missing_mark:
            _draw_missing(ax, spec, aspect)
        else:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center",
                    color=_MISSING_COLOR, fontsize=8, style="italic")
        return

    path = resolved[idx]
    native_wh = native[idx]
    target = canvas.get(key, native_wh)
    size = None if target == native_wh else target
    try:
        arr = loader.load(path, panel.kind, size)
    except Exception:  # noqa: BLE001 - decode failed after the probe succeeded (rare); degrade, don't raise
        if spec.panel_style.missing_mark:
            _draw_missing(ax, spec, aspect)
        else:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center",
                    color=_MISSING_COLOR, fontsize=8, style="italic")
        return

    if spec.panel_style.invert and gallery.is_mask(panel.kind):
        arr = 255 - arr

    cmap = "gray" if arr.ndim == 2 else None
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")

    crop = spec.crops.get(key)
    if crop is not None:
        _apply_crop(ax, arr, crop)

    roi = spec.rois.get(key)
    if roi is not None:
        # An ROI in "crop" mode sets the view itself, so it deliberately
        # supersedes the pre-crop rather than compounding with it.
        _draw_roi(ax, arr, roi)

    ps = spec.panel_style
    if ps.show_label and ps.label_loc in ("top", "bottom"):
        text = _cell_label(panel, idx, spec)
        if text:
            y = 0.97 if ps.label_loc == "top" else 0.03
            va = "top" if ps.label_loc == "top" else "bottom"
            ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va=va,
                    color=spec.label_color, fontsize=spec.label_fontsize,
                    bbox=dict(facecolor="black", alpha=0.45, pad=1.5, edgecolor="none"))

    if ps.show_metric and panel.score is not None:
        x, y, ha, va = _METRIC_LOC.get(ps.metric_loc, _METRIC_LOC["lower left"])
        text = f"{panel.score:.{ps.decimals}f}"
        alpha = min(max(ps.metric_bg_alpha, 0.0), 1.0)
        # alpha 0 still paints a fully transparent patch in vector output, which
        # leaves an invisible-but-selectable box in the SVG; drop it instead.
        bbox = (dict(facecolor=ps.metric_bg_color, alpha=alpha, pad=1.5, edgecolor="none")
                if alpha > 0 else None)
        ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                color=ps.metric_color, fontsize=ps.metric_fontsize, bbox=bbox)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _panel_slug(panel: PanelRef, index: int, spec: GridFigureSpec) -> str:
    """Filesystem-safe stem for one panel, unique within a figure.

    Two panels can carry the same kind and image (the same GT shown twice), so
    the row/column is always part of the name — a silent overwrite in a batch
    export would be worse than a slightly longer filename.
    """
    r, c = divmod(index, spec.cols)
    parts = [panel.dataset, panel.image, panel.kind]
    if panel.kind == "pred":
        parts += [panel.model_type or "", _shot(panel.num_samples)]
    stem = _slug("_".join(p for p in parts if p)) or "panel"
    return f"r{r + 1}c{c + 1}_{stem}"


def _shot(num_samples: Optional[int]) -> str:
    if num_samples is None:
        return ""
    return "0-shot" if num_samples == 0 else f"{num_samples}-shot"


def _single_panel_spec(spec: GridFigureSpec, panel: PanelRef, roi: Optional[RoiSpec],
                       crop: Optional[CropSpec]) -> GridFigureSpec:
    """A 1x1 copy of `spec` holding just one panel, with its ROI carried over.

    The ROI is re-keyed to the "figure" scope: a single-panel grid recomputes
    group keys from position (`col:0`, `panel:0`), which would no longer match
    the key the panel had in the full grid, so the marking would silently
    vanish under any scope other than "image".

    Row/column headers, the panel caption and the cell border are dropped — all
    three exist to place a cell inside a grid, and a standalone image has no
    neighbours to be separated from. The border in particular would come out as
    a frame drawn around the exported picture.
    """
    single = spec.model_copy(deep=True)
    single.rows = 1
    single.cols = 1
    single.panels = [panel]
    single.row_labels = []
    single.col_labels = []
    single.title.text = ""
    single.roi_scope = "figure"
    single.rois = {"figure": roi} if roi is not None else {}
    single.crops = {"figure": crop} if crop is not None else {}
    single.panel_style.show_label = False
    single.panel_style.border = False
    return single


_Candidate = tuple[str, PanelRef, Optional[RoiSpec], Optional[CropSpec], Optional[tuple[int, int]]]


def _reference_panels(spec: GridFigureSpec,
                      canvas: dict[str, tuple[int, int]]) -> list[_Candidate]:
    """One input and one ground truth per distinct image in the figure.

    These are *synthesized*, not looked up: a qualitative figure often shows
    only predictions, so requiring an input/GT cell to already sit in the grid
    would make the reference export silently produce nothing.

    Geometry is inherited from the grid: the crop and the ROI are taken from the
    first panel of that image that defines one, scanning the image's own
    input/GT cells before its predictions. Both are searched independently,
    since a figure may zoom without marking or mark without zooming. Falling
    through every panel of the image matters under the per-column and per-panel
    scopes, where the reference cell carries no geometry of its own but the
    prediction the figure is actually about does — without it a zoomed figure
    would ship un-zoomed reference frames.

    `canvas` is the shared drawing size per image, so the exported frame keeps
    the aspect its cells have in the grid.
    """
    order: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for i, panel in enumerate(spec.panels[: spec.rows * spec.cols]):
        if panel.kind == "empty" or not panel.dataset or not panel.image:
            continue
        rank = 0 if panel.kind in ("input", "gt") else 1
        group = _roi_key(spec.roi_scope, panel, i, spec.cols)
        order.setdefault((panel.dataset, panel.image), []).append((rank, group))

    out: list[_Candidate] = []
    for (dataset, image), entries in order.items():
        keys = [k for _, k in sorted(entries, key=lambda e: e[0])]  # stable: grid order within a rank
        roi = next((spec.rois[k] for k in keys if k in spec.rois), None)
        crop = next((spec.crops[k] for k in keys if k in spec.crops), None)
        for kind in ("input", "gt"):
            stem = _slug(f"{dataset}_{image}_{kind}")
            out.append((stem, PanelRef(kind=kind, dataset=dataset, image=image), roi, crop,
                        canvas.get(f"{dataset}:{image}")))
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-_")


def _dedupe(name: str, used: set[str]) -> str:
    """Add a numeric suffix rather than let one export overwrite another.

    Panel labels are not unique — two images both yield "Input", and the same
    model appears at several num_samples — so short naming needs this or files
    silently disappear.
    """
    if name not in used:
        used.add(name)
        return name
    n = 2
    while f"{name}_{n}" in used:
        n += 1
    used.add(f"{name}_{n}")
    return f"{name}_{n}"


def export_panels(
    spec: GridFigureSpec, which: str = "refs", naming: str = "full"
) -> tuple[list[tuple[str, bytes, str]], list[str]]:
    """Render selected panels as standalone figures.

    `refs` gives the input and ground truth behind every image in the figure —
    the reference frames a caption points at. `all` adds every prediction cell
    on top of those; predictions only, because the references are already
    covered and an input/GT cell sitting in the grid would otherwise be exported
    twice under two names. Both go through `render_grid`, so the ROI, the
    crop/inset mode, inversion and the export format behave exactly as they do
    inside the grid.

    `naming` picks the filename: "full" spells out dataset, image and run
    identity (unambiguous across figures, but long), "label" uses the panel's
    own caption — "VSUNet18_20-shot", "Input", "Ground-truth" — which is what
    you want when the files go straight into one figure's subfolder.

    Returns (files, skipped): a source image missing from disk is skipped and
    named rather than exported as an X placeholder, which would look like a
    result instead of a broken dataset path.
    """
    ext = _EXT_NAME[spec.export.format]
    grid_panels = _pad_panels(spec.panels, spec.rows * spec.cols)

    # The grid resamples each ROI group onto a shared canvas before drawing, and
    # that canvas decides the panel's aspect. Recovering it here (one size probe
    # per source file) is what keeps an exported panel the same shape as its
    # cell. References go by image rather than by group key: they stand for the
    # whole image, not for one column or one cell of it.
    loader = _Loader()
    _, native, _ = _resolve_all(grid_panels, loader)
    group_canvas = _canvas_sizes(
        native, [_roi_key(spec.roi_scope, p, i, spec.cols) for i, p in enumerate(grid_panels)])
    image_canvas = _canvas_sizes(native, [f"{p.dataset}:{p.image}" for p in grid_panels])

    candidates = _reference_panels(spec, image_canvas)
    if which != "refs":
        candidates += [
            (_panel_slug(p, i, spec), p,
             spec.rois.get(_roi_key(spec.roi_scope, p, i, spec.cols)),
             spec.crops.get(_roi_key(spec.roi_scope, p, i, spec.cols)),
             group_canvas.get(_roi_key(spec.roi_scope, p, i, spec.cols)))
            for i, p in enumerate(grid_panels)
            if p.kind == "pred"
        ]

    # Every panel renders on a canvas the size of one grid cell, so the files
    # drop into a layout beside the figure at the same scale as the cells.
    cell = _cell_size(spec)

    out: list[tuple[str, bytes, str]] = []
    skipped: list[str] = []
    used: set[str] = set()
    for stem, panel, roi, crop, canvas in candidates:
        try:
            gallery.resolve_panel(
                panel.kind, panel.dataset, panel.image,
                stage=panel.stage, experiment=panel.experiment, run_name=panel.run_name,
            ).stat()
        except Exception:  # noqa: BLE001 - unresolvable or absent; both mean "nothing to export"
            skipped.append(stem)
            continue
        name = _dedupe(_slug(_panel_label(panel)) or stem, used) if naming == "label" else stem
        # No labels to leave room for, so trim the tight-bbox margin too --
        # otherwise the picture ships with a band of background around it.
        content, mime = render_grid(_single_panel_spec(spec, panel, roi, crop),
                                    pad_inches=0.0, axes_size=cell, canvas_size=canvas)
        out.append((f"{name}.{ext}", content, mime))
    return out, skipped


def _figure_size(spec: GridFigureSpec) -> tuple[float, float]:
    """Canvas in inches: one `panel_size` square per cell plus room for headers."""
    extra_top = 0.0
    if any(spec.col_labels):
        extra_top += 0.38
    if spec.title.text:
        extra_top += 0.5
    extra_left = 0.55 if any(spec.row_labels) else 0.0
    return (max(spec.cols * spec.panel_size + extra_left, 0.5),
            max(spec.rows * spec.panel_size + extra_top, 0.5))


def _cell_size(spec: GridFigureSpec) -> tuple[float, float]:
    """Inches occupied by one cell of `spec`'s grid, headers and gaps accounted for.

    Repeats matplotlib's own gridspec arithmetic: the default subplot margins
    take their cut of the canvas first, and wspace/hspace are fractions of a
    single cell, so n cells and n-1 gaps share what is left. A standalone panel
    export renders on a canvas of exactly this size, which is what makes the
    exported file come out the same size as the cell it was lifted from — the
    margins alone differ enough between a 3x4 grid and a 1x1 one to shrink it by
    a percent or two otherwise.
    """
    fig_w, fig_h = _figure_size(spec)
    rc = plt.rcParams
    span_w = (rc["figure.subplot.right"] - rc["figure.subplot.left"]) * fig_w
    span_h = (rc["figure.subplot.top"] - rc["figure.subplot.bottom"]) * fig_h
    cols, rows = max(spec.cols, 1), max(spec.rows, 1)
    return (max(span_w / (cols + spec.wspace * (cols - 1)), 0.1),
            max(span_h / (rows + spec.hspace * (rows - 1)), 0.1))


def render_grid(spec: GridFigureSpec, pad_inches: Optional[float] = None,
                axes_size: Optional[tuple[float, float]] = None,
                canvas_size: Optional[tuple[int, int]] = None) -> tuple[bytes, str]:
    rows, cols = spec.rows, spec.cols
    n = rows * cols
    panels = _pad_panels(spec.panels, n)
    keys = [_roi_key(spec.roi_scope, p, i, cols) for i, p in enumerate(panels)]

    loader = _Loader()
    resolved, native, errors = _resolve_all(panels, loader)
    # `canvas_size` imposes the shared canvas a larger figure computed: a panel
    # pulled out on its own would otherwise be sized from its own resolution and
    # take a different aspect than the cell it came from (a DRIVE prediction is
    # square at 256x256, the cell it sits in is 565x584).
    canvas = dict.fromkeys(keys, canvas_size) if canvas_size is not None else _canvas_sizes(native, keys)
    unavailable = _unavailable_indices(panels, cols)

    row_labels, col_labels = spec.row_labels, spec.col_labels
    # `axes_size` asks for a canvas that *is* the drawing area: the caller has
    # already sized the cell and wants no margin around it (see `_cell_size`).
    fig_w, fig_h = axes_size if axes_size is not None else _figure_size(spec)

    fig = None
    try:
        with plt.rc_context({"svg.fonttype": "none"}):
            fig, axes = plt.subplots(
                rows, cols, figsize=(fig_w, fig_h), squeeze=False,
                gridspec_kw={"wspace": spec.wspace, "hspace": spec.hspace},
                facecolor=spec.background,
            )
            if axes_size is not None:
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            for r in range(rows):
                for c in range(cols):
                    idx = r * cols + c
                    panel = panels[idx]
                    ax = axes[r][c]

                    ax.set_facecolor(spec.background)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    border_on = spec.panel_style.border
                    for spine in ax.spines.values():
                        spine.set_visible(border_on)
                        if border_on:
                            spine.set_edgecolor(spec.panel_style.border_color)
                            spine.set_linewidth(spec.panel_style.border_width)

                    _render_cell(ax, panel, idx, keys[idx], canvas, resolved, native, errors,
                                 loader, spec, unavailable)

                    # Row label: ax.text (not set_ylabel -- axis('off') on an
                    # empty leftmost cell silently drops the ylabel but never
                    # a plain text artist; verified against this mpl build).
                    if c == 0 and r < len(row_labels) and row_labels[r]:
                        ax.text(-0.10, 0.5, row_labels[r], transform=ax.transAxes,
                                rotation=90, ha="right", va="center", clip_on=False,
                                color=spec.label_color, fontsize=spec.row_label_fontsize)

                    # Column label: set_title survives axis('off') too (title
                    # belongs to the Axes, not to xaxis/yaxis).
                    if r == 0 and c < len(col_labels) and col_labels[c]:
                        ax.set_title(col_labels[c], color=spec.label_color,
                                     fontsize=spec.col_label_fontsize)

            if spec.title.text:
                fig.suptitle(spec.title.text, fontsize=spec.title.fontsize, color=spec.label_color)

            fmt = spec.export.format
            save_fmt = "jpeg" if fmt == "jpg" else fmt
            buf = io.BytesIO()
            extra = {} if pad_inches is None else {"pad_inches": pad_inches}
            fig.savefig(buf, format=save_fmt, dpi=spec.export.dpi,
                        facecolor=spec.background, bbox_inches="tight", **extra)
            return buf.getvalue(), _MIME[fmt]
    finally:
        if fig is not None:
            plt.close(fig)
