"""Local filesystem browser backing the "Browse..." destination pickers used
by every save/export panel in the UI.

This is a local, single-user tool: the browser and this backend always run on
the same machine, and several endpoints already let the frontend tell the
backend to write files at an arbitrary path (`export.save_path`, dashboard
`out_dir`, ...). Listing directories and creating folders sits at the same
trust level, so there is no extra auth/whitelisting here beyond basic error
handling — the point is a friendly cross-OS folder browser, not a sandbox.
"""
from __future__ import annotations

import os
import string
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .. import paths

router = APIRouter(prefix="/api/fs", tags=["fs"])


class FsEntry(BaseModel):
    name: str
    path: str
    is_dir: bool


class FsListing(BaseModel):
    path: str
    parent: Optional[str] = None
    entries: list[FsEntry]
    favorites: list[FsEntry]


class MkdirIn(BaseModel):
    path: str
    name: str


class WriteIn(BaseModel):
    path: str
    content: str


def _favorites() -> list[FsEntry]:
    out = [FsEntry(name="Home", path=str(Path.home()), is_dir=True)]
    seen = {out[0].path}
    for name, p in (("Repository", paths.REPO_ROOT), ("Dataset source", paths.dataset_root())):
        sp = str(p)
        if sp not in seen and p.is_dir():
            out.append(FsEntry(name=name, path=sp, is_dir=True))
            seen.add(sp)
    if os.name == "nt":
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                out.append(FsEntry(name=drive, path=drive, is_dir=True))
    else:
        out.append(FsEntry(name="/", path="/", is_dir=True))
    return out


def _resolve_existing(raw: Optional[str]) -> Path:
    """Best-effort resolve: falls back to the nearest existing ancestor, then
    home. Lets the UI navigate to a path typed before its last segment (or the
    whole thing) exists yet, instead of just erroring."""
    target = Path(raw).expanduser() if raw else Path.home()
    try:
        target = target.resolve()
    except OSError:
        pass
    if target.exists():
        return target.parent if target.is_file() else target
    for ancestor in target.parents:
        if ancestor.exists():
            return ancestor
    return Path.home()


@router.get("/browse", response_model=FsListing)
def browse(path: Optional[str] = Query(None)):
    target = _resolve_existing(path)
    try:
        children = list(target.iterdir())
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {target}") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Can't list {target}: {exc}") from exc

    entries = sorted(
        (
            FsEntry(name=c.name, path=str(c), is_dir=c.is_dir())
            for c in children
            if not c.name.startswith(".")
        ),
        key=lambda e: (not e.is_dir, e.name.lower()),
    )
    parent = str(target.parent) if target.parent != target else None
    return FsListing(path=str(target), parent=parent, entries=entries, favorites=_favorites())


@router.post("/mkdir", response_model=FsListing)
def mkdir(body: MkdirIn):
    name = body.name.strip()
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid folder name")
    parent = _resolve_existing(body.path)
    try:
        (parent / name).mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        raise HTTPException(status_code=400, detail="A file or folder with that name already exists") from None
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Couldn't create folder: {exc}") from exc
    return browse(str(parent))


@router.post("/write")
def write(body: WriteIn):
    target = Path(body.path).expanduser()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body.content, encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Couldn't save {target}: {exc}") from exc
    return {"saved": True, "path": str(target)}
