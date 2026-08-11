"""Figure Studio — composed grid rendering, export and saved layouts."""
from __future__ import annotations

import io
import os
import zipfile
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from .. import figuregrid, gridstore
from ..schemas import GridFigureSpec

router = APIRouter(prefix="/api/figure", tags=["figure-grid"])

_EXT = {"svg": "svg", "png": "png", "jpg": "jpg", "pdf": "pdf"}


class LayoutIn(BaseModel):
    name: str = "Untitled layout"
    spec: dict = {}


class LayoutPatch(BaseModel):
    name: Optional[str] = None
    spec: Optional[dict] = None


@router.post("/render")
def render(spec: GridFigureSpec):
    try:
        content, mime = figuregrid.render_grid(spec)
    except Exception as exc:  # surface render errors to the UI
        raise HTTPException(status_code=400, detail=f"Render failed: {exc}") from exc
    return Response(content=content, media_type=mime)


@router.post("/export")
def export(spec: GridFigureSpec):
    try:
        content, mime = figuregrid.render_grid(spec)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Render failed: {exc}") from exc

    save_path = spec.export.save_path
    if save_path:
        path = os.path.expanduser(save_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(content)
        return {"saved": True, "path": path, "bytes": len(content)}

    fname = f"figure.{_EXT[spec.export.format]}"
    return Response(content=content, media_type=mime,
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@router.post("/export/panels")
def export_panels(
    spec: GridFigureSpec,
    which: Literal["refs", "all"] = Query("refs"),
    naming: Literal["full", "label"] = Query("full"),
):
    """Each selected panel as its own file — the input and ground truth with the
    ROI drawn on them, so a caption can point at the reference frames separately
    from the grid. Writes beside `export.save_path` when set, otherwise streams
    a zip, mirroring how `/export` chooses between disk and download."""
    try:
        items, skipped = figuregrid.export_panels(spec, which, naming)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Render failed: {exc}") from exc

    if not items:
        detail = "This grid has no panels to export."
        if skipped:
            detail = (
                f"None of the {len(skipped)} source file(s) could be found "
                f"({', '.join(skipped[:3])}{'…' if len(skipped) > 3 else ''}). "
                "Check the dataset folder under Dataset source."
            )
        raise HTTPException(status_code=400, detail=detail)

    save_path = spec.export.save_path
    if save_path:
        # `save_path` names the grid file; siblings take its stem as a prefix so
        # one export lands as figure.svg + figure_r1c1_dca1_31_input.svg, ...
        # Panel-label naming drops the prefix — the whole point of asking for it
        # is a short "VSUNet18_20-shot.svg", not the figure name repeated.
        target = os.path.expanduser(save_path)
        directory = os.path.dirname(target) or "."
        prefix = "" if naming == "label" else os.path.splitext(os.path.basename(target))[0]
        os.makedirs(directory, exist_ok=True)
        written = []
        for name, content, _ in items:
            out = os.path.join(directory, f"{prefix}_{name}" if prefix else name)
            with open(out, "wb") as fh:
                fh.write(content)
            written.append(out)
        return {"saved": True, "paths": written, "count": len(written), "skipped": skipped}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content, _ in items:
            zf.writestr(name, content)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="figure_panels.zip"'},
    )


@router.get("/layouts")
def list_layouts():
    return gridstore.list_layouts()


@router.post("/layouts")
def create_layout(body: LayoutIn):
    return gridstore.create_layout(body.name, body.spec)


@router.put("/layouts/{layout_id}")
def update_layout(layout_id: str, body: LayoutPatch):
    record = gridstore.update_layout(layout_id, body.model_dump(exclude_unset=True))
    if record is None:
        raise HTTPException(status_code=404, detail="layout not found")
    return record


@router.delete("/layouts/{layout_id}")
def delete_layout(layout_id: str):
    if not gridstore.delete_layout(layout_id):
        raise HTTPException(status_code=404, detail="layout not found")
    return {}
