"""Evaluation endpoints: metadata, palettes, render, export, rescan."""
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Query, Response

from .. import data, palettes, plotting
from ..schemas import PlotSpec

router = APIRouter(prefix="/api/eval", tags=["eval"])

_EXT = {"svg": "svg", "png": "png", "jpg": "jpg", "pdf": "pdf"}


@router.get("/metadata")
def metadata(source: str = Query("per_run"),
             datasets: Optional[str] = Query(None, description="comma-separated")):
    ds = [d for d in datasets.split(",") if d] if datasets else None
    return data.get_metadata(source, ds)


@router.get("/palettes")
def get_palettes():
    return palettes.catalog_with_swatches()


@router.post("/render")
def render(spec: PlotSpec):
    try:
        content, mime = plotting.render_figure(spec)
    except Exception as exc:  # surface render errors to the UI
        raise HTTPException(status_code=400, detail=f"Render failed: {exc}") from exc
    return Response(content=content, media_type=mime)


@router.post("/export")
def export(spec: PlotSpec):
    try:
        content, mime = plotting.render_figure(spec)
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


@router.post("/rescan")
def rescan(payload: dict = Body(default={})):
    datasets = payload.get("datasets")
    per_image = payload.get("per_image", True)
    try:
        return data.rescan(datasets, per_image=per_image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rescan failed: {exc}") from exc
