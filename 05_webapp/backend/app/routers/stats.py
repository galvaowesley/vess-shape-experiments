"""Wilcoxon signed-rank endpoints."""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Response

from .. import stats as stats_mod
from ..schemas import WilcoxonSpec

router = APIRouter(prefix="/api/stats/wilcoxon", tags=["stats"])

_EXT = {"svg": "svg", "png": "png", "jpg": "jpg", "pdf": "pdf"}


@router.post("/data")
def data(spec: WilcoxonSpec):
    try:
        return stats_mod.compute_panels(spec)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Compute failed: {exc}") from exc


@router.post("/render")
def render(spec: WilcoxonSpec):
    try:
        content, mime = stats_mod.render_wilcoxon(spec)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Render failed: {exc}") from exc
    return Response(content=content, media_type=mime)


@router.post("/export")
def export(spec: WilcoxonSpec):
    try:
        content, mime = stats_mod.render_wilcoxon(spec)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Render failed: {exc}") from exc
    if spec.export.save_path:
        path = os.path.expanduser(spec.export.save_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(content)
        return {"saved": True, "path": path, "bytes": len(content)}
    fname = f"wilcoxon.{_EXT[spec.export.format]}"
    return Response(content=content, media_type=mime,
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})
