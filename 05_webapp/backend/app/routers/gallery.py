"""Figure Studio — options and metric-driven ranking of inferences."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import gallery, paths
from ..schemas import AutoFillRequest, RankImagesRequest, RankRunsRequest

router = APIRouter(prefix="/api/figure", tags=["figure"])


class DatasetRootIn(BaseModel):
    # None/"" restores the built-in default (or $VESSLAB_DATASET_ROOT).
    dataset_root: Optional[str] = None


def _settings_payload() -> dict:
    """Where inputs/GT are read from, plus a per-dataset probe.

    Predictions live in the repo and are always reachable; only the source
    images and ground truth depend on the external drive, so the probe reports
    exactly what a wrong path would break.
    """
    root = paths.dataset_root()
    sources = {}
    for name, cfg in gallery.DATASET_SOURCES.items():
        sub = root / cfg["root"]
        sources[name] = {"path": str(sub), "found": sub.is_dir()}
    return {
        "dataset_root": str(root),
        "default_root": str(paths.DATASET_ROOT_DEFAULT),
        "is_default": root == paths.DATASET_ROOT_DEFAULT,
        "found": root.is_dir(),
        "sources": sources,
    }


@router.get("/options")
def get_options():
    try:
        return gallery.options()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Options failed: {exc}") from exc


@router.post("/rank/images")
def rank_images(req: RankImagesRequest):
    try:
        return gallery.rank_images(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Rank images failed: {exc}") from exc


@router.post("/rank/runs")
def rank_runs(req: RankRunsRequest):
    try:
        return gallery.rank_runs(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Rank runs failed: {exc}") from exc


@router.post("/autofill")
def autofill(req: AutoFillRequest):
    try:
        return gallery.autofill(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Autofill failed: {exc}") from exc


@router.get("/settings")
def get_settings():
    return _settings_payload()


@router.put("/settings")
def put_settings(body: DatasetRootIn):
    value = (body.dataset_root or "").strip() or None
    paths.set_dataset_root(value)
    return _settings_payload()
