"""Figure-comparison dashboard endpoints."""
from __future__ import annotations

import os
import re

from fastapi import APIRouter, HTTPException

from .. import plotting, stats, store
from ..schemas import (
    DashboardExportRequest,
    DashboardItem,
    PlotSpec,
    ReorderRequest,
    WilcoxonSpec,
)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_") or "figure"


@router.get("")
def list_items():
    return store.list_items()


@router.post("")
def add_item(item: DashboardItem):
    return store.add_item(item)


@router.delete("/{item_id}")
def delete_item(item_id: str):
    if not store.delete_item(item_id):
        raise HTTPException(status_code=404, detail="Item not found")
    return {"deleted": True, "id": item_id}


@router.patch("/reorder")
def reorder(req: ReorderRequest):
    return store.reorder(req.order)


@router.post("/export")
def export_all(req: DashboardExportRequest):
    out_dir = os.path.expanduser(req.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for item in store.list_items():
        spec_dict = dict(item.get("spec", {}))
        spec_dict.setdefault("export", {})
        spec_dict["export"] = {**spec_dict.get("export", {}),
                               "format": req.format, "dpi": req.dpi, "save_path": None}
        try:
            if item.get("kind") == "wilcoxon":
                content, _ = stats.render_wilcoxon(WilcoxonSpec(**spec_dict))
            else:
                content, _ = plotting.render_figure(PlotSpec(**spec_dict))
        except Exception as exc:  # skip a broken item, keep going
            written.append({"id": item.get("id"), "error": str(exc)})
            continue
        fname = f"{_slug(item.get('title', 'figure'))}_{item.get('id', '')}.{req.format}"
        path = os.path.join(out_dir, fname)
        with open(path, "wb") as fh:
            fh.write(content)
        written.append({"id": item.get("id"), "path": path})
    return {"out_dir": out_dir, "written": written}
