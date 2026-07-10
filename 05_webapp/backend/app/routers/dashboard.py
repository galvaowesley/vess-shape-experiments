"""Figure-comparison dashboard endpoints (multiple named dashboards)."""
from __future__ import annotations

import os
import re
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from .. import plotting, stats, store, tables
from ..schemas import (
    DashboardCreate,
    DashboardExportRequest,
    DashboardItem,
    DashboardItemUpdate,
    DashboardRename,
    PlotSpec,
    ReorderRequest,
    TableSpec,
    WilcoxonSpec,
)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_") or "figure"


# --------------------------------------------------------------------------- #
# Dashboards (named collections)
# --------------------------------------------------------------------------- #
@router.get("/dashboards")
def list_dashboards():
    return store.list_dashboards()


@router.post("/dashboards")
def create_dashboard(req: DashboardCreate):
    return store.create_dashboard(req.name)


@router.patch("/dashboards/{dashboard_id}")
def rename_dashboard(dashboard_id: str, req: DashboardRename):
    d = store.rename_dashboard(dashboard_id, req.name)
    if d is None:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return d


@router.delete("/dashboards/{dashboard_id}")
def delete_dashboard(dashboard_id: str):
    if not store.delete_dashboard(dashboard_id):
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return {"deleted": True, "id": dashboard_id}


# --------------------------------------------------------------------------- #
# Items (pinned figures)
# --------------------------------------------------------------------------- #
@router.get("")
def list_items(dashboard_id: Optional[str] = Query(None)):
    return store.list_items(dashboard_id)


@router.post("")
def add_item(item: DashboardItem):
    return store.add_item(item)


@router.put("/{item_id}")
def update_item(item_id: str, patch: DashboardItemUpdate):
    updated = store.update_item(item_id, patch.model_dump(exclude_none=True))
    if updated is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return updated


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
    for item in store.list_items(req.dashboard_id):
        spec_dict = dict(item.get("spec", {}))
        kind = item.get("kind")
        try:
            if kind == "table":  # tables export as Markdown, not an image
                content = tables.render_table(TableSpec(**spec_dict))["markdown"].encode("utf-8")
                ext = "md"
            elif kind == "wilcoxon":
                spec_dict["export"] = {**spec_dict.get("export", {}),
                                       "format": req.format, "dpi": req.dpi, "save_path": None}
                content, _ = stats.render_wilcoxon(WilcoxonSpec(**spec_dict))
                ext = req.format
            else:
                spec_dict["export"] = {**spec_dict.get("export", {}),
                                       "format": req.format, "dpi": req.dpi, "save_path": None}
                content, _ = plotting.render_figure(PlotSpec(**spec_dict))
                ext = req.format
        except Exception as exc:  # skip a broken item, keep going
            written.append({"id": item.get("id"), "error": str(exc)})
            continue
        fname = f"{_slug(item.get('title', 'figure'))}_{item.get('id', '')}.{ext}"
        path = os.path.join(out_dir, fname)
        with open(path, "wb") as fh:
            fh.write(content)
        written.append({"id": item.get("id"), "path": path})
    return {"out_dir": out_dir, "written": written}
