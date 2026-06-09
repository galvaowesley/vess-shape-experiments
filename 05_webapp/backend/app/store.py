"""Persistence for the figure-comparison dashboard.

Pinned figures are stored as JSON under `05_webapp/.data/figures.json`. Each
item keeps the full spec so the dashboard can re-render it (WYSIWYG) and the
user can reopen it in the builder.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone

from . import paths
from .schemas import DashboardItem

_lock = threading.Lock()


def _read() -> list[dict]:
    paths.ensure_data_dir()
    if not paths.FIGURES_STORE.exists():
        return []
    try:
        return json.loads(paths.FIGURES_STORE.read_text())
    except Exception:
        return []


def _write(items: list[dict]) -> None:
    paths.ensure_data_dir()
    paths.FIGURES_STORE.write_text(json.dumps(items, indent=2))


def list_items() -> list[dict]:
    with _lock:
        return _read()


def add_item(item: DashboardItem) -> dict:
    with _lock:
        items = _read()
        record = item.model_dump()
        record["id"] = record.get("id") or uuid.uuid4().hex[:12]
        record["created_at"] = datetime.now(timezone.utc).isoformat()
        items.append(record)
        _write(items)
        return record


def delete_item(item_id: str) -> bool:
    with _lock:
        items = _read()
        new = [it for it in items if it.get("id") != item_id]
        _write(new)
        return len(new) != len(items)


def reorder(order: list[str]) -> list[dict]:
    with _lock:
        items = _read()
        index = {it["id"]: it for it in items if "id" in it}
        ordered = [index[i] for i in order if i in index]
        # keep any items not mentioned in `order` at the end
        ordered += [it for it in items if it.get("id") not in set(order)]
        _write(ordered)
        return ordered
