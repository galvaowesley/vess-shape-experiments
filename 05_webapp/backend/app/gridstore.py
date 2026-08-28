"""Persistence for saved Figure Studio grid layouts.

Mirrors `app/store.py`'s conventions (module-level `threading.Lock`, JSON
read/write helpers, `uuid.uuid4().hex[:12]` ids, ISO timestamps) but keeps a
flat list rather than store.py's dashboards-of-items shape, since a saved
layout is a standalone, independently named `GridFigureSpec` snapshot with no
grouping concept. A corrupt or missing `paths.GRID_STORE` file degrades to an
empty list, matching `store._read`'s tolerance for a broken JSON blob.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from . import paths

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _read() -> list[dict]:
    paths.ensure_data_dir()
    if not paths.GRID_STORE.exists():
        return []
    try:
        raw = json.loads(paths.GRID_STORE.read_text())
    except Exception:
        return []
    return raw if isinstance(raw, list) else []


def _write(layouts: list[dict]) -> None:
    paths.ensure_data_dir()
    paths.GRID_STORE.write_text(json.dumps(layouts, indent=2))


def list_layouts() -> list[dict]:
    with _lock:
        return _read()


def create_layout(name: str, spec: dict) -> dict:
    with _lock:
        layouts = _read()
        now = _now()
        record = {
            "id": _new_id(),
            "name": (name or "").strip() or "Untitled layout",
            "spec": spec or {},
            "created_at": now,
            "updated_at": now,
        }
        layouts.append(record)
        _write(layouts)
        return record


def update_layout(layout_id: str, patch: dict) -> Optional[dict]:
    """Partial update: only overwrite `name`/`spec` keys that are present."""
    with _lock:
        layouts = _read()
        for record in layouts:
            if record.get("id") == layout_id:
                if patch.get("name") is not None:
                    record["name"] = patch["name"].strip() or record["name"]
                if patch.get("spec") is not None:
                    record["spec"] = patch["spec"]
                record["updated_at"] = _now()
                _write(layouts)
                return record
        return None


def delete_layout(layout_id: str) -> bool:
    with _lock:
        layouts = _read()
        remaining = [record for record in layouts if record.get("id") != layout_id]
        if len(remaining) == len(layouts):
            return False
        _write(remaining)
        return True
