"""Persistence for the figure-comparison dashboards.

Pinned figures live in `05_webapp/.data/figures.json`, now organized into one or
more *named dashboards*. Each item keeps the full spec so a dashboard can
re-render it (WYSIWYG) and the user can reopen it in the builder to edit it.

On-disk shape::

    {
      "dashboards": [{"id", "name", "created_at"}],
      "items":      [{"id", "dashboard_id", "title", "kind", "regime",
                      "spec", "created_at"}]
    }

The legacy flat-list format (a bare ``[ {item}, ... ]``) is migrated on read into
a single "Default" dashboard, so existing pins are preserved.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone

from . import paths
from .schemas import DashboardItem

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _blank() -> dict:
    return {"dashboards": [], "items": []}


def _read() -> dict:
    paths.ensure_data_dir()
    if not paths.FIGURES_STORE.exists():
        return _blank()
    try:
        raw = json.loads(paths.FIGURES_STORE.read_text())
    except Exception:
        return _blank()
    # Migrate the legacy flat list -> a single "Default" dashboard.
    if isinstance(raw, list):
        dash = {"id": _new_id(), "name": "Default", "created_at": _now()}
        for it in raw:
            it["dashboard_id"] = dash["id"]
        return {"dashboards": [dash], "items": raw}
    raw.setdefault("dashboards", [])
    raw.setdefault("items", [])
    return raw


def _write(state: dict) -> None:
    paths.ensure_data_dir()
    paths.FIGURES_STORE.write_text(json.dumps(state, indent=2))


def _ensure_default(state: dict) -> dict:
    """Return a dashboard to fall back on, creating "Default" if there is none."""
    if state["dashboards"]:
        return state["dashboards"][0]
    dash = {"id": _new_id(), "name": "Default", "created_at": _now()}
    state["dashboards"].append(dash)
    return dash


# --------------------------------------------------------------------------- #
# Dashboards
# --------------------------------------------------------------------------- #
def list_dashboards() -> list[dict]:
    with _lock:
        state = _read()
        counts: dict[str, int] = {}
        for it in state["items"]:
            counts[it.get("dashboard_id")] = counts.get(it.get("dashboard_id"), 0) + 1
        return [{**d, "count": counts.get(d["id"], 0)} for d in state["dashboards"]]


def create_dashboard(name: str) -> dict:
    with _lock:
        state = _read()
        dash = {"id": _new_id(), "name": name.strip() or "Untitled dashboard",
                "created_at": _now()}
        state["dashboards"].append(dash)
        _write(state)
        return {**dash, "count": 0}


def rename_dashboard(dashboard_id: str, name: str) -> dict | None:
    with _lock:
        state = _read()
        for d in state["dashboards"]:
            if d["id"] == dashboard_id:
                d["name"] = name.strip() or d["name"]
                _write(state)
                return d
        return None


def delete_dashboard(dashboard_id: str) -> bool:
    with _lock:
        state = _read()
        before = len(state["dashboards"])
        state["dashboards"] = [d for d in state["dashboards"] if d["id"] != dashboard_id]
        if len(state["dashboards"]) == before:
            return False
        state["items"] = [it for it in state["items"] if it.get("dashboard_id") != dashboard_id]
        _write(state)
        return True


# --------------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------------- #
def list_items(dashboard_id: str | None = None) -> list[dict]:
    with _lock:
        items = _read()["items"]
        if dashboard_id is None:
            return items
        return [it for it in items if it.get("dashboard_id") == dashboard_id]


def add_item(item: DashboardItem) -> dict:
    with _lock:
        state = _read()
        record = item.model_dump()
        record["id"] = record.get("id") or _new_id()
        record["created_at"] = _now()
        valid = {d["id"] for d in state["dashboards"]}
        if not record.get("dashboard_id") or record["dashboard_id"] not in valid:
            record["dashboard_id"] = _ensure_default(state)["id"]
        state["items"].append(record)
        _write(state)
        return record


def update_item(item_id: str, patch: dict) -> dict | None:
    """Replace the spec/title/regime of an existing pinned figure in place."""
    with _lock:
        state = _read()
        for it in state["items"]:
            if it.get("id") == item_id:
                for key in ("title", "regime", "spec", "kind", "dashboard_id"):
                    if key in patch and patch[key] is not None:
                        it[key] = patch[key]
                _write(state)
                return it
        return None


def delete_item(item_id: str) -> bool:
    with _lock:
        state = _read()
        new = [it for it in state["items"] if it.get("id") != item_id]
        if len(new) == len(state["items"]):
            return False
        state["items"] = new
        _write(state)
        return True


def reorder(order: list[str]) -> list[dict]:
    """Reorder the items whose ids appear in `order` (typically one dashboard),
    keeping every other item anchored in its original slot."""
    with _lock:
        state = _read()
        items = state["items"]
        by_id = {it["id"]: it for it in items if "id" in it}
        new_seq = iter([by_id[i] for i in order if i in by_id])
        order_set = set(order)
        result = [next(new_seq) if it.get("id") in order_set else it for it in items]
        state["items"] = result
        _write(state)
        return result
