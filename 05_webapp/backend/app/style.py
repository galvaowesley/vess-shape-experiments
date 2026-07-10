"""Global style preferences (colors + order) shared by every chart.

Persisted as JSON under `05_webapp/.data/style_prefs.json` (same place as the
pinned figures). A single map keyed by *series value* (a model_type or a dataset
name) lets `plotting._resolve_colors` apply a consistent color everywhere, and
the per-field `order` lists drive series/legend and bar-category ordering. The
file is webapp-private — the experiment folders under `04_evaluation/` are never
touched.
"""
from __future__ import annotations

import json
import threading

from . import paths
from .schemas import StylePrefs

_lock = threading.Lock()


def _read() -> dict:
    paths.ensure_data_dir()
    if not paths.STYLE_STORE.exists():
        return {"colors": {}, "order": {}}
    try:
        raw = json.loads(paths.STYLE_STORE.read_text())
    except Exception:
        return {"colors": {}, "order": {}}
    return {"colors": raw.get("colors") or {}, "order": raw.get("order") or {}}


def _write(prefs: dict) -> None:
    paths.ensure_data_dir()
    paths.STYLE_STORE.write_text(json.dumps(prefs, indent=2))


def get_prefs() -> dict:
    with _lock:
        return _read()


def save_prefs(prefs: StylePrefs) -> dict:
    with _lock:
        record = {"colors": dict(prefs.colors), "order": dict(prefs.order)}
        _write(record)
        return record


def color_map() -> dict[str, str]:
    """name -> hex color (empty if nothing saved)."""
    return _read().get("colors", {})


def order_for(field: str) -> list[str]:
    """Saved display order for a categorical field (e.g. 'model_type', 'dataset')."""
    if not field:
        return []
    return list(_read().get("order", {}).get(field, []))
