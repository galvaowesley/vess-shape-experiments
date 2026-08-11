"""Filesystem layout discovery and sys.path wiring.

The backend runs in the *same* Python environment as the experiments so it can
import `src.*` and `04_evaluation/utils_eval` directly. We never copy or modify
those modules; we only add their locations to `sys.path`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

# 05_webapp/backend/app/paths.py  ->  parents[3] == repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
WEBAPP_DIR: Path = Path(__file__).resolve().parents[2]      # 05_webapp/

EVAL_DIR: Path = REPO_ROOT / "04_evaluation"
RESULTS_DIR: Path = EVAL_DIR / "results"
ZERO_SHOT_DIR: Path = EVAL_DIR / "zero_shot"
FIGURES_DIR: Path = EVAL_DIR / "figures"
# Raw zero-shot artifacts: <variant>/inference_results_<dataset>/{inferences/*.png,metrics.csv}
ZERO_SHOT_RAW_DIR: Path = EVAL_DIR / "zero_shot_inferences"

# Source images + ground truth live outside the repo (external drive).
DATASET_ROOT_DEFAULT: Path = Path(
    os.environ.get("VESSLAB_DATASET_ROOT", "/media/wesleygalvao/1_TB_LINUX/Datasets/blood_vessels")
)

# Per-stage launcher locations (the two trainable stages).
STAGES: dict[str, dict[str, str]] = {
    "scratch": {
        "dir": str(REPO_ROOT / "02_few_shot_training_from_scratch"),
        "launcher": "run_serial_training.py",
        "label": "Stage 02 — From Scratch",
    },
    "finetune": {
        "dir": str(REPO_ROOT / "03_few_shot_fine-tuning"),
        "launcher": "run_serial_fine-tuning.py",
        "label": "Stage 03 — Fine-tuning",
    },
}

# App-private runtime state (gitignored). Never written into experiment folders.
DATA_DIR: Path = WEBAPP_DIR / ".data"
FIGURES_STORE: Path = DATA_DIR / "figures.json"
STYLE_STORE: Path = DATA_DIR / "style_prefs.json"
GRID_STORE: Path = DATA_DIR / "figure_grids.json"
SETTINGS_STORE: Path = DATA_DIR / "settings.json"
# Converted/resized panel images (TIFF -> PNG, thumbnails). Safe to delete anytime.
IMG_CACHE_DIR: Path = DATA_DIR / "imgcache"


# --------------------------------------------------------------------------- #
# Dataset root — configurable at runtime
# --------------------------------------------------------------------------- #
# The external drive holding source images and ground truth is not always
# mounted at the same place (and is absent entirely on another machine), so this
# is settable from the UI and persisted. Read it through `dataset_root()`, never
# as a module constant: a constant captured at import time would keep pointing
# at the old drive after the user changes it.
_dataset_root: Optional[Path] = None


def dataset_root() -> Path:
    global _dataset_root
    if _dataset_root is None:
        _dataset_root = DATASET_ROOT_DEFAULT
        try:
            saved = json.loads(SETTINGS_STORE.read_text()).get("dataset_root")
            if saved:
                _dataset_root = Path(saved).expanduser()
        except (OSError, ValueError):
            pass
    return _dataset_root


def set_dataset_root(value: Optional[str]) -> Path:
    """Point image/GT lookup at `value` (None or "" restores the default)."""
    global _dataset_root
    _dataset_root = Path(value).expanduser() if value else DATASET_ROOT_DEFAULT
    ensure_data_dir()
    data: dict = {}
    try:
        data = json.loads(SETTINGS_STORE.read_text())
    except (OSError, ValueError):
        pass
    data["dataset_root"] = None if not value else str(_dataset_root)
    SETTINGS_STORE.write_text(json.dumps(data, indent=2))
    return _dataset_root


def image_roots() -> tuple[Path, ...]:
    """Roots the image endpoint is allowed to read from. Anything resolving
    outside these is a 404 — the endpoint takes user-supplied path components."""
    return (REPO_ROOT, dataset_root())


def ensure_sys_path() -> None:
    """Make `import src.*`, `import utils_eval`, `import utils` resolvable."""
    for p in (str(REPO_ROOT), str(EVAL_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Wire paths on import so any module importing `paths` can rely on it.
ensure_sys_path()
