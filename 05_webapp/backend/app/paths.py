"""Filesystem layout discovery and sys.path wiring.

The backend runs in the *same* Python environment as the experiments so it can
import `src.*` and `04_evaluation/utils_eval` directly. We never copy or modify
those modules; we only add their locations to `sys.path`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# 05_webapp/backend/app/paths.py  ->  parents[3] == repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
WEBAPP_DIR: Path = Path(__file__).resolve().parents[2]      # 05_webapp/

EVAL_DIR: Path = REPO_ROOT / "04_evaluation"
RESULTS_DIR: Path = EVAL_DIR / "results"
ZERO_SHOT_DIR: Path = EVAL_DIR / "zero_shot"
FIGURES_DIR: Path = EVAL_DIR / "figures"

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


def ensure_sys_path() -> None:
    """Make `import src.*`, `import utils_eval`, `import utils` resolvable."""
    for p in (str(REPO_ROOT), str(EVAL_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Wire paths on import so any module importing `paths` can rely on it.
ensure_sys_path()
