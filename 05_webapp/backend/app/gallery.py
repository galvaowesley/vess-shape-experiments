"""Inference gallery: on-disk path resolution + metric-driven ranking.

Every row of `04_evaluation/results/<ds>_all_results_per_image.csv` maps 1:1 onto
exactly one prediction PNG through a pure formula (verified across all datasets
and all three stages), so nothing here ever walks the 166k-file result tree — it
queries the consolidated CSVs and computes paths.

    scratch/finetune
        <root>/{02_few_shot_training_from_scratch|03_few_shot_fine-tuning}
              /{dataset}/experiments/{experiment}/{run_name}
              /inference_results/inferences/{image}.png
    zero_shot   (experiment == "__zero_shot_raw__")
        <root>/04_evaluation/zero_shot_inferences/{variant}
              /inference_results_{dataset}/inferences/{image}.png
        where variant == run_name without its "zero_shot_" prefix.

Source images and ground truth live outside the repo, under `paths.dataset_root()`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import paths

STAGE_DIR: dict[str, str] = {
    "scratch": "02_few_shot_training_from_scratch",
    "finetune": "03_few_shot_fine-tuning",
}

ZERO_SHOT_EXPERIMENT = "__zero_shot_raw__"
ZERO_SHOT_PREFIX = "zero_shot_"

# Where each dataset keeps its inputs and ground truth, relative to `paths.dataset_root()`.
# Conventions mirror torchtrainer's `datasets/vessel_base.py`; kept as a plain
# table so the backend never has to import torch.
DATASET_SOURCES: dict[str, dict] = {
    "vessmap": {
        "root": "VessMAP",
        "image_dir": "images",
        "image_ext": ".tiff",
        "label_dir": "annotator1/labels",
        "label_ext": ".png",
    },
    "drive": {
        "root": "DRIVE",
        # DRIVE is split into test/ and training/ subtrees and its GT stem
        # differs from the image stem ("01_test" -> "01_manual1"), so it gets
        # dedicated resolution below.
        "split": True,
        "image_ext": ".tif",
        "label_ext": ".gif",
    },
    "dca1": {
        "root": "DCA1",
        "image_dir": "images",
        "image_ext": ".tif",
        "label_dir": "labels",
        "label_ext": ".png",
        "label_suffix": "_gt",
    },
    "octa2d": {
        "root": "OCTA2D",
        "image_dir": "images",
        "image_ext": ".bmp",
        "label_dir": "labels",
        "label_ext": ".bmp",
    },
}

DATASETS = list(DATASET_SOURCES)


class PanelNotFound(FileNotFoundError):
    """Raised when a requested panel does not resolve to a readable file."""


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #
def _component(value: str, field: str) -> str:
    """Validate one user-supplied path segment.

    `_safe` alone is not enough: a `..` inside e.g. `image` can climb out of the
    inferences folder and still land inside REPO_ROOT, which would turn the image
    endpoint into an arbitrary repo-file reader. Segments are opaque names, so
    rejecting separators outright is both correct and simplest.
    """
    s = str(value)
    if not s or s in (".", "..") or "/" in s or "\\" in s or "\x00" in s:
        raise PanelNotFound(f"illegal {field}: {value!r}")
    return s


def _safe(path: Path) -> Path:
    """Resolve `path` and assert it stays inside a whitelisted root."""
    resolved = Path(path).resolve()
    for root in paths.image_roots():
        try:
            if resolved.is_relative_to(root.resolve()):
                return resolved
        except (OSError, ValueError):
            continue
    raise PanelNotFound(f"path outside allowed roots: {path}")


def _first_existing(candidates: list[Path]) -> Optional[Path]:
    for c in candidates:
        if c.is_file():
            return c
    return None


def _glob_stem(directory: Path, stem: str) -> Optional[Path]:
    """Fallback when the recorded extension does not match what is on disk."""
    if not directory.is_dir():
        return None
    for p in sorted(directory.glob(f"{stem}.*")):
        if p.is_file():
            return p
    return None


# --------------------------------------------------------------------------- #
# Prediction masks
# --------------------------------------------------------------------------- #
def zero_shot_variant(run_name: str) -> str:
    """"zero_shot_resnet18_vessshape" -> "resnet18_vessshape"."""
    return run_name[len(ZERO_SHOT_PREFIX):] if run_name.startswith(ZERO_SHOT_PREFIX) else run_name


# Fine-tuned model_type -> its zero-shot counterpart in the CSVs. Reuses the
# canonical map from utils_eval (same source app/stats.py relies on).
_ZERO_SHOT_FALLBACK = {
    "VSUNet18": "Zero-Shot VSUNet18",
    "VSUNet50": "Zero-Shot VSUNet50",
    "IN-UNet18": "Zero-Shot IN-UNet18",
    "IN-UNet50": "Zero-Shot IN-UNet50",
    "LiteMedSAM-FT": "Zero-Shot LiteMedSAM",
}


def zero_shot_alias(model_type: str) -> Optional[str]:
    """The zero-shot name for a fine-tuned model, or None if it has none.

    Models trained from scratch (UNet18/UNet50) genuinely have no zero-shot
    counterpart — there are no pretrained weights to evaluate.
    """
    try:
        import utils_eval
        mapping = dict(utils_eval.FINE_TUNED_TO_ZEROSHOT)
    except Exception:
        mapping = _ZERO_SHOT_FALLBACK
    return mapping.get(model_type)


def pred_path(dataset: str, stage: str, experiment: str, run_name: str, image: str) -> Path:
    """Absolute path to one prediction PNG. Does not check existence."""
    if not dataset or not run_name or image in ("", None):
        raise PanelNotFound("pred panel needs dataset, run_name and image")
    dataset = _component(dataset, "dataset")
    run_name = _component(run_name, "run_name")
    image = _component(image, "image")

    if stage == "zero_shot":
        variant = _component(zero_shot_variant(run_name), "variant")
        base = paths.ZERO_SHOT_RAW_DIR / variant / f"inference_results_{dataset}"
    else:
        stage_dir = STAGE_DIR.get(stage)
        if stage_dir is None:
            raise PanelNotFound(f"unknown stage: {stage!r}")
        if not experiment:
            raise PanelNotFound("pred panel needs an experiment for this stage")
        experiment = _component(experiment, "experiment")
        base = paths.REPO_ROOT / stage_dir / dataset / "experiments" / experiment / run_name / "inference_results"

    return _safe(base / "inferences" / f"{image}.png")


# --------------------------------------------------------------------------- #
# Source images + ground truth
# --------------------------------------------------------------------------- #
def _drive_paths(image: str) -> tuple[Path, Path]:
    """DRIVE: "01_test" -> test/images/01_test.tif + test/1st_manual/01_manual1.gif."""
    root = paths.dataset_root() / "DRIVE"
    split = "training" if image.endswith("_training") else "test"
    number = image.split("_")[0]
    return (
        root / split / "images" / f"{image}.tif",
        root / split / "1st_manual" / f"{number}_manual1.gif",
    )


def input_path(dataset: str, image: str) -> Path:
    cfg = DATASET_SOURCES.get(dataset)
    if cfg is None:
        raise PanelNotFound(f"unknown dataset: {dataset!r}")
    image = _component(image, "image")
    if cfg.get("split"):
        candidate = _drive_paths(image)[0]
    else:
        candidate = paths.dataset_root() / cfg["root"] / cfg["image_dir"] / f"{image}{cfg['image_ext']}"

    found = _first_existing([candidate]) or _glob_stem(candidate.parent, image)
    if found is None:
        raise PanelNotFound(f"input image not found for {dataset}/{image} (looked at {candidate})")
    return _safe(found)


def gt_path(dataset: str, image: str) -> Path:
    cfg = DATASET_SOURCES.get(dataset)
    if cfg is None:
        raise PanelNotFound(f"unknown dataset: {dataset!r}")
    image = _component(image, "image")
    if cfg.get("split"):
        candidate = _drive_paths(image)[1]
        stem = candidate.stem
    else:
        stem = f"{image}{cfg.get('label_suffix', '')}"
        candidate = paths.dataset_root() / cfg["root"] / cfg["label_dir"] / f"{stem}{cfg['label_ext']}"

    found = _first_existing([candidate]) or _glob_stem(candidate.parent, stem)
    if found is None:
        raise PanelNotFound(f"ground truth not found for {dataset}/{image} (looked at {candidate})")
    return _safe(found)


def resolve_panel(
    kind: str,
    dataset: str,
    image: str,
    stage: Optional[str] = None,
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Path:
    """Absolute, whitelist-checked path for any panel kind."""
    if kind == "input":
        return input_path(dataset, image)
    if kind == "gt":
        return gt_path(dataset, image)
    if kind == "pred":
        return pred_path(dataset, stage or "", experiment or "", run_name or "", image)
    raise PanelNotFound(f"unsupported panel kind: {kind!r}")


def is_mask(kind: str) -> bool:
    """Masks resample with NEAREST; photographic inputs with LANCZOS."""
    return kind in ("pred", "gt")


# --------------------------------------------------------------------------- #
# Metric-driven ranking (Figure Studio)
# --------------------------------------------------------------------------- #
from functools import lru_cache  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import data  # noqa: E402
from .schemas import AutoFillRequest, GridFigureSpec, RankImagesRequest, RankRunsRequest  # noqa: E402

_INDEX_COLS = ["image", "run_name", "num_samples", "run", "rep", "model_type", "stage", "experiment"]


def _sanitize(value):
    """NaN/Inf are not valid JSON; convert to None (native Python numbers otherwise)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if np.isnan(f) or np.isinf(f):
        return None
    return f


def _round_or_none(value, ndigits: int = 4):
    s = _sanitize(value)
    return round(s, ndigits) if s is not None else None


def _int_or_none(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(value)


def _shot_label(n: int) -> str:
    return "Zero-shot" if n == 0 else f"{n}-shot"


def _image_sort_key(image: str):
    # Numeric-looking stems sort as numbers (so "10" doesn't precede "2");
    # everything else sorts lexicographically after them.
    return (0, int(image)) if image.isdigit() else (1, image)


@lru_cache(maxsize=64)
def _index_cached(dataset: str, metrics: tuple[str, ...], _mtime: tuple) -> pd.DataFrame:
    df = data.load_long_dataframe("per_image", [dataset])
    if df.empty:
        return df
    if "is_mock" in df.columns:
        df = df[~(df["is_mock"] == True)]  # noqa: E712 (keeps False and NaN rows)
    cols = [c for c in _INDEX_COLS + list(metrics) if c in df.columns]
    df = df[cols].copy()
    df["image"] = df["image"].astype(str)
    return df.reset_index(drop=True)


def index_for(dataset: str, metrics: list[str] | None = None) -> pd.DataFrame:
    """Slim, cached per-image index: identity columns + the requested metrics."""
    mkey = tuple(sorted(metrics)) if metrics else ()
    return _index_cached(dataset, mkey, data._mtime_key((dataset,), "per_image"))


def pick_run(rows: pd.DataFrame, metric: str, policy: str,
             fixed_run: int | None = None, fixed_rep: int | None = None) -> pd.Series | None:
    """Select one row from a (dataset, image, model_type[, num_samples]) candidate set."""
    rows = rows.dropna(subset=[metric])
    if rows.empty:
        return None
    # Ascending sort so index (n-1)//2 is the *lower* median for even counts.
    ordered = rows.sort_values(metric, ascending=True, kind="mergesort")
    n = len(ordered)
    if policy == "best":
        return ordered.iloc[-1]
    if policy == "worst":
        return ordered.iloc[0]
    if policy == "median":
        return ordered.iloc[(n - 1) // 2]
    if policy == "fixed":
        match = rows[(rows["run"] == fixed_run) & (rows["rep"] == fixed_rep)]
        if not match.empty:
            return match.iloc[0]
        return ordered.iloc[(n - 1) // 2]
    raise ValueError(f"unknown policy: {policy!r}")


def rank_images(req: RankImagesRequest) -> list[dict]:
    """Rank test images by a metric aggregated across the selected models/regimes."""
    df = index_for(req.dataset, [req.metric])
    if df.empty or req.metric not in df.columns:
        return []
    if req.num_samples:
        df = df[df["num_samples"].isin(req.num_samples)]
    if req.model_types:
        df = df[df["model_type"].isin(req.model_types)]
    if req.stages:
        df = df[df["stage"].isin(req.stages)]
    df = df.dropna(subset=[req.metric])
    if df.empty:
        return []

    metric = req.metric
    per_model = df.groupby(["image", "model_type"])[metric].mean().unstack("model_type")

    if req.agg == "mean":
        score = df.groupby("image")[metric].mean()
    elif req.agg == "median":
        score = df.groupby("image")[metric].median()
    elif req.agg == "min":
        score = df.groupby("image")[metric].min()
    elif req.agg == "max":
        score = df.groupby("image")[metric].max()
    elif req.agg == "spread":
        score = per_model.max(axis=1) - per_model.min(axis=1)
    else:
        raise ValueError(f"unknown agg: {req.agg!r}")

    counts = df.groupby("image").size()
    ordered_images = score.sort_values(ascending=req.ascending).index[: req.limit]

    items = []
    for image in ordered_images:
        per_model_row = per_model.loc[image]
        items.append({
            "image": image,
            "score": _sanitize(score.loc[image]),
            "n": int(counts.loc[image]),
            "per_model": {mt: _round_or_none(v) for mt, v in per_model_row.items()},
        })
    return items


def rank_runs(req: RankRunsRequest) -> list[dict]:
    """Rank the candidate runs (splits x seeds) that could fill one grid cell."""
    df = index_for(req.dataset, [req.metric])
    if df.empty or req.metric not in df.columns:
        return []
    same_image = df[df["image"] == str(req.image)]
    sub = same_image[same_image["model_type"] == req.model_type]
    if req.num_samples is not None:
        sub = sub[sub["num_samples"] == req.num_samples]
    # Zero-shot rows are filed under "Zero-Shot <model>", so asking for the
    # fine-tuned name at n=0 would otherwise return nothing.
    if sub.empty and req.num_samples == 0:
        alias = zero_shot_alias(req.model_type)
        if alias:
            sub = same_image[(same_image["model_type"] == alias) & (same_image["num_samples"] == 0)]
    sub = sub.dropna(subset=[req.metric])
    if sub.empty:
        return []

    best_row = pick_run(sub, req.metric, "best")
    median_row = pick_run(sub, req.metric, "median")
    best_idx = best_row.name if best_row is not None else None
    median_idx = median_row.name if median_row is not None else None

    ordered = sub.sort_values(req.metric, ascending=req.ascending, kind="mergesort").head(req.limit)

    items = []
    for idx, row in ordered.iterrows():
        items.append({
            "image": row["image"],
            "score": _sanitize(row[req.metric]),
            "stage": row.get("stage"),
            "experiment": row.get("experiment"),
            "run_name": row.get("run_name"),
            "model_type": row.get("model_type"),
            "num_samples": _int_or_none(row.get("num_samples")),
            "run": _int_or_none(row.get("run")),
            "rep": _int_or_none(row.get("rep")),
            "is_median": bool(idx == median_idx),
            "is_best": bool(idx == best_idx),
        })
    return items


def autofill(req: AutoFillRequest) -> dict:
    """Build a complete GridFigureSpec by picking one run per (model_type, num_samples) cell."""
    model_types = list(req.model_types)
    num_samples = list(req.num_samples)
    metric = req.metric

    idx = index_for(req.dataset, [metric])
    if not idx.empty:
        idx = idx[idx["image"] == str(req.image)]

    def cell(model_type: str, n: int) -> dict:
        if idx.empty:
            sub = idx
        else:
            sub = idx[(idx["model_type"] == model_type) & (idx["num_samples"] == n)]
            # At n=0 the fine-tuned names do not exist — the zero-shot rows are
            # filed under "Zero-Shot <model>". Without this alias a "Zero-shot"
            # row in the grid would silently come back empty.
            if sub.empty and n == 0:
                alias = zero_shot_alias(model_type)
                if alias:
                    sub = idx[(idx["model_type"] == alias) & (idx["num_samples"] == n)]
        row = pick_run(sub, metric, req.policy, req.fixed_run, req.fixed_rep)
        if row is None:
            return {"kind": "empty", "dataset": req.dataset, "image": str(req.image),
                    "model_type": model_type, "num_samples": n}
        return {
            "kind": "pred",
            "dataset": req.dataset,
            "image": str(req.image),
            "stage": row.get("stage"),
            "experiment": row.get("experiment"),
            "run_name": row.get("run_name"),
            # Report the *requested* name, not the aliased "Zero-Shot X" row
            # label: the grid's row header already carries the regime, so the
            # alias would make every zero-shot caption repeat it.
            "model_type": model_type,
            "num_samples": _int_or_none(row.get("num_samples")),
            "run": _int_or_none(row.get("run")),
            "rep": _int_or_none(row.get("rep")),
            "score": _sanitize(row.get(metric)),
        }

    extra = req.include_input or req.include_gt
    models_as_cols = req.orientation == "models_as_cols"

    if models_as_cols:
        rows, cols = len(num_samples), len(model_types) + (1 if extra else 0)
        col_labels = ([""] if extra else []) + list(model_types)
        row_labels = [_shot_label(n) for n in num_samples]
    else:
        rows, cols = len(model_types) + (1 if extra else 0), len(num_samples)
        row_labels = ([""] if extra else []) + list(model_types)
        col_labels = [_shot_label(n) for n in num_samples]

    panels: list[dict] = [{"kind": "empty"} for _ in range(rows * cols)]

    def set_panel(r: int, c: int, panel: dict) -> None:
        panels[r * cols + c] = panel

    input_panel = {"kind": "input", "dataset": req.dataset, "image": str(req.image)}
    gt_panel = {"kind": "gt", "dataset": req.dataset, "image": str(req.image)}

    if models_as_cols:
        offset = 1 if extra else 0
        for r, n in enumerate(num_samples):
            for c, mt in enumerate(model_types):
                set_panel(r, c + offset, cell(mt, n))
        if extra:
            if req.include_input:
                set_panel(0, 0, input_panel)
            if req.include_gt:
                set_panel(1 if rows > 1 else 0, 0, gt_panel)
    else:
        offset = 1 if extra else 0
        for r, mt in enumerate(model_types):
            for c, n in enumerate(num_samples):
                set_panel(r + offset, c, cell(mt, n))
        if extra:
            if req.include_input:
                set_panel(0, 0, input_panel)
            if req.include_gt:
                set_panel(0, 1 if cols > 1 else 0, gt_panel)

    spec = GridFigureSpec(rows=rows, cols=cols, panels=panels,
                           row_labels=row_labels, col_labels=col_labels)
    return spec.model_dump()


def options() -> dict:
    """Dataset-scoped selector options for the Figure Studio UI."""
    by_dataset: dict[str, dict] = {}
    all_model_types: set[str] = set()
    all_stages: set[str] = set()

    for ds in data.DATASETS:
        df = index_for(ds)
        if df.empty:
            continue
        images = sorted(df["image"].unique().tolist(), key=_image_sort_key)
        model_types = sorted(df["model_type"].dropna().unique().tolist())
        num_samples = sorted(int(n) for n in df["num_samples"].dropna().unique().tolist())
        by_dataset[ds] = {"num_samples": num_samples, "images": images, "model_types": model_types}
        all_model_types.update(model_types)
        all_stages.update(df["stage"].dropna().unique().tolist())

    return {
        "datasets": list(by_dataset),
        "metrics": data.METRICS,
        "model_types": sorted(all_model_types),
        "stages": sorted(all_stages),
        "by_dataset": by_dataset,
    }
