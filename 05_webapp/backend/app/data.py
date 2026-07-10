"""Data access for the evaluation chart builder.

Reads the consolidated CSVs produced by the eval pipeline
(`04_evaluation/results/<ds>_all_results[_per_image].csv` and
`cross_dataset_summary.csv`). The `rescan` action regenerates those CSVs by
walking the experiment folders via the existing `utils_eval` loaders, so the
notebooks are never required to refresh data.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd

from . import paths

DATASETS = ["vessmap", "drive", "dca1", "octa2d"]
METRICS = ["Accuracy", "IoU", "Precision", "Recall", "Dice", "AUC"]

# Fields that make sense as categorical selectors / axes in the UI.
CATEGORICAL_FIELDS = ["model_type", "stage", "regime", "dataset", "num_samples", "run", "rep"]
# Fields valid as a continuous/numeric axis.
NUMERIC_FIELDS = METRICS + ["num_samples"]


def regime_of(num_samples) -> str | None:
    if num_samples is None or (isinstance(num_samples, float) and np.isnan(num_samples)):
        return None
    try:
        n = int(num_samples)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return "Zero-shot"
    if n == 1:
        return "One-shot"
    return "Few-shot"


def _csv_path(dataset: str, source: str) -> str:
    suffix = "_per_image" if source == "per_image" else ""
    return os.path.join(paths.RESULTS_DIR, f"{dataset}_all_results{suffix}.csv")


def _mtime_key(datasets: tuple[str, ...], source: str) -> tuple:
    """Cache key tied to file mtimes so edits invalidate the cache."""
    key = []
    for ds in datasets:
        p = _csv_path(ds, source)
        key.append((ds, os.path.getmtime(p) if os.path.exists(p) else 0.0))
    return tuple(key)


@lru_cache(maxsize=16)
def _load_cached(datasets: tuple[str, ...], source: str, _mtime: tuple) -> pd.DataFrame:
    if source == "summary":
        p = os.path.join(paths.RESULTS_DIR, "cross_dataset_summary.csv")
        df = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
        if not df.empty and datasets:
            df = df[df["dataset"].isin(datasets)]
        if "num_samples" in df.columns:
            df["regime"] = df["num_samples"].map(regime_of)
        return df.reset_index(drop=True)

    frames = []
    for ds in datasets:
        p = _csv_path(ds, source)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        if d.empty:
            continue
        d["dataset"] = ds
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["regime"] = out["num_samples"].map(regime_of)
    return out


def load_long_dataframe(source: str, datasets: list[str] | None) -> pd.DataFrame:
    ds = tuple(datasets) if datasets else tuple(DATASETS)
    return _load_cached(ds, source, _mtime_key(ds, source)).copy()


def _options_from(df: pd.DataFrame) -> dict:
    """Distinct values for categorical fields, min/max for numeric ones."""
    fields: dict[str, dict] = {}
    for col in CATEGORICAL_FIELDS:
        if col in df.columns:
            vals = df[col].dropna().unique().tolist()
            # keep numbers sorted numerically, strings alphabetically
            try:
                vals = sorted(vals)
            except TypeError:
                vals = sorted(map(str, vals))
            fields[col] = {"type": "categorical", "values": vals}
    for col in NUMERIC_FIELDS:
        if col in df.columns and col not in fields:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                fields[col] = {"type": "numeric", "min": float(s.min()), "max": float(s.max())}
    return fields


def get_metadata(source: str, datasets: list[str] | None) -> dict:
    df = load_long_dataframe(source, datasets)
    available_columns = [c for c in df.columns]
    plottable_x = [c for c in (CATEGORICAL_FIELDS + METRICS) if c in df.columns]
    plottable_y = [c for c in (METRICS + ["num_samples"]) if c in df.columns]
    return {
        "datasets": DATASETS,
        "metrics": METRICS,
        "source": source,
        "available_columns": available_columns,
        "x_fields": plottable_x,
        "y_fields": plottable_y,
        "series_fields": [c for c in CATEGORICAL_FIELDS if c in df.columns],
        "facet_fields": [c for c in ["regime", "dataset", "stage", "model_type"] if c in df.columns],
        "options": _options_from(df),
        "regimes": ["Zero-shot", "One-shot", "Few-shot"],
        "row_count": int(len(df)),
    }


def rescan(datasets: list[str] | None, per_image: bool = True) -> dict:
    """Regenerate the consolidated CSVs from the experiment folders.

    Reuses `utils_eval.load_dataset_results` / `load_dataset_per_image` and
    `save_result_matrices` exactly as the notebooks do.
    """
    import utils_eval  # noqa: WPS433 (lazy: keeps cold import light)

    targets = datasets or DATASETS
    summary: dict[str, dict] = {}
    for ds in targets:
        df = utils_eval.load_dataset_results(ds, root=str(paths.REPO_ROOT), mock_zero_shot=False)
        utils_eval.save_result_matrices(df, ds)
        info = {"per_run_rows": int(len(df))}
        if per_image:
            dfp = utils_eval.load_dataset_per_image(ds, root=str(paths.REPO_ROOT))
            utils_eval.save_result_matrices(dfp, ds, suffix="_per_image")
            info["per_image_rows"] = int(len(dfp))
        summary[ds] = info

    _load_cached.cache_clear()
    return {"rescanned": targets, "detail": summary}
