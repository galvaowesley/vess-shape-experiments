"""Loader / labeler / mock helpers for the multi-dataset evaluation pipeline.

Reads `inference_results/metrics_stats.csv` (mean row) per run from the
two-stage experiment layout (`02_few_shot_training_from_scratch/<ds>/experiments/...`
and `03_few_shot_fine-tuning/<ds>/experiments/...`).

Reuses plotting/aggregation from the legacy `utils.py` without modifying it.
The legacy `aggregate_inference_means` is NOT reused because it expects the
wandb-nested config schema; this module reads the flat top-level
`<run>/config.yaml` (with a wandb fallback).
"""
from __future__ import annotations

import math
import os
import re
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

DATASETS = ['vessmap', 'drive', 'dca1', 'octa2d']
STAGES = {
    'scratch': '02_few_shot_training_from_scratch',
    'finetune': '03_few_shot_fine-tuning',
}
DEFAULT_IGNORE_DIRS = {
    'multi_finetune_on_drive_resnet50_A',
    'multi_finetune_on_drive_unet50_imagenet_weights',
    'multi_finetune_on_drive_unet18_imagenet_weights',
}
METRICS = ['Accuracy', 'IoU', 'Precision', 'Recall', 'Dice', 'AUC']

# --- Output / artifact directories (relative to this module) -----------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_MODULE_DIR, 'results')
FIGURES_DIR = os.path.join(_MODULE_DIR, 'figures')
ZERO_SHOT_DIR = os.path.join(_MODULE_DIR, 'zero_shot')
# Raw zero-shot inference outputs (per-image metrics + images). Kept separate
# from the consolidated CSVs in ZERO_SHOT_DIR; also referenced by the legacy
# models_evalation.ipynb.
ZERO_SHOT_RAW_DIR = os.path.join(_MODULE_DIR, 'zero_shot_inferences')


def ensure_output_dirs() -> None:
    """Create the artifact directories if they don't exist (idempotent)."""
    for d in (RESULTS_DIR, FIGURES_DIR, ZERO_SHOT_DIR):
        os.makedirs(d, exist_ok=True)


DEFAULT_LABEL_MAP = {
    ('unet18', 'scratch'):    'UNet18',
    ('unet50', 'scratch'):    'UNet50',
    ('unet18', 'vessshape'):  'VSUNet18',
    ('unet50', 'vessshape'):  'VSUNet50',
    ('unet18', 'imagenet'):   'IN-UNet18',
    ('unet50', 'imagenet'):   'IN-UNet50',
    ('litemedsam', 'litemedsam'): 'LiteMedSAM-FT',
}

ZERO_SHOT_LABEL = {
    'resnet18_unet': 'Zero-Shot VSUNet18',
    'resnet50_unet': 'Zero-Shot VSUNet50',
}

# Registry of zero-shot variants produced by run_zero_shot_inference.ipynb.
# `variant_key` is also the raw output subdir under ZERO_SHOT_RAW_DIR:
#   zero_shot_inferences/<variant_key>/inference_results_<dataset>/{metrics,metrics_stats}.csv
# Because resnet{18,50}_unet now has two weight origins (vessshape vs imagenet), the
# model_class alone no longer identifies the label — `model_type` is authored explicitly.
ZERO_SHOT_VARIANTS = {
    'resnet18_vessshape': {'model_class': 'resnet18_unet', 'weights': 'vessshape',  'model_type': 'Zero-Shot VSUNet18'},
    'resnet50_vessshape': {'model_class': 'resnet50_unet', 'weights': 'vessshape',  'model_type': 'Zero-Shot VSUNet50'},
    'resnet18_imagenet':  {'model_class': 'resnet18_unet', 'weights': 'imagenet',   'model_type': 'Zero-Shot IN-UNet18'},
    'resnet50_imagenet':  {'model_class': 'resnet50_unet', 'weights': 'imagenet',   'model_type': 'Zero-Shot IN-UNet50'},
    'litemedsam':         {'model_class': 'litemedsam',    'weights': 'litemedsam', 'model_type': 'Zero-Shot LiteMedSAM'},
    'litemedsam_normalized': {
        'model_class': 'litemedsam',
        'weights': 'litemedsam',          # pesos: lite_medsam.pth — NÃO encoder_weights='imagenet'
        'model_type': 'Zero-Shot LiteMedSAM+IN',
        'imagenet_normalize': True,       # só pré-processamento de entrada (_PreprocessModel)
    },
}

# Legacy raw dirs (pre-phase-3) held the vessshape resnet zero-shot under these names.
# Used as a fallback so existing outputs keep working until regenerated.
_LEGACY_RAW_DIR_FOR_VARIANT = {
    'resnet18_vessshape': 'resnet18',
    'resnet50_vessshape': 'resnet50',
}

_RUN_TOKEN_RE = re.compile(r'run:(\d+)_rep:(\d+)_ns:(\d+)')


def _read_flat_config(run_dir: str) -> tuple[str | None, str | None]:
    """Return (wandb_group, model_class) from the run dir.

    Prefers the flat `<run>/config.yaml`; falls back to the wandb-nested
    `<run>/wandb/latest-run/files/config.yaml` (`{value: ...}` form).
    """
    flat_path = os.path.join(run_dir, 'config.yaml')
    if os.path.exists(flat_path):
        try:
            with open(flat_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            wandb_group = cfg.get('wandb_group')
            model_class = cfg.get('model_class')
            if wandb_group is not None or model_class is not None:
                return wandb_group, model_class
        except Exception:
            pass

    wandb_path = os.path.join(run_dir, 'wandb', 'latest-run', 'files', 'config.yaml')
    if os.path.exists(wandb_path):
        try:
            with open(wandb_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            wg = cfg.get('wandb_group', {})
            mc = cfg.get('model_class', {})
            return (
                wg.get('value') if isinstance(wg, dict) else wg,
                mc.get('value') if isinstance(mc, dict) else mc,
            )
        except Exception:
            pass
    return None, None


def _aggregate_experiment(experiment_path: str) -> pd.DataFrame:
    """Iterate run subdirs and collect the 'mean' row of metrics_stats.csv.

    Silently skips runs that don't have `inference_results/metrics_stats.csv`
    (or whose CSV is malformed). Returns columns:
    [run_name, num_samples, run, rep, wandb_group, model_class, *METRICS].
    """
    rows = []
    if not os.path.isdir(experiment_path):
        return pd.DataFrame(columns=['run_name', 'num_samples', 'run', 'rep',
                                     'wandb_group', 'model_class'] + METRICS)

    for run_name in sorted(os.listdir(experiment_path)):
        run_dir = os.path.join(experiment_path, run_name)
        if not os.path.isdir(run_dir):
            continue

        stats_path = os.path.join(run_dir, 'inference_results', 'metrics_stats.csv')
        if not os.path.exists(stats_path):
            continue

        try:
            df_stats = pd.read_csv(stats_path)
        except Exception:
            continue
        if 'statistic' not in df_stats.columns:
            continue
        mean_row = df_stats[df_stats['statistic'] == 'mean']
        if mean_row.empty:
            continue
        mean_row = mean_row.iloc[0]

        match = _RUN_TOKEN_RE.search(run_name)
        if match:
            run_id = int(match.group(1))
            rep_id = int(match.group(2))
            num_samples = int(match.group(3))
        else:
            run_id = rep_id = num_samples = None

        wandb_group, model_class = _read_flat_config(run_dir)

        row = {
            'run_name': run_name,
            'num_samples': num_samples,
            'run': run_id,
            'rep': rep_id,
            'wandb_group': wandb_group,
            'model_class': model_class,
        }
        for m in METRICS:
            row[m] = float(mean_row[m]) if m in mean_row.index and pd.notna(mean_row[m]) else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=['run_name', 'num_samples', 'run', 'rep',
                                     'wandb_group', 'model_class'] + METRICS)
    return pd.DataFrame(rows)


def _classify_experiment_dir(name: str) -> dict | None:
    """Classify an experiment dir name into {arch, origin, stage}.

    Returns None if it cannot be classified (caller skips).
    """
    s = name.lower()
    if 'multi_train_scratch' in s:
        if 'unet18' in s:
            return {'arch': 'unet18', 'origin': 'scratch', 'stage': 'scratch'}
        if 'unet50' in s:
            return {'arch': 'unet50', 'origin': 'scratch', 'stage': 'scratch'}
        return None
    if 'multi_finetune_on' in s:
        if 'litemedsam_weights' in s:
            return {'arch': 'litemedsam', 'origin': 'litemedsam', 'stage': 'finetune'}
        if 'unet18_vessshape_weights' in s:
            return {'arch': 'unet18', 'origin': 'vessshape', 'stage': 'finetune'}
        if 'unet50_vessshape_weights' in s:
            return {'arch': 'unet50', 'origin': 'vessshape', 'stage': 'finetune'}
        if 'unet18_imagenet_weights' in s:
            return {'arch': 'unet18', 'origin': 'imagenet', 'stage': 'finetune'}
        if 'unet50_imagenet_weights' in s:
            return {'arch': 'unet50', 'origin': 'imagenet', 'stage': 'finetune'}
    return None


def _compact_label(classification: dict, label_map: dict | None = None) -> str | None:
    label_map = label_map if label_map is not None else DEFAULT_LABEL_MAP
    return label_map.get((classification['arch'], classification['origin']))


def generate_mock_zero_shot(dataset: str,
                            model_classes: Iterable[str] = ('resnet18_unet', 'resnet50_unet'),
                            placeholder: float = math.nan) -> pd.DataFrame:
    """Build a placeholder zero-shot DataFrame (NaN metrics) for datasets
    whose real zero-shot inference hasn't been produced yet.

    Adds `(mock)` to the model_type label so plots/legends make the
    placeholder obvious.
    """
    rows = []
    for mc in model_classes:
        base_label = ZERO_SHOT_LABEL.get(mc, f'Zero-Shot {mc}')
        row = {
            'run_name': f'zero_shot_mock_{dataset}_{mc}',
            'num_samples': 0,
            'run': None,
            'rep': None,
            'wandb_group': None,
            'model_class': mc,
            'model_type': f'{base_label} (mock)',
            'stage': 'zero_shot',
            'experiment': '__mock__',
            'is_mock': True,
        }
        for m in METRICS:
            row[m] = placeholder
        rows.append(row)
    return pd.DataFrame(rows)


def _find_zero_shot_csv(dataset: str, zero_shot_dir: str) -> str | None:
    """Locate the consolidated zero-shot CSV for a dataset.

    Search order: the provided `zero_shot_dir` first, then the raw dir
    `ZERO_SHOT_RAW_DIR` (backward compatibility). Returns the path or None.
    This makes the pipeline auto-discover real zero-shot for ANY dataset as
    soon as its CSV is dropped in `zero_shot/` — no code change needed.
    """
    fname = f'zero_shot_inference_results_on_{dataset}.csv'
    for d in (zero_shot_dir, ZERO_SHOT_RAW_DIR):
        if d is None:
            continue
        candidate = os.path.join(d, fname)
        if os.path.exists(candidate):
            return candidate
    return None


def _load_real_zero_shot(dataset: str, zero_shot_dir: str) -> pd.DataFrame | None:
    """Load the real zero-shot CSV for a dataset. Returns None if absent.

    Trusts an explicit `model_type` column when present (the consolidated CSV authored by
    `build_zero_shot_csv` carries the correct label per variant, since `model_class` alone
    no longer disambiguates vessshape vs imagenet). Falls back to the `ZERO_SHOT_LABEL`
    map only for rows whose `model_type` is missing/empty.
    """
    csv_path = _find_zero_shot_csv(dataset, zero_shot_dir)
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    df = df.copy()
    if 'num_samples' not in df.columns:
        df['num_samples'] = 0
    mapped = df['model_class'].map(ZERO_SHOT_LABEL)
    if 'model_type' in df.columns:
        existing = df['model_type']
        df['model_type'] = existing.where(existing.notna() & (existing.astype(str) != ''), mapped)
    else:
        df['model_type'] = mapped
    df['stage'] = 'zero_shot'
    df['experiment'] = '__zero_shot_csv__'
    df['is_mock'] = False
    for col in ('run', 'rep', 'wandb_group'):
        if col not in df.columns:
            df[col] = None
    return df


def _variant_raw_dir(variant_key: str, dataset: str, raw_dir: str) -> str | None:
    """Return the existing raw inference dir for a zero-shot variant/dataset, or None.

    Tries the phase-3 layout `<raw_dir>/<variant_key>/inference_results_<ds>` first, then
    the legacy vessshape resnet dirs (`resnet18`/`resnet50`).
    """
    candidate = os.path.join(raw_dir, variant_key, f'inference_results_{dataset}')
    if os.path.isdir(candidate):
        return candidate
    legacy = _LEGACY_RAW_DIR_FOR_VARIANT.get(variant_key)
    if legacy:
        alt = os.path.join(raw_dir, legacy, f'inference_results_{dataset}')
        if os.path.isdir(alt):
            return alt
    return None


def build_zero_shot_csv(dataset: str,
                        raw_dir: str = ZERO_SHOT_RAW_DIR,
                        out_dir: str = ZERO_SHOT_DIR,
                        variants: dict | None = None,
                        write: bool = True) -> pd.DataFrame:
    """Consolidate raw zero-shot inferences into the per-dataset CSV the eval pipeline reads.

    For each variant in `variants` (default `ZERO_SHOT_VARIANTS`), reads the `mean` row of
    `<raw_dir>/<variant_key>/inference_results_<dataset>/metrics_stats.csv` (with legacy
    fallback) and emits one row. Writes
    `<out_dir>/zero_shot_inference_results_on_<dataset>.csv` and returns the DataFrame.

    Columns: [run_name, num_samples=0, wandb_group, model_class, weights, *METRICS, model_type].
    """
    variants = variants if variants is not None else ZERO_SHOT_VARIANTS
    cols = ['run_name', 'num_samples', 'wandb_group', 'model_class', 'weights'] + METRICS + ['model_type']
    rows = []
    for variant_key, meta in variants.items():
        run_dir = _variant_raw_dir(variant_key, dataset, raw_dir)
        if run_dir is None:
            continue
        stats_path = os.path.join(run_dir, 'metrics_stats.csv')
        if not os.path.exists(stats_path):
            continue
        try:
            df_stats = pd.read_csv(stats_path)
        except Exception:
            continue
        if 'statistic' not in df_stats.columns:
            continue
        mean_row = df_stats[df_stats['statistic'] == 'mean']
        if mean_row.empty:
            continue
        mean_row = mean_row.iloc[0]
        row = {
            'run_name': f'zero_shot_{variant_key}',
            'num_samples': 0,
            'wandb_group': meta['weights'],
            'model_class': meta['model_class'],
            'weights': meta['weights'],
            'model_type': meta['model_type'],
        }
        for m in METRICS:
            row[m] = float(mean_row[m]) if m in mean_row.index and pd.notna(mean_row[m]) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows, columns=cols)
    if write and not df.empty:
        ensure_output_dirs()
        out_path = os.path.join(out_dir, f'zero_shot_inference_results_on_{dataset}.csv')
        df.to_csv(out_path, index=False)
    return df


def load_dataset_results(dataset: str,
                         root: str = '..',
                         label_map: dict | None = None,
                         ignore_dirs: Iterable[str] | None = None,
                         include_zero_shot: bool = True,
                         mock_zero_shot: bool = False,
                         zero_shot_dir: str | None = None) -> pd.DataFrame:
    """Single entry point: load all model variants for one dataset.

    Returns a tidy DataFrame:
        [run_name, num_samples, run, rep, wandb_group, model_class,
         Accuracy, IoU, Precision, Recall, Dice, AUC,
         model_type, stage, experiment, is_mock]
    Rows whose experiment dir is in `ignore_dirs` (default: DEFAULT_IGNORE_DIRS)
    are skipped. Empty experiment dirs are skipped silently.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Expected one of {DATASETS}.")
    ignore_dirs = set(ignore_dirs) if ignore_dirs is not None else set(DEFAULT_IGNORE_DIRS)
    if zero_shot_dir is None:
        zero_shot_dir = ZERO_SHOT_DIR

    frames: list[pd.DataFrame] = []
    for stage_label, stage_dir in STAGES.items():
        exp_root = os.path.join(root, stage_dir, dataset, 'experiments')
        if not os.path.isdir(exp_root):
            continue
        for exp_name in sorted(os.listdir(exp_root)):
            if exp_name in ignore_dirs:
                continue
            exp_path = os.path.join(exp_root, exp_name)
            if not os.path.isdir(exp_path):
                continue
            classification = _classify_experiment_dir(exp_name)
            if classification is None:
                continue
            if classification['stage'] != stage_label:
                # Defensive: dir name says scratch but lives in finetune root, etc.
                continue
            label = _compact_label(classification, label_map)
            if label is None:
                continue

            df = _aggregate_experiment(exp_path)
            if df.empty:
                continue
            df = df.assign(
                model_type=label,
                stage=classification['stage'],
                experiment=exp_name,
                is_mock=False,
            )
            frames.append(df)

    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=['run_name', 'num_samples', 'run', 'rep',
                                    'wandb_group', 'model_class'] + METRICS +
                                   ['model_type', 'stage', 'experiment', 'is_mock'])

    if include_zero_shot:
        zs = _load_real_zero_shot(dataset, zero_shot_dir)
        if zs is None and mock_zero_shot:
            zs = generate_mock_zero_shot(dataset)
        if zs is not None and not zs.empty:
            # Align columns with `out` to allow clean concat.
            for col in out.columns:
                if col not in zs.columns:
                    zs[col] = np.nan if col in METRICS else None
            for col in zs.columns:
                if col not in out.columns:
                    out[col] = np.nan if col in METRICS else None
            zs = zs[out.columns]
            out = pd.concat([out, zs], ignore_index=True)

    return out


# --- Per-image granularity ---------------------------------------------------
_RESNET_DIR_TO_MC = {'resnet18': 'resnet18_unet', 'resnet50': 'resnet50_unet'}


def _aggregate_experiment_per_image(experiment_path: str) -> pd.DataFrame:
    """Like `_aggregate_experiment` but keeps every per-image row from
    `inference_results/metrics.csv` (schema: image, *METRICS).
    """
    rows = []
    if not os.path.isdir(experiment_path):
        return pd.DataFrame()

    for run_name in sorted(os.listdir(experiment_path)):
        run_dir = os.path.join(experiment_path, run_name)
        if not os.path.isdir(run_dir):
            continue
        metrics_path = os.path.join(run_dir, 'inference_results', 'metrics.csv')
        if not os.path.exists(metrics_path):
            continue
        try:
            df = pd.read_csv(metrics_path)
        except Exception:
            continue
        if df.empty:
            continue

        match = _RUN_TOKEN_RE.search(run_name)
        if match:
            run_id, rep_id, num_samples = (int(match.group(1)),
                                           int(match.group(2)),
                                           int(match.group(3)))
        else:
            run_id = rep_id = num_samples = None

        wandb_group, model_class = _read_flat_config(run_dir)
        df = df.assign(
            run_name=run_name,
            num_samples=num_samples,
            run=run_id,
            rep=rep_id,
            wandb_group=wandb_group,
            model_class=model_class,
        )
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _load_real_zero_shot_per_image(dataset: str,
                                   raw_dir: str = ZERO_SHOT_RAW_DIR,
                                   variants: dict | None = None) -> pd.DataFrame | None:
    """Read raw per-image zero-shot metrics for every available variant from
    `<raw_dir>/<variant_key>/inference_results_<dataset>/metrics.csv` (legacy
    `resnet18`/`resnet50` dirs are tried as fallback). Returns None if nothing is found.
    """
    variants = variants if variants is not None else ZERO_SHOT_VARIANTS
    frames = []
    for variant_key, meta in variants.items():
        run_dir = _variant_raw_dir(variant_key, dataset, raw_dir)
        if run_dir is None:
            continue
        metrics_path = os.path.join(run_dir, 'metrics.csv')
        if not os.path.exists(metrics_path):
            continue
        try:
            df = pd.read_csv(metrics_path)
        except Exception:
            continue
        if df.empty:
            continue
        df = df.assign(
            run_name=f'zero_shot_{variant_key}',
            num_samples=0,
            run=None,
            rep=None,
            wandb_group=meta['weights'],
            model_class=meta['model_class'],
            weights=meta['weights'],
            model_type=meta['model_type'],
            stage='zero_shot',
            experiment='__zero_shot_raw__',
            is_mock=False,
        )
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_dataset_per_image(dataset: str,
                           root: str = '..',
                           label_map: dict | None = None,
                           ignore_dirs: Iterable[str] | None = None,
                           include_zero_shot: bool = True,
                           zero_shot_raw_dir: str | None = None) -> pd.DataFrame:
    """Per-image matrix for one dataset (one row per test image per run).

    Mirrors `load_dataset_results` but reads `inference_results/metrics.csv`.
    Zero-shot is read from the raw per-image dir; mock zero-shot has no
    per-image data and is therefore omitted here.

    Columns: [image, Accuracy, IoU, Precision, Recall, Dice, AUC,
              run_name, num_samples, run, rep, wandb_group, model_class,
              model_type, stage, experiment, is_mock]
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Expected one of {DATASETS}.")
    ignore_dirs = set(ignore_dirs) if ignore_dirs is not None else set(DEFAULT_IGNORE_DIRS)
    if zero_shot_raw_dir is None:
        zero_shot_raw_dir = ZERO_SHOT_RAW_DIR

    frames: list[pd.DataFrame] = []
    for stage_label, stage_dir in STAGES.items():
        exp_root = os.path.join(root, stage_dir, dataset, 'experiments')
        if not os.path.isdir(exp_root):
            continue
        for exp_name in sorted(os.listdir(exp_root)):
            if exp_name in ignore_dirs:
                continue
            exp_path = os.path.join(exp_root, exp_name)
            if not os.path.isdir(exp_path):
                continue
            classification = _classify_experiment_dir(exp_name)
            if classification is None or classification['stage'] != stage_label:
                continue
            label = _compact_label(classification, label_map)
            if label is None:
                continue
            df = _aggregate_experiment_per_image(exp_path)
            if df.empty:
                continue
            df = df.assign(
                model_type=label,
                stage=classification['stage'],
                experiment=exp_name,
                is_mock=False,
            )
            frames.append(df)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if include_zero_shot:
        zs = _load_real_zero_shot_per_image(dataset, zero_shot_raw_dir)
        if zs is not None and not zs.empty:
            out = pd.concat([out, zs], ignore_index=True) if not out.empty else zs

    return out


def split_by_regime(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a results DataFrame into (few_shot, zero_shot) by the `stage` column."""
    if df.empty or 'stage' not in df.columns:
        return df, df.iloc[0:0]
    few = df[df['stage'] != 'zero_shot'].copy()
    zs = df[df['stage'] == 'zero_shot'].copy()
    return few, zs


def save_result_matrices(df: pd.DataFrame, dataset: str,
                         out_dir: str = RESULTS_DIR, suffix: str = '') -> dict[str, str]:
    """Write the three intermediate matrices for a dataset:
        <ds>_few_shot_results<suffix>.csv
        <ds>_zero_shot_results<suffix>.csv
        <ds>_all_results<suffix>.csv      (few + zero-shot)
    `suffix` is '' for the per-run-mean matrix and '_per_image' for the
    per-image matrix. Returns a dict regime -> written path.
    """
    ensure_output_dirs()
    few, zs = split_by_regime(df)
    paths = {
        'few_shot': os.path.join(out_dir, f'{dataset}_few_shot_results{suffix}.csv'),
        'zero_shot': os.path.join(out_dir, f'{dataset}_zero_shot_results{suffix}.csv'),
        'all': os.path.join(out_dir, f'{dataset}_all_results{suffix}.csv'),
    }
    few.to_csv(paths['few_shot'], index=False)
    zs.to_csv(paths['zero_shot'], index=False)
    df.to_csv(paths['all'], index=False)
    return paths


def load_all_datasets(datasets: Iterable[str] = DATASETS, **kwargs) -> dict[str, pd.DataFrame]:
    return {ds: load_dataset_results(ds, **kwargs) for ds in datasets}


def build_summary_table(df: pd.DataFrame,
                        metrics: Iterable[str] = METRICS,
                        group_cols: Iterable[str] = ('model_type', 'num_samples'),
                        sample_filter: Iterable[int] | None = None) -> pd.DataFrame:
    """Mean/std per (model_type, num_samples) using the legacy aggregator."""
    from utils import get_experiments_grouped_stats
    sub = df
    if sample_filter is not None:
        sub = sub[sub['num_samples'].isin(list(sample_filter))]
    return get_experiments_grouped_stats(sub.copy(), list(group_cols), list(metrics))


def default_line_styles_7way() -> dict:
    """Explicit line-style dict so plot_mean_dice_score does not silently
    cycle 4 patterns over 7+ hues.
    """
    return {
        'UNet18':              '-',
        'UNet50':              '-',
        'VSUNet18':            '--',
        'VSUNet50':            '--',
        'IN-UNet18':           '-.',
        'IN-UNet50':           '-.',
        'LiteMedSAM-FT':       ':',
        'Zero-Shot VSUNet18':  '--',
        'Zero-Shot VSUNet50':  '--',
        'Zero-Shot IN-UNet18': '-.',
        'Zero-Shot IN-UNet50': '-.',
        'Zero-Shot LiteMedSAM': ':',
        'Zero-Shot LiteMedSAM+IN': '--',
        'Zero-Shot VSUNet18 (mock)': '--',
        'Zero-Shot VSUNet50 (mock)': '--',
    }


def load_model_palette(yaml_path: str | None = None,
                       warn_missing: Iterable[str] | None = None) -> dict[str, str] | None:
    """Load standardized model_type -> color palette from a YAML file.

    Returns None if the file does not exist. Passed downstream as
    `color_map_override` to plot helpers so colors stay consistent across figures.

    yaml_path: defaults to `<module_dir>/model_colors.yaml`.
    warn_missing: optional iterable of labels expected in the YAML; emits
        a warning listing any that are absent (does not raise).
    """
    if yaml_path is None:
        yaml_path = os.path.join(_MODULE_DIR, 'model_colors.yaml')
    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path) as f:
        palette = yaml.safe_load(f) or {}
    if warn_missing:
        missing = [lbl for lbl in warn_missing if lbl not in palette]
        if missing:
            import warnings
            warnings.warn(
                f"load_model_palette: {len(missing)} label(s) absent in {yaml_path}: {missing}",
                stacklevel=2,
            )
    return palette


# =============================================================================
# Cross-dataset bar charts & Wilcoxon significance helpers
# =============================================================================
# Moved here from `eval_master_cross_dataset.ipynb` so the master notebook holds
# only the orchestration loop. All `plot_*` helpers do lazy matplotlib/scipy
# imports to keep `utils_eval` light on cold load.

ALPHA = 0.05
DATASETS_ORDER = ['dca1', 'drive', 'octa2d', 'vessmap']

# Pairwise mapping fine-tuned model -> zero-shot counterpart for N=0 column
# in Wilcoxon heatmaps and the significance markers on bar charts.
FINE_TUNED_TO_ZEROSHOT = {
    'VSUNet18':      'Zero-Shot VSUNet18',
    'VSUNet50':      'Zero-Shot VSUNet50',
    'IN-UNet18':     'Zero-Shot IN-UNet18',
    'IN-UNet50':     'Zero-Shot IN-UNet50',
    'LiteMedSAM-FT': 'Zero-Shot LiteMedSAM',
}

# VSUNet vs alternative-pretraining comparisons for §8 Wilcoxon heatmaps.
VSUPAIR_COMPARISONS = [
    ('VSUNet18', 'IN-UNet18'),
    ('VSUNet18', 'LiteMedSAM-FT'),
    ('VSUNet50', 'IN-UNet50'),
    ('VSUNet50', 'LiteMedSAM-FT'),
]


def load_per_image_dict(datasets: Iterable[str] | None = None,
                        results_dir: str = RESULTS_DIR) -> dict[str, pd.DataFrame]:
    """Load `{dataset: DataFrame}` from `results/<ds>_all_results_per_image.csv`.

    Used by the Wilcoxon helpers below. Datasets default to `DATASETS_ORDER`.
    """
    if datasets is None:
        datasets = DATASETS_ORDER
    return {ds: pd.read_csv(os.path.join(results_dir, f'{ds}_all_results_per_image.csv'))
            for ds in datasets}


def _sig_label(p, alpha: float = ALPHA) -> str | None:
    """Return `*`, `**`, `***` or None (not significant / nan)."""
    if p is None or (isinstance(p, float) and math.isnan(p)) or p >= alpha:
        return None
    return '***' if p < 0.001 else ('**' if p < 0.01 else '*')


def compute_bar_sig_pvalues(pivot: pd.DataFrame,
                            per_image: dict[str, pd.DataFrame],
                            n_samples: int = 1,
                            metric: str = 'Dice',
                            fine_tuned_to_zeroshot: dict[str, str] | None = None,
                            min_common_images: int = 5) -> dict[str, dict[str, float]]:
    """Wilcoxon one-sided p-values: fine-tuned @ n_samples vs zero-shot counterpart.

    Returns dict `{ft_model: {dataset: p_value}}`. NaN entries indicate insufficient
    paired-image overlap (< `min_common_images`) or missing data.
    """
    from scipy import stats
    if fine_tuned_to_zeroshot is None:
        fine_tuned_to_zeroshot = FINE_TUNED_TO_ZEROSHOT

    sig_p: dict[str, dict[str, float]] = {}
    for ft_model, zs_model in fine_tuned_to_zeroshot.items():
        if ft_model not in pivot.index:
            continue
        p_by_ds: dict[str, float] = {}
        for ds, df_pi in per_image.items():
            zs_df = df_pi[(df_pi['model_type'] == zs_model) & (~df_pi['is_mock'])]
            if zs_df.empty:
                p_by_ds[ds] = float('nan')
                continue
            zs_by_img = zs_df.groupby('image')[metric].mean()
            ft_df = df_pi[
                (df_pi['model_type'] == ft_model)
                & (df_pi['num_samples'] == n_samples)
                & (df_pi['stage'] == 'finetune')
            ]
            if ft_df.empty:
                p_by_ds[ds] = float('nan')
                continue
            ft_by_img = ft_df.groupby('image')[metric].mean()
            common = ft_by_img.index.intersection(zs_by_img.index)
            if len(common) < min_common_images:
                p_by_ds[ds] = float('nan')
                continue
            ft_v, zs_v = ft_by_img[common].values, zs_by_img[common].values
            if (ft_v == zs_v).all():
                p_by_ds[ds] = 1.0
                continue
            _, p = stats.wilcoxon(ft_v, zs_v, alternative='greater')
            p_by_ds[ds] = p
        sig_p[ft_model] = p_by_ds
    return sig_p


def compute_pairwise_sig_matrix(model_a: str,
                                model_b: str,
                                per_image: dict[str, pd.DataFrame],
                                metric: str = 'Dice',
                                datasets_order: Iterable[str] | None = None,
                                fine_tuned_to_zeroshot: dict[str, str] | None = None,
                                min_common_images: int = 5) -> pd.DataFrame:
    """Wilcoxon p-values for H: Metric(model_a) > Metric(model_b), per dataset × N.

    N=0 reuses zero-shot variants via `fine_tuned_to_zeroshot`; N>=1 uses fine-tuned runs.
    Returns DataFrame: rows = datasets (ordered), cols = num_samples.
    """
    from scipy import stats
    if datasets_order is None:
        datasets_order = DATASETS_ORDER
    if fine_tuned_to_zeroshot is None:
        fine_tuned_to_zeroshot = FINE_TUNED_TO_ZEROSHOT

    all_ns = {0}
    for df_pi in per_image.values():
        for m in [model_a, model_b]:
            all_ns.update(
                df_pi[(df_pi['model_type'] == m) & (df_pi['stage'] == 'finetune')][
                    'num_samples'
                ].unique()
            )
    ns_ordered = sorted(all_ns)

    rows: dict[str, dict[int, float]] = {}
    for ds, df_pi in per_image.items():
        row: dict[int, float] = {}
        for n in ns_ordered:
            if n == 0:
                ma = fine_tuned_to_zeroshot.get(model_a)
                mb = fine_tuned_to_zeroshot.get(model_b)
                if ma is None or mb is None:
                    row[n] = float('nan')
                    continue
                df_a = df_pi[(df_pi['model_type'] == ma) & (~df_pi['is_mock'])]
                df_b = df_pi[(df_pi['model_type'] == mb) & (~df_pi['is_mock'])]
            else:
                df_a = df_pi[(df_pi['model_type'] == model_a)
                             & (df_pi['num_samples'] == n) & (df_pi['stage'] == 'finetune')]
                df_b = df_pi[(df_pi['model_type'] == model_b)
                             & (df_pi['num_samples'] == n) & (df_pi['stage'] == 'finetune')]
            if df_a.empty or df_b.empty:
                row[n] = float('nan')
                continue
            a_img = df_a.groupby('image')[metric].mean()
            b_img = df_b.groupby('image')[metric].mean()
            common = a_img.index.intersection(b_img.index)
            if len(common) < min_common_images:
                row[n] = float('nan')
                continue
            a_v, b_v = a_img[common].values, b_img[common].values
            if (a_v == b_v).all():
                row[n] = 1.0
                continue
            _, p = stats.wilcoxon(a_v, b_v, alternative='greater')
            row[n] = p
        rows[ds] = row

    return pd.DataFrame(rows, index=ns_ordered).T.reindex(list(datasets_order))


def plot_crossdataset_bar(pivot: pd.DataFrame,
                          title: str = '',
                          ylabel: str = '',
                          xlabel: str = 'Dataset',
                          save_path: str | None = None,
                          y_limits: tuple | None = None,
                          ymin: float | None = None,
                          ymax: float | None = None,
                          sig_pvalues: dict | None = None,
                          alpha: float = ALPHA,
                          color_map_override: dict | None = None,
                          palette=None,
                          percentage: bool = False,
                          font_sizes: dict | None = None,
                          figsize: tuple = (12, 5.5),
                          rotation: float = 0,
                          legend_loc: str = 'upper left',
                          legend_bbox: tuple = (1.01, 1),
                          legend_title: str = 'Model',
                          datasets_order: Iterable[str] | None = None,
                          dpi: int = 150) -> pd.DataFrame:
    """Grouped bar chart: x-axis = dataset, grouped colors = model_type.

    Parameters
    ----------
    pivot : DataFrame
        index = model_type, columns = dataset.
    sig_pvalues : dict, optional
        {model: {dataset: p_value}} — adds *, **, *** above bars.
    color_map_override : dict, optional
        {model_type: color}. Wins over `palette`. Pass `load_model_palette()`
        for cross-figure consistency.
    palette : sequence, optional
        Fallback palette when a model_type is missing in `color_map_override`.
        Defaults to `sns.color_palette("muted")`.
    percentage : bool
        Multiplies values by 100 and formats yaxis with `%`. Default False.
    font_sizes : dict, optional
        Keys: title, xlabel, ylabel, xaxis, yaxis, legend, sig_marker.
    y_limits : tuple, optional
        (ymin, ymax). Has priority over `ymin`/`ymax`.
    datasets_order : iterable, optional
        Row order of the resulting bar groups. Defaults to `DATASETS_ORDER`.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import PercentFormatter

    fs = font_sizes or {}
    if datasets_order is None:
        datasets_order = DATASETS_ORDER
    if palette is None:
        palette = sns.color_palette("muted")
    pivot_T = pivot.T.reindex(list(datasets_order)).dropna(how='all')
    if percentage:
        pivot_T = pivot_T * 100.0
    n_models = len(pivot_T.columns)

    fallback_cycle = (list(palette) * 10)[:n_models]
    colors = [
        (color_map_override or {}).get(col, fallback_cycle[i])
        for i, col in enumerate(pivot_T.columns)
    ]

    fig, ax = plt.subplots(figsize=figsize)
    pivot_T.plot(kind='bar', ax=ax, color=colors,
                 edgecolor='white', linewidth=0.5, width=0.75)

    ax.set_ylabel(ylabel, fontsize=fs.get('ylabel', 11))
    ax.set_xlabel(xlabel, fontsize=fs.get('xlabel', 11))
    ax.set_title(title, fontsize=fs.get('title', 12))
    ax.legend(title=legend_title, bbox_to_anchor=legend_bbox, loc=legend_loc,
              fontsize=fs.get('legend', 8), title_fontsize=fs.get('legend', 9), framealpha=0.9)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6, linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha='center', fontsize=fs.get('xaxis', 10))
    plt.setp(ax.get_yticklabels(), fontsize=fs.get('yaxis', 10))

    if percentage:
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    ylim_auto = ax.get_ylim()
    offset = (ylim_auto[1] - ylim_auto[0]) * 0.025

    if sig_pvalues is not None:
        bar_w = 0.75 / n_models
        for d_idx, dataset in enumerate(pivot_T.index):
            for m_idx, model in enumerate(pivot_T.columns):
                p = sig_pvalues.get(model, {}).get(dataset, float('nan'))
                label = _sig_label(p, alpha)
                if label is None:
                    continue
                bar_x = d_idx + (m_idx - (n_models - 1) / 2) * bar_w
                bar_h = pivot_T.loc[dataset, model]
                if pd.isna(bar_h):
                    continue
                ax.text(bar_x, bar_h + offset, label,
                        ha='center', va='bottom',
                        fontsize=fs.get('sig_marker', 7),
                        fontweight='bold', color='#333333')

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        ax.set_ylim(
            ymin if ymin is not None else ylim_auto[0],
            ymax if ymax is not None else ylim_auto[1],
        )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.show()
    return pivot_T


def plot_sig_heatmap(p_matrix: pd.DataFrame,
                     title: str = '',
                     save_path: str | None = None,
                     alpha: float = ALPHA,
                     xlabel: str = '# Samples',
                     ylabel: str = 'Dataset',
                     font_sizes: dict | None = None,
                     figsize: tuple | None = None,
                     dpi: int = 150) -> None:
    """Publication-quality Wilcoxon significance heatmap.

    - Dark blue = significant (p < alpha).
    - Light blue = not significant.
    - Gray + red X = no data (NaN).
    - No numeric annotations inside cells.

    font_sizes: dict with optional keys `title, xlabel, ylabel, xaxis, yaxis`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fs = font_sizes or {}
    sig = (p_matrix < alpha).astype(float)
    nan_mask = p_matrix.isna()
    plot_data = sig.where(~nan_mask, other=-1.0)

    nrows, ncols = plot_data.shape
    if figsize is None:
        figsize = (max(5, ncols * 0.55 + 1.5), max(2.5, nrows * 0.65 + 1.2))

    # gray (NaN=-1) | light blue (0=not sig) | dark blue (1=sig)
    cmap = ListedColormap(['#cccccc', '#d6e4f0', '#1a5276'])

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(plot_data.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    for i, ds in enumerate(plot_data.index):
        for j, col in enumerate(plot_data.columns):
            if nan_mask.loc[ds, col]:
                ax.plot(j, i, 'x', color='#cc0000', markersize=7, markeredgewidth=2)

    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_xticklabels(plot_data.columns, fontsize=fs.get('xaxis', 9))
    ax.set_yticklabels(plot_data.index, fontsize=fs.get('yaxis', 9))
    ax.set_xlabel(xlabel, fontsize=fs.get('xlabel', 12))
    ax.set_ylabel(ylabel, fontsize=fs.get('ylabel', 12))
    ax.set_title(title, fontsize=fs.get('title', 12))

    # White cell borders
    import numpy as np
    ax.set_xticks(np.arange(ncols) - 0.5, minor=True)
    ax.set_yticks(np.arange(nrows) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.spines[:].set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.show()


def report_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Quick coverage table: count of rows per (model_type, stage) and
    range of num_samples observed. Useful for the sanity cell.
    """
    if df.empty:
        return pd.DataFrame()
    return (df.groupby(['model_type', 'stage'])['num_samples']
              .agg(['count', 'min', 'max'])
              .sort_index())
