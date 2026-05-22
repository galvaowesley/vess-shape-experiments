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
    """Load the real zero-shot CSV for a dataset. Returns None if absent."""
    csv_path = _find_zero_shot_csv(dataset, zero_shot_dir)
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    df = df.copy()
    if 'num_samples' not in df.columns:
        df['num_samples'] = 0
    df['model_type'] = df['model_class'].map(ZERO_SHOT_LABEL).fillna(
        df.get('model_type', pd.Series([None] * len(df)))
    )
    df['stage'] = 'zero_shot'
    df['experiment'] = '__zero_shot_csv__'
    df['is_mock'] = False
    if 'run' not in df.columns:
        df['run'] = None
    if 'rep' not in df.columns:
        df['rep'] = None
    if 'wandb_group' not in df.columns:
        df['wandb_group'] = None
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
                                   raw_dir: str = ZERO_SHOT_RAW_DIR) -> pd.DataFrame | None:
    """Read raw per-image zero-shot metrics from
    `<raw_dir>/resnet{18,50}/inference_results_<dataset>/metrics.csv`.
    Returns None if nothing is found.
    """
    frames = []
    for arch_dir, model_class in _RESNET_DIR_TO_MC.items():
        metrics_path = os.path.join(raw_dir, arch_dir,
                                    f'inference_results_{dataset}', 'metrics.csv')
        if not os.path.exists(metrics_path):
            continue
        try:
            df = pd.read_csv(metrics_path)
        except Exception:
            continue
        if df.empty:
            continue
        df = df.assign(
            run_name=f'zero_shot_{arch_dir}',
            num_samples=0,
            run=None,
            rep=None,
            wandb_group=None,
            model_class=model_class,
            model_type=ZERO_SHOT_LABEL.get(model_class, f'Zero-Shot {model_class}'),
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
        'Zero-Shot VSUNet18 (mock)': '--',
        'Zero-Shot VSUNet50 (mock)': '--',
    }


def report_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Quick coverage table: count of rows per (model_type, stage) and
    range of num_samples observed. Useful for the sanity cell.
    """
    if df.empty:
        return pd.DataFrame()
    return (df.groupby(['model_type', 'stage'])['num_samples']
              .agg(['count', 'min', 'max'])
              .sort_index())
