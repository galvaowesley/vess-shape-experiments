# VessShape Experiments

This repository contains experiments for vessel shape segmentation across multiple retinal (or vascular) datasets. The workflow is organized into four main stages: (1) pre-training, (2) few‑shot training from scratch, (3) few‑shot fine‑tuning of pre-trained weights, and (4) evaluation / analysis. Core reusable code lives in `src/`.

## Project Tree (trimmed)

```text
├── setup.sh
├── 01_pretraining_on_vessshape/
│   ├── multi-validation/              # Pre-train with simultaneous validation on multiple datasets
│   │   ├── config.yaml
│   │   ├── dataset.py                 # Builds train + multiple validation datasets
│   │   ├── run_training.py            # Example launcher (defines params dict → MultiTrainer)
│   │   ├── static_vess_shape_dataset.py
│   │   └── train.py                   # MultiTrainer / MultiModuleRunner definitions
│   └── vessmap-from-scratch/          # Pre-training variant focused on VessMAP only
│       ├── run_training.py
│       └── train.py
├── 02_few_shot_training_from_scratch/
│   ├── drive/                         # Few-shot from scratch on DRIVE
│   │   ├── config.yaml                # YAML with train + experiment + test sections
│   │   └── run_serial_fine-tuning.py  # Orchestrates multiple few-shot runs
│   └── vessmap/                       # Few-shot from scratch on VessMAP
│       ├── config.yaml
│       ├── run_serial_fine-tuning.py
│       └── experiments/               # Auto-generated outputs (logs, metrics, images)
├── 03_few_shot_fine-tuning/
│   ├── drive/                         # Fine-tune pre-trained weights with few samples (DRIVE)
│   └── vessmap/                       # Fine-tune pre-trained weights (VessMAP)
│       ├── config.yaml
│       ├── run_serial_fine-tuning.py
│       └── experiments/               # Structured per run: model checkpoints, metrics, plots
├── 04_evaluation/
│   ├── models_evalation.ipynb         # (Notebook) Aggregated metrics, visualization, comparisons
│   └── utils.py                       # Helpers for evaluation / plotting
├── src/                               # Core library-like code reused across stages
│   ├── dataset.py                     # Dataset builders / parsing
│   ├── static_vess_shape_dataset.py   # Static dataset utilities
│   ├── multi_val_dataset.py           # Multi-validation dataset handling
│   ├── train.py                       # Generic training loop utilities
│   ├── multi_val_train.py             # Multi-dataset training loop logic
│   ├── few_shot_train.py              # Orchestrates few-shot experiment grid & inference
│   ├── test.py                        # Inference / evaluation entry point
│   └── __init__.py
└── README.md
```

Note: The `experiments/` folders contain generated artifacts: `config.yaml` (frozen run config), `log.csv`, `plots.png`, `images/` (epoch snapshots), `inference_results/` (metrics + optional predictions), and `checkpoint.pt` / `best_model.pt` (unless suppressed / deleted).

## Stage Overview

1. Pre-training (`01_pretraining_on_vessshape`)
   - Goal: Learn strong initial representations on larger / combined datasets.
   - `multi-validation`: Trains while validating simultaneously on multiple datasets (e.g. VessShape, DRIVE, VessMAP) using a unified logging & metric pipeline (`MultiTrainer`).
   - `vessmap-from-scratch`: Variant focusing exclusively on VessMAP (no multi-validation logic).

2. Few-shot Training From Scratch (`02_few_shot_training_from_scratch`)
   - Goal: Understand sample efficiency when no pre-training is used.
   - Script iterates over number of labeled samples (1, 2, 4, ... up to a max) and repeats randomized runs & reps to estimate variability.
   - Dynamically builds subsets based on CSV file list; logs subset composition to `selected_samples_report.csv`.

3. Few-shot Fine-tuning (`03_few_shot_fine-tuning`)
   - Goal: Quantify gains from starting with pre-trained weights (supply `weights_id`).
   - Same orchestration as stage 2, but loading and adapting existing representations; typically faster convergence and higher Dice on small n.

4. Evaluation (`04_evaluation`)
   - Aggregates experiment outputs, computes summary statistics, and produces comparative plots (e.g. Dice vs. #samples curves, boxplots).
   - Notebook-driven for exploratory analysis; `utils.py` centralizes reusable evaluation helpers.

5. Core Library (`src/`)
   - Reusable training loop abstractions (Trainer / ModuleRunner), dataset assembly, inference driver, few-shot orchestration, and metric logging glue shared by all stages.

## Key Scripts & How They Interact

- `run_training.py` (stage 1) crafts a parameter dictionary → instantiates `MultiTrainer` → `fit()`.
- `run_serial_fine-tuning.py` (stages 2 & 3) loads a YAML (`config.yaml`) with three sections:
   - `train_params`: static training hyperparameters.
   - `experiment_params`: grid specification (min/max samples, runs, repetitions, step, weight id, etc.).
   - `test_params`: optional inference configuration executed right after each training run.
- `few_shot_train.py` implements the experiment orchestration: sampling, launching training subprocesses, recording chosen samples, triggering immediate inference, and optional checkpoint deletion to save space.
- `test.py` performs inference on a saved run directory, writing metrics (`metrics.csv`, `metrics_stats.csv`) and optional predictions.

## Setup

1. Clone repository

```bash
git clone <repo-url>
cd vess-shape-experiments
```

1. (Recommended) Create and activate environment (Python 3.12+)

```bash
python -m venv .venv
source .venv/bin/activate
# or: conda create -n vessshape python=3.12 && conda activate vessshape
```

1. Install dependencies & external editable packages

```bash
bash setup.sh
```

2. (Optional) Install extra packages you might need (e.g. jupyter, seaborn) manually.

## Running Examples

Pre-training (multi-validation example):

```bash
python 01_pretraining_on_vessshape/multi-validation/run_training.py
```

Few-shot from scratch on VessMAP:

```bash
python 02_few_shot_training_from_scratch/vessmap/run_serial_fine-tuning.py
```

Few-shot fine-tuning (provide pre-trained weights id in YAML):

```bash
python 03_few_shot_fine-tuning/vessmap/run_serial_fine-tuning.py
```

Inspect results: open the corresponding folder under `experiments/` and view `log.csv`, `plots.png`, and `inference_results/`.

## Output Conventions

- One directory per run: `<model>_weights_id:<ID>_run:<r>_rep:<k>_ns:<n>/`.
- Checkpoints: `checkpoint.pt` (last), `best_model.pt` (only if validation metric improves and not suppressed).
- Metrics: appended row-wise to `log.csv`; aggregated inference metrics in `inference_results/`.
- Sample selection trace: `selected_samples_report.csv` (global across runs for a given experiment root).

## Reproducibility Notes

- Seeds: repetition index (`rep_idx`) is used as seed for deterministic sampling inside each run.
- Unique combination tracking avoids duplicate subsets when `with_replacement` is False.
- Hardware determinism flags (`deterministic`, `benchmark`) are controlled via trainer args (see `train.py`).

## Datasets

You must provide local dataset paths in each YAML or params dict (e.g. `dataset_path: /path/to/VessMAP`). CSV file referenced by `csv_path` should list image identifiers (first column). Adjust paths to your environment before running.

---

Feel free to extend: add new model backbones (`get_model` in training scripts), additional metrics, or alternative sampling strategies. Contributions are welcome.

---

Contact: open an issue or PR if something is unclear.
