# VessShape Experiments (Master’s Project)

Experiments and analysis for a master’s thesis on few-shot blood vessel segmentation. The core idea is to instill a strong shape bias in segmentation models via synthetic pre-training (VessShape), and then measure transfer to real datasets (DRIVE, VessMAP) with very few labeled samples.

This repo is organized into four stages: (1) pre-training, (2) few‑shot training from scratch, (3) few‑shot fine‑tuning of pre-trained weights, and (4) evaluation. Reusable utilities live in `src/`.

## Purpose (What this project aims to show)

- Build shape-aware vessel segmentation models by pre-training on synthetic images with tubular priors (VessShape), discouraging texture reliance.
- Evaluate how much labeled data is needed to reach strong performance on DRIVE (fundus) and VessMAP (cortex microscopy) when:
   1) training from scratch vs. 2) fine‑tuning VessShape-pretrained weights.
- Report zero-shot behavior (no target fine-tuning) and few-shot curves (Dice vs. # labeled samples).

## Methodology (Short overview)

- Synthetic data (VessShape): images generated from Bézier-based tubular masks blended with diverse foreground/background textures; encourages geometry-first features.
- Two backbones: U‑Net encoders with ResNet18/ResNet50.
- Training regimes:
   - From scratch on target dataset (DRIVE or VessMAP).
   - Pre-train on VessShape, then few-shot fine‑tune on the target dataset.
- Evaluation: dice/accuracy/IoU/precision/recall/AUC; compare across #samples; analyze zero‑shot points.

For a detailed description, see the thesis paper source at `vess-shape-paper/main.tex` (provided externally in your workspace).

## Setup

1) Clone this repository and create an environment (Python 3.12+ recommended):

```bash
git clone <repo-url>
cd vess-shape-experiments
python -m venv .venv && source .venv/bin/activate
# or: conda create -n vessshape python=3.12 && conda activate vessshape
```

2) Install external editable packages via the provided script:

```bash
bash setup.sh
```

What `setup.sh` does:
- Clones and installs `torchtrainer` (editable) — training utilities.
- Clones and installs `vess-shape-dataset` (editable) — synthetic dataset generator.

If you plan to use notebooks and plots interactively, also install Jupyter/Plotly/Seaborn as needed in your environment.

## Related repositories

- Torchtrainer (training utilities)
   - Repo: [github.com/chcomin/torchtrainer](https://github.com/chcomin/torchtrainer)
- VessShape Dataset (synthetic dataset generator)
   - Repo: [github.com/galvaowesley/vess-shape-dataset](https://github.com/galvaowesley/vess-shape-dataset)

## Project Structure (trimmed)

```text
├── setup.sh
├── 01_pretraining_on_vessshape/
│   ├── multi-validation/              # Pre-train with simultaneous validation on multiple datasets
│   │   ├── config.yaml
│   │   ├── dataset.py                 # Builds train + multiple validation datasets
│   │   ├── run_training.py            # Example launcher (params dict → MultiTrainer)
│   │   └── train.py                   # MultiTrainer / MultiModuleRunner definitions
│   └── vessmap-from-scratch/
│       ├── run_training.py
│       └── train.py
├── 02_few_shot_training_from_scratch/
│   ├── drive/
│   │   ├── config.yaml                # train / experiment / test sections
│   │   └── run_serial_fine-tuning.py  # Orchestrates multiple few-shot runs
│   └── vessmap/
│       ├── config.yaml
│       ├── run_serial_fine-tuning.py
│       └── experiments/               # Auto-generated outputs
├── 03_few_shot_fine-tuning/
│   ├── drive/
│   └── vessmap/
│       ├── config.yaml
│       ├── run_serial_fine-tuning.py
│       └── experiments/
├── 04_evaluation/
│   ├── models_evalation.ipynb         # Aggregation/plots (Dice vs. samples, etc.)
│   └── utils.py                       # Plotting/helpers
├── src/
│   ├── dataset.py                     # Dataset builders / parsing
│   ├── static_vess_shape_dataset.py   # Static dataset utilities
│   ├── multi_val_dataset.py           # Multi-validation dataset handling
│   ├── train.py                       # Generic training loop utilities
│   ├── multi_val_train.py             # Multi-dataset training loop logic
│   ├── few_shot_train.py              # Few-shot grid orchestration & inference
│   ├── test.py                        # Inference/metrics entry point
│   └── __init__.py
└── README.md
```

Generated artifacts in `experiments/` include frozen configs (`config.yaml`), training logs (`log.csv`), plots, per-epoch images, `inference_results/` (metrics and optionally predictions), and checkpoints (`checkpoint.pt`, `best_model.pt`).

## How to Run

Prerequisites

- Provide dataset paths in each YAML or params dict (e.g. `dataset_path: /path/to/VessMAP`).
- CSV referenced by `csv_path` should list the image identifiers to sample from.

1) Pre-training (VessShape examples)

```bash
python 01_pretraining_on_vessshape/multi-validation/run_training.py
# or
python 01_pretraining_on_vessshape/vessmap-from-scratch/run_training.py
```

1) Few-shot training from scratch

```bash
python 02_few_shot_training_from_scratch/vessmap/run_serial_fine-tuning.py
python 02_few_shot_training_from_scratch/drive/run_serial_fine-tuning.py
```

1) Few-shot fine-tuning (starting from pre-trained weights)

```bash
python 03_few_shot_fine-tuning/vessmap/run_serial_fine-tuning.py
python 03_few_shot_fine-tuning/drive/run_serial_fine-tuning.py
```

1) Evaluation & plots

- Open `04_evaluation/models_evalation.ipynb` to aggregate results and draw figures.
- Reusable helpers in `04_evaluation/utils.py` (matplotlib/plotly curves, zero-shot annotations, etc.).

## Output Conventions & Reproducibility

- One directory per run: `<model>_weights_id:<ID>_run:<r>_rep:<k>_ns:<n>/`.
- Determinism: repetition index seeds key random steps; few-shot samplers track unique subsets when `with_replacement=False`.
- Metrics: appended to `log.csv`; inference metrics live under `inference_results/` (with `metrics_stats.csv`).

## Contact

If something is unclear or breaks in your environment, please open an issue or PR.
