# VessShape Experiments (Master’s Project)

Experiments and analysis for a master’s thesis on few-shot blood vessel segmentation. The core idea is to instill a strong shape bias in segmentation models via synthetic pre-training (VessShape), and then measure transfer to real datasets (DRIVE, VessMAP) with very few labeled samples.

This repo is organized into four stages: (1) pre-training, (2) few‑shot training from scratch, (3) few‑shot fine‑tuning of pre-trained weights, and (4) evaluation. Reusable utilities live in `src/`. A local **web app** (`05_webapp/`) consolidates training (stages 02 & 03) and the generation of publication-quality evaluation figures — see [Web App](#web-app-training--evaluation-ui).

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
├── 02_few_shot_training_from_scratch/   # Train from scratch (random init)
│   ├── run_serial_training.py           # Single launcher: --dataset <ds> --config_file <cfg>
│   ├── dca1/  drive/  octa2d/  vessmap/  # One folder per target dataset, each with:
│   │   ├── config_unet18.yaml            #   ResNet18-U-Net, from scratch
│   │   ├── config_unet50.yaml            #   ResNet50-U-Net, from scratch
│   │   └── experiments/                  #   Auto-generated outputs
├── 03_few_shot_fine-tuning/             # Fine-tune from pre-trained weights
│   ├── run_serial_fine-tuning.py        # Single launcher: --dataset <ds> --config_file <cfg>
│   ├── dca1/  drive/  octa2d/  vessmap/  # One folder per target dataset, each with:
│   │   ├── config_unet18_imagenet.yaml   #   ResNet18-U-Net, ImageNet init
│   │   ├── config_unet18_vessshape.yaml  #   ResNet18-U-Net, VessShape-pretrained init
│   │   ├── config_unet50_imagenet.yaml   #   ResNet50-U-Net, ImageNet init
│   │   ├── config_unet50_vessshape.yaml  #   ResNet50-U-Net, VessShape-pretrained init
│   │   ├── config_litemedsam.yaml        #   LiteMedSAM fine-tuning
│   │   └── experiments/                  #   Auto-generated outputs
├── 04_evaluation/
│   ├── eval_master_cross_dataset.ipynb # Cross-dataset aggregation, bars, Wilcoxon
│   ├── eval_{vessmap,drive,dca1,octa2d}.ipynb
│   ├── utils.py / utils_eval.py        # Plotting / loaders / stats helpers
│   ├── model_colors.yaml               # Shared model→color palette
│   └── results/  figures/  zero_shot/  # Consolidated CSVs and figures
├── 05_webapp/                          # Local training + evaluation web app
│   ├── backend/                        # FastAPI (rendering, stats, dashboard, training jobs)
│   ├── frontend/                       # React + Vite + TypeScript + Tailwind control panel
│   └── README.md                       # Setup & usage
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

Each stage has a **single launcher** at its root that takes `--dataset` (which target dataset
folder to run) and `--config_file` (which config inside that folder). Config files follow the
scheme `config_<model>_<init>.yaml`, where `<model> ∈ {unet18, unet50, litemedsam}` and, for
fine-tuning, `<init> ∈ {imagenet, vessshape}` (stage 02 trains from scratch, so it has no `<init>`).

1) Few-shot training from scratch

```bash
cd 02_few_shot_training_from_scratch
python run_serial_training.py --dataset drive --config_file config_unet18.yaml
python run_serial_training.py --dataset drive --config_file config_unet50.yaml
# swap --dataset for dca1 / octa2d / vessmap
```

1) Few-shot fine-tuning (starting from pre-trained weights)

```bash
cd 03_few_shot_fine-tuning
python run_serial_fine-tuning.py --dataset drive --config_file config_unet18_imagenet.yaml
python run_serial_fine-tuning.py --dataset drive --config_file config_unet18_vessshape.yaml
python run_serial_fine-tuning.py --dataset drive --config_file config_unet50_imagenet.yaml
python run_serial_fine-tuning.py --dataset drive --config_file config_unet50_vessshape.yaml
python run_serial_fine-tuning.py --dataset drive --config_file config_litemedsam.yaml
# swap --dataset for dca1 / octa2d / vessmap
```

1) Evaluation & plots

- Open the `04_evaluation/eval_*.ipynb` notebooks to aggregate results and draw figures.
- Reusable helpers in `04_evaluation/utils.py` and `utils_eval.py` (matplotlib curves, grouped
  bars, Wilcoxon heatmaps, zero-shot annotations, etc.).
- Or use the **web app** (below) for interactive, publication-quality figure generation.

## Web App (training & evaluation UI)

`05_webapp/` is a local React + FastAPI control panel that consolidates training and evaluation
without editing notebooks or YAML by hand. It is **additive** — it imports/reads `src/*`,
`04_evaluation/utils_eval.py`, the launchers, and the result CSVs, and never modifies them.

Capabilities:

- **Chart Builder** — line / bar / scatter figures rendered with matplotlib on the backend
  (live preview equals the exported file). Full control over data, shot-regime
  (Zero-/One-/Few-shot) filtering & faceting, axes (fields, min/max, labels, sizes, rotation),
  titles, fonts, **seaborn palettes + a Photoshop-style color wheel**, per-series style, legend,
  and **export to SVG / PNG / JPG / PDF**.
- **Significance** — flexible **Wilcoxon signed-rank** grid (alpha, alternative, comparisons,
  datasets, per-N panels) reusing `utils_eval`/`scipy`.
- **Dashboard** — pin figures and compare them side-by-side, grouped by shot-regime; export all.
- **Training** — configure & launch stages **02 / 03** (dataset, model, pretraining,
  hyperparameters, few-shot loop), preview the exact YAML, **Save config** or **Save & Run**
  with a live log console. Stage **01** is a "Coming soon" placeholder.

Run it:

```bash
# backend (port 8000) — uses the `base` conda env (fastapi + eval deps);
# training jobs are launched with the `mestrado_env` interpreter automatically.
cd 05_webapp/backend
conda run -n base python -m uvicorn app.main:app --reload --port 8000

# frontend (port 5173) — proxies /api to the backend
cd 05_webapp/frontend
npm install && npm run dev   # open http://localhost:5173
```

See `05_webapp/README.md` for details, the environment matrix, and the full API surface.

## Config files: parameters (stages 02 & 03)

Every config is a single YAML with three sections: `train_params`, `experiment_params`, `test_params`.

**Value convention.** Parameters are forwarded to the training/inference scripts as command-line
flags (`--key value`). A value of `''` (empty string) means a **boolean flag that is ON** (e.g.
`ignore_class_weights: ''`); to turn it **OFF**, comment out or remove the line. List-valued
parameters are space-separated strings (`resize_size: '512 512'` → two integers).

### `train_params`

| Parameter | Example / values | Description |
|---|---|---|
| `experiment_name` | `multi_finetune_on_drive_unet18_imagenet_weights_lr_e-2_ignore_class_weights` | Names the output folder under `experiments/` and the W&B project. |
| `weights_strategy` | path to `checkpoint.pt` | Pre-trained (VessShape) checkpoint to initialize from. Omit/comment for no checkpoint. |
| `encoder_weights` | `imagenet` \| omit | Encoder backbone weights (U-Net). Omit for random init. |
| `run_name` | `''` | Set automatically by the launcher (leave empty). |
| `validate_every` | `5` | Validate every N epochs. |
| `save_val_imgs` | flag | Save validation images. |
| `val_img_indices` | `'0 1 2'` | Indices of validation images to save. |
| `dataset_path` | `/.../DRIVE` | Dataset root directory. |
| `dataset_class` | `drive_few` | Training dataset class (`*_few` = few-shot subset sampler). |
| `split_strategy` | `''` | Set automatically (selected sample IDs). |
| `channels` | `gray\|rgb\|all\|green` | Input channels; used by LiteMedSAM (`rgb`) and VessMAP (`all`). Omit for the per-dataset default. |
| `resize_size` | `'512 512'` | Resize `H W` (drive 512, octa2d 384, dca1 288, vessmap 256; litemedsam 256). |
| `loss_function` | `bce` | Loss; LiteMedSAM uses `bce`, U-Nets default to `cross_entropy`. |
| `model_class` | `resnet18_unet\|resnet50_unet\|litemedsam` | Model architecture. |
| `model_params` | `freeze_encoder=False` | Extra model kwargs (LiteMedSAM). |
| `num_epochs` | `500` | Training epochs. |
| `validation_metric` | `Dice` | Metric tracked for validation. |
| `maximize_validation_metric` | flag | Treat the metric as "higher is better". |
| `bs_train` / `bs_valid` | `8` / `20` | Train / validation batch sizes. |
| `weight_decay` | `0.0` | Optimizer weight decay. |
| `lr` | `1e-2` (unet18) / `1e-3` (unet50) | Learning rate (varies per model). |
| `lr_decay` | `1.0` | LR decay factor. |
| `ignore_class_weights` | flag | Ignore class weights (ON for dca1/drive/octa2d, OFF for vessmap). |
| `optimizer` | `adam` | Optimizer. |
| `num_workers` | `12` | DataLoader workers. |
| `log_wandb` | flag | Log to Weights & Biases. |
| `wandb_project` | = `experiment_name` | W&B project name. |
| `wandb_group` | `''` | Set automatically by the launcher. |
| `suppress_checkpoint` | flag (commented) | Do not save periodic checkpoints. |
| `suppress_best_checkpoint` | flag | Do not save `best_model.pt`. |
| `checkpoint_every` | `-1` | Save a checkpoint every N epochs (`-1` = only the last). |

Advanced defaults inherited from torchtrainer's `DefaultTrainer` (rarely set here): `device`,
`use_amp`, `deterministic`, `benchmark`, `momentum`, `patience`, `augmentation_strategy`, `seed`,
`copy_model_every`, `meta`.

### `experiment_params` (few-shot sweep loop)

| Parameter | Example | Description |
|---|---|---|
| `min_samples` | `1` | First number of labeled samples in the sweep. |
| `max_samples` | drive/vessmap 16, dca1/octa2d 20 | Last number of labeled samples. |
| `runs` | `5` | Random sample combinations per sample count. |
| `reps` | `3` | Repetitions (seeds) per combination. |
| `with_replacement` | `False` | Sample with replacement. |
| `output_dir` | `experiments` | Base output dir (relative to the dataset folder). |
| `step` | `2` | Increment of the sample count between sweep points. |
| `csv_path` | `/.../train.csv` | CSV of image IDs to sample from. |
| `weights_id` | `imagenet\|vsunet18\|vsunet50\|FromScratch` | Label embedded in output folder names. |

### `test_params` (inference)

| Parameter | Example | Description |
|---|---|---|
| `run_path` | `''` | Set automatically by the launcher. |
| `dataset_path` | `/.../DRIVE` | Dataset root for inference. |
| `dataset_class` | `drive` | Test dataset class (without `_few`). |
| `model_class` | `resnet18_unet` | Model architecture. |
| `channels` / `resize_size` | `rgb` / `'256 256'` | Used by LiteMedSAM. |
| `checkpoint_type` | `best\|last` | Which checkpoint to load. |
| `use_amp` | flag | Mixed precision. |
| `save_inference_images` | flag | Save prediction images. |
| `inference_dir_name` | `inference_results` | Output subfolder name. |
| `tta_type` | `none\|logits\|probs` | Test-time augmentation. |
| `threshold` | `0.5` | Binary threshold (`-1` maximizes Dice on train). |
| `encoder_weights` / `imagenet_normalize` / `skip_checkpoint_loading` | — | Loading variants (e.g. ImageNet zero-shot). |
| `seed` / `device` / `deterministic` / `benchmark` | — | Execution controls. |
| `force_headless` / `skip_boxplot` | flag | Plotting orchestration. |

**Inference orchestration** (read by the launcher, not `test.py`): `delete_checkpoint`,
`batch_inference`, `enable_inference`, `max_inference_retries`, `delete_only_on_success`,
`aggregate_inference_means`.

## Output Conventions & Reproducibility

- One directory per run: `<model>_weights_id:<ID>_run:<r>_rep:<k>_ns:<n>/`.
- Determinism: repetition index seeds key random steps; few-shot samplers track unique subsets when `with_replacement=False`.
- Metrics: appended to `log.csv`; inference metrics live under `inference_results/` (with `metrics_stats.csv`).

## Contact

If something is unclear or breaks in your environment, please open an issue or PR.
