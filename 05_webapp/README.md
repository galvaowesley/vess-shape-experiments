# VessShape Lab — Web App

A local control-panel web app that consolidates **training** (stages 02 & 03) and
**publication-quality evaluation** for the vessel-segmentation experiments in this repo.

It is **additive**: it imports/reads the existing code (`src/*`, `04_evaluation/utils_eval.py`,
the launchers, the result CSVs) but never modifies them. A new generic matplotlib renderer
lives alongside the notebook utilities.

```
05_webapp/
  backend/   FastAPI service (rendering, stats, dashboard store, training jobs)
  frontend/  React + Vite + TypeScript + Tailwind control panel
```

---

## What it does

### Evaluation — Chart Builder (`/charts`)
A flexible builder that renders **publication figures with matplotlib on the backend**, so the
live preview *is* the exported file (true WYSIWYG). Controls:

- Chart type: **line / bar / scatter**
- Data source: per-run, per-image, or summary; dataset / model-method / stage selection
- **Shot-regime segmentation** (Zero-shot / One-shot / Few-shot) as a filter, an axis, or a facet
- Per-axis **data field**, **min/max**, label + label size + tick size + tick rotation, percentage
- Title + size, **global base font size**
- **Colors**: seaborn named **palettes** (with swatch preview) *and* a **Photoshop-style color
  wheel** (HSV + hex/RGB) per series; per-series line style, marker, width
- Legend show/title/position/size; error bands; grid; figure size
- **Live updates** (~250 ms debounce); **export** to **SVG / PNG / JPG / PDF** with DPI and a
  save path (or download)
- **Presets** that reproduce the existing paper figures as a starting point
- **Rescan experiments** regenerates the consolidated CSVs via `utils_eval` (no notebook needed)

### Evaluation — Significance (`/significance`)
Flexible **Wilcoxon signed-rank** grid (paired per-image, `scipy.stats.wilcoxon`):
pick a reference model, comparison models, datasets, metric, and **one panel per N**
(0 = Zero-shot, 1 = One-shot, …); control **alpha**, alternative hypothesis, minimum overlap,
row/column axes, cell colors, fonts, and export.

### Dashboard (`/dashboard`)
Pin any figure or Wilcoxon panel and view them **side-by-side, grouped by shot-regime** for
comparison; reorder, remove, and **export all** to a chosen directory/format.

### Training (`/training`)
Configure stages **02 (from scratch)** and **03 (fine-tuning)**: dataset, model, pretraining
(ImageNet / VessShape checkpoint), hyperparameters, few-shot loop, and inference options. Shows
a **live YAML preview** in the exact 3-section format the launchers expect. Two actions:
**Save config** (writes the YAML into `0{2,3}_*/<dataset>/`) and **Save & Run** (launches the
existing `run_serial_*.py` and streams stdout to the in-app **console**, with Stop). One job at a
time.

### Pretraining (`/pretraining`)
Stage 01 placeholder — **Coming soon**.

---

## Environments

This repo uses conda. Two interpreters are relevant (auto-detected):

| Role | Env | Why |
|------|-----|-----|
| **Backend server** (web + rendering + stats) | `base` (Py 3.11) | has `fastapi`, `uvicorn`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `pyyaml` |
| **Training launcher** (subprocess) | `mestrado_env` (Py 3.12) | the only env with `torchtrainer` installed |

The backend launches training with `mestrado_env`'s Python automatically. Override with the
`VESSLAB_TRAIN_PYTHON` environment variable if your training env differs.

> If you prefer a single env, install the backend deps into `mestrado_env`
> (`pip install -r backend/requirements.txt`) and run the backend there; training then uses the
> same interpreter.

---

## Run it

### 1) Backend (port 8000)
```bash
cd 05_webapp/backend
# deps are already in the `base` env; otherwise:  pip install -r requirements.txt
conda run -n base python -m uvicorn app.main:app --reload --port 8000
# health check:
curl http://localhost:8000/api/health
```

### 2) Frontend (port 5173)
```bash
cd 05_webapp/frontend
npm install      # first time only
npm run dev
# open http://localhost:5173   (it proxies /api and the WebSocket to :8000)
```

Production build: `npm run build` (outputs `frontend/dist/`).

---

## Notes & safety

- **Data source:** the chart builder reads `04_evaluation/results/*.csv` and
  `04_evaluation/zero_shot/*.csv`. "Rescan experiments" regenerates them from the experiment
  folders using the existing loaders.
- **Training is real:** "Save & Run" launches the actual serial grid and **overwrites a config
  file of the same name** in the dataset folder. Review the YAML preview first. The app keeps at
  most one active job (GPU constraint).
- **Runtime state** (pinned figures) lives in `05_webapp/.data/` (gitignored); it never touches
  experiment folders.
- Nothing in `src/`, `04_evaluation/`, the launchers, or the notebooks is modified by this app.

## API surface (backend)

```
GET  /api/health
GET  /api/eval/metadata?source=&datasets=      POST /api/eval/render   POST /api/eval/export
GET  /api/eval/palettes                          POST /api/eval/rescan
POST /api/stats/wilcoxon/render|export|data
GET  /api/dashboard   POST /api/dashboard   DELETE /api/dashboard/{id}
PATCH /api/dashboard/reorder                      POST /api/dashboard/export
GET  /api/training/options                        POST /api/training/config|save|run|stop
GET  /api/training/status                         WS   /api/training/logs
```
