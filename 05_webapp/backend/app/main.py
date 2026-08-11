"""VessShape Lab backend — FastAPI app entrypoint.

Run (from 05_webapp/backend, in an env with fastapi + the eval deps):
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import paths
from .routers import dashboard, eval, figuregrid, gallery, images, stats, training

app = FastAPI(title="VessShape Lab API", version="1.0.0")

# Local dev: Vite serves on 5173. Allow localhost origins.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(eval.router)
app.include_router(stats.router)
app.include_router(dashboard.router)
app.include_router(training.router)
# Figure Studio: three modules sharing the /api/figure prefix.
app.include_router(gallery.router)
app.include_router(images.router)
app.include_router(figuregrid.router)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "repo_root": str(paths.REPO_ROOT),
        "results_dir": str(paths.RESULTS_DIR),
        "results_exists": paths.RESULTS_DIR.is_dir(),
    }
