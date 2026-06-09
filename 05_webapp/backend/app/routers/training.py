"""Training endpoints: options, config preview, save, run, stop, status, WS logs."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from .. import data, training
from ..schemas import SaveConfigRequest, TrainingRequest

router = APIRouter(prefix="/api/training", tags=["training"])


@router.get("/options")
def options():
    return {
        "datasets": data.DATASETS,
        "stages": [
            {"value": "scratch", "label": "Stage 02 — From Scratch",
             "models": ["resnet18_unet", "resnet50_unet"],
             "pretraining": ["scratch"]},
            {"value": "finetune", "label": "Stage 03 — Fine-tuning",
             "models": ["resnet18_unet", "resnet50_unet", "litemedsam"],
             "pretraining": ["imagenet", "vessshape"]},
        ],
        "optimizers": ["adam", "sgd", "adamw"],
        "loss_functions": ["cross_entropy", "bce"],
        "checkpoint_types": ["last", "best"],
        "dataset_defaults": training._DATASET_DEFAULTS,
    }


@router.post("/config")
def config_preview(req: TrainingRequest):
    try:
        return {"yaml": training.to_yaml(req),
                "filename": training.default_config_filename(req)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/save")
def save(req: SaveConfigRequest):
    try:
        return training.save_config(req.request, overwrite=req.overwrite)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/run")
async def run(req: TrainingRequest):
    try:
        loop = asyncio.get_running_loop()
        return training.manager.run(req, loop)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/stop")
def stop():
    return training.manager.stop()


@router.get("/status")
def status():
    return training.manager.status()


@router.websocket("/logs")
async def logs(ws: WebSocket):
    await ws.accept()
    q = training.manager.subscribe()
    try:
        for line in training.manager.backlog():
            await ws.send_text(line)
        while True:
            line = await q.get()
            await ws.send_text(line)
    except WebSocketDisconnect:
        pass
    finally:
        training.manager.unsubscribe(q)
