"""Training config builder + managed job runner.

Builds the 3-section YAML the existing launchers expect and (optionally) runs
`run_serial_{training,fine-tuning}.py` as a single managed subprocess, streaming
stdout to WebSocket subscribers. The launcher itself is unchanged.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import yaml

from . import paths
from .schemas import TrainingRequest

# Per-dataset defaults observed in the existing configs.
_DATASET_DEFAULTS = {
    "vessmap": {"resize": "256 256", "channels": "all"},
    "drive":   {"resize": "256 256", "channels": "gray"},
    "dca1":    {"resize": "288 288", "channels": "gray"},
    "octa2d":  {"resize": "384 384", "channels": "gray"},
}

_WEIGHTS_ID = {
    ("scratch", "resnet18_unet"): "FromScratch",
    ("scratch", "resnet50_unet"): "FromScratch",
    ("finetune", "imagenet"): "imagenet",
}


def _train_python() -> str:
    """Interpreter able to import torchtrainer (the training env)."""
    env = os.environ.get("VESSLAB_TRAIN_PYTHON")
    if env:
        return env
    candidate = Path.home() / "anaconda3" / "envs" / "mestrado_env" / "bin" / "python"
    return str(candidate) if candidate.exists() else "python"


def default_config_filename(req: TrainingRequest) -> str:
    if req.config_filename:
        return req.config_filename if req.config_filename.endswith(".yaml") else f"{req.config_filename}.yaml"
    short = {"resnet18_unet": "unet18", "resnet50_unet": "unet50", "litemedsam": "litemedsam"}[req.model_class]
    if req.stage == "finetune" and req.model_class != "litemedsam":
        return f"config_{short}_{req.pretraining}.yaml"
    return f"config_{short}.yaml"


def _weights_id(req: TrainingRequest) -> str:
    if req.weights_id:
        return req.weights_id
    if req.stage == "finetune" and req.pretraining == "vessshape":
        return "vsunet18" if req.model_class == "resnet18_unet" else "vsunet50"
    return _WEIGHTS_ID.get((req.stage, req.pretraining)) or _WEIGHTS_ID.get((req.stage, req.model_class), "FromScratch")


def build_config(req: TrainingRequest) -> dict:
    is_lms = req.model_class == "litemedsam"
    dd = _DATASET_DEFAULTS.get(req.dataset, {"resize": "256 256", "channels": "all"})
    channels = req.channels or ("rgb" if is_lms else dd["channels"])
    resize = req.resize_size or dd["resize"]
    exp_name = req.experiment_name or f"multi_{req.stage}_{req.dataset}_{req.model_class}_{_weights_id(req)}"

    train: dict = {"experiment_name": exp_name}
    # pretraining source (finetune only)
    if req.stage == "finetune":
        if req.pretraining == "vessshape" and req.weights_path:
            train["weights_strategy"] = req.weights_path
        elif req.pretraining == "imagenet":
            train["encoder_weights"] = "imagenet"

    train.update({
        "validate_every": req.validate_every,
        "dataset_path": req.dataset_path,
        "dataset_class": f"{req.dataset}_few",
        "split_strategy": "",
        "channels": channels,
        "resize_size": resize,
    })
    if is_lms:
        train["loss_function"] = "bce"
        train["model_params"] = "freeze_encoder=False"
    elif req.loss_function:
        train["loss_function"] = req.loss_function
    train.update({
        "model_class": req.model_class,
        "num_epochs": req.num_epochs,
        "validation_metric": req.validation_metric,
        "maximize_validation_metric": "",
        "bs_train": req.bs_train,
        "bs_valid": req.bs_valid,
        "weight_decay": req.weight_decay,
        "lr": req.lr,
        "lr_decay": req.lr_decay,
        "optimizer": req.optimizer,
        "num_workers": req.num_workers,
        "suppress_best_checkpoint": "",
        "checkpoint_every": -1,
    })
    if req.ignore_class_weights:
        train["ignore_class_weights"] = ""

    experiment = {
        "min_samples": req.min_samples,
        "max_samples": req.max_samples,
        "runs": req.runs,
        "reps": req.reps,
        "with_replacement": req.with_replacement,
        "output_dir": req.output_dir,
        "csv_path": req.csv_path,
        "step": req.step,
        "weights_id": _weights_id(req),
    }

    test = {
        "run_path": "",
        "dataset_path": req.dataset_path,
        "dataset_class": req.dataset,
        "model_class": req.model_class,
        "resize_size": resize,
        "checkpoint_type": req.checkpoint_type,
        "inference_dir_name": "inference_results",
        "delete_checkpoint": req.delete_checkpoint,
        "batch_inference": req.batch_inference,
        "enable_inference": req.enable_inference,
        "force_headless": True,
        "skip_boxplot": True,
        "max_inference_retries": 1,
        "delete_only_on_success": True,
        "aggregate_inference_means": True,
    }
    if is_lms:
        test["channels"] = channels
    if req.save_inference_images:
        test["save_inference_images"] = ""

    return {"train_params": train, "experiment_params": experiment, "test_params": test}


def to_yaml(req: TrainingRequest) -> str:
    cfg = build_config(req)
    # Validate the 3-section shape round-trips.
    parsed = yaml.safe_load(yaml.safe_dump(cfg))
    assert {"train_params", "experiment_params", "test_params"} <= set(parsed)
    return yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)


def save_config(req: TrainingRequest, overwrite: bool = False) -> dict:
    stage_dir = Path(paths.STAGES[req.stage]["dir"])
    ds_dir = stage_dir / req.dataset
    if not ds_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {ds_dir}")
    filename = default_config_filename(req)
    target = ds_dir / filename
    if target.exists() and not overwrite:
        return {"saved": False, "path": str(target), "reason": "exists"}
    target.write_text(to_yaml(req))
    return {"saved": True, "path": str(target), "filename": filename}


class JobManager:
    """At most one active training subprocess; streams stdout to subscribers."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._buffer: deque[str] = deque(maxlen=5000)
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._meta: dict = {}
        self._lock = threading.Lock()

    # --- subscription (used by the WS endpoint) ---------------------------- #
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def backlog(self) -> list[str]:
        return list(self._buffer)

    def _emit(self, line: str) -> None:
        self._buffer.append(line)
        if self._loop is None:
            return
        for q in list(self._subscribers):
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, line)
            except RuntimeError:
                pass

    # --- lifecycle -------------------------------------------------------- #
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict:
        running = self.is_running()
        rc = None if running or self._proc is None else self._proc.returncode
        return {"running": running, "returncode": rc, **self._meta}

    def run(self, req: TrainingRequest, loop: asyncio.AbstractEventLoop) -> dict:
        with self._lock:
            if self.is_running():
                raise RuntimeError("A training job is already running.")
            saved = save_config(req, overwrite=True)
            stage = paths.STAGES[req.stage]
            cmd = [_train_python(), "-u", stage["launcher"],
                   "--dataset", req.dataset, "--config_file", saved["filename"]]
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            self._loop = loop
            self._buffer.clear()
            self._proc = subprocess.Popen(
                cmd, cwd=stage["dir"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env, start_new_session=True,
            )
            self._meta = {
                "stage": req.stage, "dataset": req.dataset, "config_file": saved["filename"],
                "experiment_name": build_config(req)["train_params"]["experiment_name"],
                "pid": self._proc.pid, "started_at": datetime.now().isoformat(),
                "cmd": " ".join(cmd),
            }
            self._emit(f"[webapp] $ {' '.join(cmd)}  (cwd={stage['dir']})")
            threading.Thread(target=self._pump, args=(self._proc,), daemon=True).start()
            return self._meta

    def _pump(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self._emit(line.rstrip("\n"))
        finally:
            proc.wait()
            self._emit(f"[webapp] process exited with code {proc.returncode}")

    def stop(self) -> dict:
        if not self.is_running() or self._proc is None:
            return {"stopped": False, "reason": "not running"}
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        self._emit("[webapp] stop requested (SIGTERM)")
        return {"stopped": True}


# Module-level singleton used by the router.
manager = JobManager()
