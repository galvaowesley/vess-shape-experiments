"""Background poller for CPU, RAM, GPU metrics with ring-buffer history.

Samples every INTERVAL seconds using psutil + nvidia-smi.
Works whether or not a training job is running — always-on daemon thread.
"""
from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from typing import Optional

import psutil

INTERVAL = 2          # seconds between samples
HISTORY = 300         # samples kept (~10 min at 2 s)

_NVIDIA_FIELDS = (
    "temperature.gpu",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
)
_NVIDIA_QUERY = ",".join(_NVIDIA_FIELDS)


def _query_nvidia() -> Optional[dict]:
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_NVIDIA_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
        )
        if r.returncode != 0:
            return None
        row = r.stdout.strip().split("\n")[0].split(", ")
        if len(row) < len(_NVIDIA_FIELDS):
            return None
        gpu_temp, gpu_util, mem_used, mem_total, pwr_draw, pwr_limit = row
        return {
            "gpu_temp": float(gpu_temp),
            "gpu_util": float(gpu_util),
            "vram_used_gb": round(float(mem_used) / 1024, 2),
            "vram_total_gb": round(float(mem_total) / 1024, 2),
            "vram_pct": round(float(mem_used) / max(float(mem_total), 1) * 100, 1),
            "gpu_power_w": round(float(pwr_draw), 1),
            "gpu_power_limit_w": round(float(pwr_limit), 1),
        }
    except Exception:
        return None


def _cpu_temp() -> Optional[float]:
    """Return AMD k10temp/Tctl (or first available sensor core) in °C."""
    try:
        sensors = psutil.sensors_temperatures()
        # Prefer k10temp (AMD Ryzen)
        for name in ("k10temp", "coretemp"):
            entries = sensors.get(name, [])
            for e in entries:
                if e.label in ("Tctl", "Package id 0", ""):
                    return round(e.current, 1)
        # Fall back to first entry of any sensor
        for entries in sensors.values():
            if entries:
                return round(entries[0].current, 1)
    except Exception:
        pass
    return None


class SystemMetricsCollector:
    """Singleton; call .start() once at app startup."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: dict[str, deque] = {
            "cpu_pct":       deque(maxlen=HISTORY),
            "cpu_temp":      deque(maxlen=HISTORY),
            "ram_used_gb":   deque(maxlen=HISTORY),
            "ram_total_gb":  deque(maxlen=HISTORY),
            "ram_pct":       deque(maxlen=HISTORY),
            "gpu_temp":      deque(maxlen=HISTORY),
            "gpu_util":      deque(maxlen=HISTORY),
            "vram_used_gb":  deque(maxlen=HISTORY),
            "vram_total_gb": deque(maxlen=HISTORY),
            "vram_pct":      deque(maxlen=HISTORY),
            "gpu_power_w":   deque(maxlen=HISTORY),
            "gpu_power_limit_w": deque(maxlen=HISTORY),
        }
        self._last_cpu_cores: list = []
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self) -> None:
        while True:
            try:
                self._collect()
            except Exception:
                pass
            time.sleep(INTERVAL)

    def _collect(self) -> None:
        cpu_cores = psutil.cpu_percent(percpu=True)
        cpu_pct = round(sum(cpu_cores) / len(cpu_cores), 1) if cpu_cores else 0.0
        cpu_temp = _cpu_temp()
        mem = psutil.virtual_memory()
        ram_used = round(mem.used / 1024 ** 3, 2)
        ram_total = round(mem.total / 1024 ** 3, 2)
        ram_pct = round(mem.percent, 1)

        gpu = _query_nvidia()

        with self._lock:
            self._last_cpu_cores = [round(v, 1) for v in cpu_cores]
            self._history["cpu_pct"].append(cpu_pct)
            self._history["cpu_temp"].append(cpu_temp)
            self._history["ram_used_gb"].append(ram_used)
            self._history["ram_total_gb"].append(ram_total)
            self._history["ram_pct"].append(ram_pct)
            if gpu:
                for k in ("gpu_temp", "gpu_util", "vram_used_gb", "vram_total_gb",
                          "vram_pct", "gpu_power_w", "gpu_power_limit_w"):
                    self._history[k].append(gpu[k])
            else:
                for k in ("gpu_temp", "gpu_util", "vram_used_gb", "vram_total_gb",
                          "vram_pct", "gpu_power_w", "gpu_power_limit_w"):
                    self._history[k].append(None)

    def snapshot(self) -> dict:
        with self._lock:
            current: dict = {}
            history: dict = {}
            for key, dq in self._history.items():
                lst = list(dq)
                history[key] = lst
                current[key] = lst[-1] if lst else None
            current["cpu_cores"] = list(self._last_cpu_cores)
        return {"current": current, "history": history}


# Module-level singleton — started by training.py on first import.
collector = SystemMetricsCollector()
