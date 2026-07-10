"""Parse training subprocess stdout to extract progress state and epoch metrics.

Handles two sources:
- [INFO] lines from few_shot_train.py  → experiment progress counters
- tqdm Epochs lines from torchtrainer  → per-epoch metric values

Thread-safety: all mutations happen on the JobManager thread that calls on_line();
reads via progress()/epoch_metrics() are unsynchronised but only return copies,
so a momentary inconsistent read is acceptable (UI polls every 2 s).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Matches: [INFO] Starting experiments: min_samples=1, max_samples=20, runs=5, reps=3, step=2
_RE_START = re.compile(
    r"\[INFO\] Starting experiments:.*?min_samples=(\d+).*?max_samples=(\d+)"
    r".*?runs=(\d+).*?reps=(\d+).*?step=(\d+)"
)
# Matches: [INFO] === num_samples: 4 ===
_RE_NS = re.compile(r"\[INFO\] === num_samples: (\d+) ===")
# Matches: [INFO]  Run number: 2
_RE_RUN = re.compile(r"\[INFO\]\s+Run number: (\d+)")
# Matches: [INFO]    [rep 1] Running:
_RE_REP = re.compile(r"\[INFO\]\s+\[rep (\d+)\] Running:")
# Matches tqdm epoch line: "Epochs:  10%|…| 50/500 […, Dice=0.78, Train loss=0.34]"
_RE_TQDM = re.compile(r"Epochs:\s+\d+%.*?\|\s*(\d+)/(\d+)\s*\[")
# Matches key=value pairs in tqdm postfix
_RE_KV = re.compile(r"([\w ]+)=([\d.]+(?:e[+-]?\d+)?)")


def _sample_sizes(min_s: int, max_s: int, step: int) -> list[int]:
    """Reproduce the few_shot_train.py sample-size sequence."""
    sizes: list[int] = []
    ns = min_s
    while ns <= max_s:
        sizes.append(ns)
        if ns == 1:
            ns = 2
        else:
            ns += step
    return sizes


@dataclass
class _SampleProgress:
    total_reps: int   # runs × reps
    done: int = 0


class LogParser:
    def __init__(self) -> None:
        self._runs: int = 0
        self._reps: int = 0
        self._sample_sizes: list[int] = []
        self._total_steps: int = 0
        self._done_steps: int = 0

        self._current_ns: Optional[int] = None
        self._current_run: Optional[int] = None
        self._current_rep: Optional[int] = None

        self._per_ns: dict[int, _SampleProgress] = {}

        # epoch metrics: list of {run_id, num_samples, epoch, total_epochs, <metric>: value, ...}
        self._epoch_metrics: list[dict] = []

    # ------------------------------------------------------------------ #

    def reset(self, runs: int, reps: int, min_samples: int, max_samples: int, step: int) -> None:
        self.__init__()
        self._runs = runs
        self._reps = reps
        self._sample_sizes = _sample_sizes(min_samples, max_samples, step)
        self._total_steps = len(self._sample_sizes) * runs * reps
        for ns in self._sample_sizes:
            self._per_ns[ns] = _SampleProgress(total_reps=runs * reps)

    def on_line(self, line: str) -> None:
        # [INFO] Starting experiments — fallback if reset() params weren't exact
        m = _RE_START.search(line)
        if m:
            min_s, max_s, runs, reps, step = (int(x) for x in m.groups())
            if not self._sample_sizes:
                self.reset(runs, reps, min_s, max_s, step)
            return

        # [INFO] === num_samples: X ===
        m = _RE_NS.search(line)
        if m:
            self._current_ns = int(m.group(1))
            self._current_run = None
            self._current_rep = None
            return

        # [INFO]  Run number: X
        m = _RE_RUN.search(line)
        if m:
            self._current_run = int(m.group(1))
            return

        # [INFO]    [rep X] Running: — marks start of a training rep
        m = _RE_REP.search(line)
        if m:
            self._current_rep = int(m.group(1))
            return

        # [webapp] process exited — marks end of a rep
        if "[webapp] process exited" in line and self._current_ns is not None:
            prog = self._per_ns.get(self._current_ns)
            if prog is not None and prog.done < prog.total_reps:
                prog.done += 1
                self._done_steps = min(self._done_steps + 1, self._total_steps)
            return

        # tqdm epoch line
        m = _RE_TQDM.search(line)
        if m:
            epoch, total_epochs = int(m.group(1)), int(m.group(2))
            metrics: dict[str, float] = {}
            for km in _RE_KV.finditer(line):
                key = km.group(1).strip()
                if key not in ("Epochs",):
                    try:
                        metrics[key] = float(km.group(2))
                    except ValueError:
                        pass
            if metrics:
                run_id = (
                    f"ns={self._current_ns}_run={self._current_run}_rep={self._current_rep}"
                )
                self._epoch_metrics.append({
                    "run_id": run_id,
                    "num_samples": self._current_ns,
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    **metrics,
                })
                # Keep last 20 000 points to avoid unbounded growth
                if len(self._epoch_metrics) > 20_000:
                    self._epoch_metrics = self._epoch_metrics[-20_000:]

    # ------------------------------------------------------------------ #

    def progress(self) -> dict:
        overall_pct = (
            round(self._done_steps / self._total_steps * 100, 1)
            if self._total_steps else 0.0
        )
        per_ns = []
        for ns in self._sample_sizes:
            p = self._per_ns.get(ns)
            if p:
                pct = round(p.done / p.total_reps * 100, 1) if p.total_reps else 0.0
                per_ns.append({"num_samples": ns, "done": p.done, "total": p.total_reps, "pct": pct})
        return {
            "overall_pct": overall_pct,
            "done_steps": self._done_steps,
            "total_steps": self._total_steps,
            "current_ns": self._current_ns,
            "current_run": self._current_run,
            "current_rep": self._current_rep,
            "per_ns": per_ns,
        }

    def epoch_metrics(self) -> list[dict]:
        return list(self._epoch_metrics)
