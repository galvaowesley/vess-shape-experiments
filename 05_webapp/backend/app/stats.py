"""Flexible Wilcoxon signed-rank significance grid.

Generalizes the notebook's fixed 4-pair comparison: a chosen reference model is
compared against any set of comparison models, across datasets, at any set of N
values (each N rendered as its own panel — N=0 Zero-shot, N=1 One-shot, ...).
Paired per-image metrics are tested with `scipy.stats.wilcoxon`; the alpha,
alternative and minimum overlap are user-controlled.
"""
from __future__ import annotations

import io
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import data, paths  # noqa: E402
from .schemas import WilcoxonSpec  # noqa: E402

_MIME = {"svg": "image/svg+xml", "png": "image/png", "jpg": "image/jpeg", "pdf": "application/pdf"}


def _per_image(datasets: list[str]) -> dict[str, pd.DataFrame]:
    """Per-image frames keyed by dataset (from results/<ds>_all_results_per_image.csv)."""
    out: dict[str, pd.DataFrame] = {}
    for ds in datasets:
        df = data.load_long_dataframe("per_image", [ds])
        if not df.empty:
            out[ds] = df
    return out


def _zero_shot_map() -> dict[str, str]:
    try:
        import utils_eval
        return dict(utils_eval.FINE_TUNED_TO_ZEROSHOT)
    except Exception:
        return {
            "VSUNet18": "Zero-Shot VSUNet18",
            "VSUNet50": "Zero-Shot VSUNet50",
            "IN-UNet18": "Zero-Shot IN-UNet18",
            "IN-UNet50": "Zero-Shot IN-UNet50",
            "LiteMedSAM-FT": "Zero-Shot LiteMedSAM",
        }


def _by_image(df: pd.DataFrame, model: str, n: int, metric: str, zs_map: dict) -> pd.Series | None:
    if n == 0:
        zs = zs_map.get(model, model)
        sub = df[(df["model_type"] == zs) & (~df.get("is_mock", False).astype(bool))]
    else:
        sub = df[(df["model_type"] == model) & (df["num_samples"] == n)
                 & (df["stage"] == "finetune")]
    if sub.empty or "image" not in sub.columns:
        return None
    return sub.groupby("image")[metric].mean()


def _pvalue(df: pd.DataFrame, ref: str, comp: str, n: int, spec: WilcoxonSpec, zs_map: dict) -> float:
    from scipy import stats
    a = _by_image(df, ref, n, spec.metric, zs_map)
    b = _by_image(df, comp, n, spec.metric, zs_map)
    if a is None or b is None:
        return float("nan")
    common = a.index.intersection(b.index)
    if len(common) < spec.min_common_images:
        return float("nan")
    av, bv = a[common].values, b[common].values
    if np.allclose(av, bv):
        return 1.0
    try:
        _, p = stats.wilcoxon(av, bv, alternative=spec.alternative)
    except ValueError:
        return float("nan")
    return float(p)


def _regime_label(n: int) -> str:
    return {0: "Zero-shot", 1: "One-shot"}.get(n, f"N={n}")


def compute_panels(spec: WilcoxonSpec) -> dict:
    """Return p-value matrices (one per N) as plain lists for inspection/JSON."""
    datasets = spec.datasets or data.DATASETS
    per_img = _per_image(datasets)
    zs_map = _zero_shot_map()

    if spec.row_axis == "dataset":
        rows, cols = datasets, spec.comparison_models
    else:
        rows, cols = spec.comparison_models, datasets

    panels = []
    for n in spec.n_values:
        matrix = []
        for r in rows:
            row_vals = []
            for c in cols:
                ds = r if spec.row_axis == "dataset" else c
                comp = c if spec.row_axis == "dataset" else r
                df = per_img.get(ds, pd.DataFrame())
                p = _pvalue(df, spec.reference_model, comp, n, spec, zs_map) if not df.empty else float("nan")
                row_vals.append(p)
            matrix.append(row_vals)
        panels.append({"n": n, "label": _regime_label(n),
                       "rows": rows, "cols": cols, "pvalues": matrix})
    return {"reference": spec.reference_model, "metric": spec.metric,
            "alpha": spec.alpha, "panels": panels}


def render_wilcoxon(spec: WilcoxonSpec) -> tuple[bytes, str]:
    result = compute_panels(spec)
    panels = result["panels"]
    base = float(spec.fonts.get("base", 12.0))

    with plt.rc_context({"font.size": base, "svg.fonttype": "none"}):
        w, h = spec.figure.size
        fig, axes = plt.subplots(1, max(len(panels), 1), figsize=(w, h),
                                 dpi=spec.export.dpi, squeeze=False)
        axes = axes[0]

        for ax, panel in zip(axes, panels):
            rows, cols = panel["rows"], panel["cols"]
            pv = np.array(panel["pvalues"], dtype=float)
            # color code: 2 = significant, 1 = not significant, 0 = nan
            code = np.where(np.isnan(pv), 0, np.where(pv < spec.alpha, 2, 1))
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap([spec.colors.nan, spec.colors.nonsignificant, spec.colors.significant])
            ax.imshow(code, cmap=cmap, vmin=0, vmax=2, aspect="auto")

            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=base - 1)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels(rows, fontsize=base - 1)
            ax.set_title(panel["label"], fontsize=base + 1)
            ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.tick_params(which="minor", length=0)

            for i in range(len(rows)):
                for j in range(len(cols)):
                    p = pv[i, j]
                    if math.isnan(p):
                        ax.text(j, i, "—", ha="center", va="center",
                                color="#d62728", fontsize=base)
                    elif spec.annotate_pvalues:
                        ax.text(j, i, f"{p:.3f}", ha="center", va="center",
                                color="white" if p < spec.alpha else "#1e293b",
                                fontsize=base - 2)

        title = spec.title.text or (
            f"Wilcoxon: {spec.reference_model} {spec.alternative} others — {spec.metric} "
            f"(α={spec.alpha})")
        fig.suptitle(title, fontsize=spec.title.fontsize)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        fmt = spec.export.format
        save_fmt = "jpeg" if fmt == "jpg" else fmt
        buf = io.BytesIO()
        face = "white" if fmt in ("png", "jpg") else "none"
        fig.savefig(buf, format=save_fmt, bbox_inches="tight", facecolor=face, dpi=spec.export.dpi)
        plt.close(fig)
        return buf.getvalue(), _MIME[fmt]
