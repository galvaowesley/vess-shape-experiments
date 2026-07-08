"""Generic publication-quality matplotlib renderer.

`render_figure(spec)` is the single code path for both the live preview (SVG)
and the exported paper figure (svg/png/jpg/pdf) — so what you see is exactly
what you export. It is generic (any column on any axis) but defaults its colors
and dashes from the same `model_colors.yaml` palette the notebooks use, so a
fresh chart already matches the paper look.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from . import data, palettes, style  # noqa: E402
from .schemas import PlotSpec  # noqa: E402

_MIME = {
    "svg": "image/svg+xml",
    "png": "image/png",
    "jpg": "image/jpeg",
    "pdf": "application/pdf",
}

_DEFAULT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", ">", "<"]


def _model_palette() -> dict:
    try:
        import utils_eval
        return utils_eval.load_model_palette() or {}
    except Exception:
        return {}


def _line_styles() -> dict:
    try:
        import utils_eval
        return utils_eval.default_line_styles_7way() or {}
    except Exception:
        return {}


def _apply_filters(df: pd.DataFrame, spec: PlotSpec) -> pd.DataFrame:
    d = spec.data
    if d.model_types and "model_type" in df.columns:
        df = df[df["model_type"].isin(d.model_types)]
    if d.stages and "stage" in df.columns:
        df = df[df["stage"].isin(d.stages)]
    if d.regimes and "regime" in df.columns:
        df = df[df["regime"].isin(d.regimes)]
    if d.num_samples_range and "num_samples" in df.columns:
        lo, hi = d.num_samples_range
        ns = pd.to_numeric(df["num_samples"], errors="coerce")
        df = df[(ns >= lo) & (ns <= hi)]
    return df


def _resolve_colors(series_names: list[str], spec: PlotSpec) -> dict[str, str]:
    """figure override -> global style colors -> named palette -> model_colors.yaml -> mpl cycle.

    The global style colors win over a chosen seaborn palette so a model/dataset
    keeps the same color across every chart; the palette only fills series that
    have no global color set.
    """
    colors: dict[str, str] = {}
    pal_list = palettes.resolve(spec.palette, len(series_names)) if spec.palette else None
    global_colors = style.color_map()
    model_pal = _model_palette()
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i, name in enumerate(series_names):
        override = spec.series_styles.get(name)
        if override and override.color:
            colors[name] = override.color
        elif name in global_colors:
            colors[name] = global_colors[name]
        elif pal_list is not None:
            colors[name] = pal_list[i % len(pal_list)]
        elif name in model_pal:
            colors[name] = model_pal[name]
        else:
            colors[name] = cycle[i % len(cycle)] if cycle else "#1f77b4"
    return colors


def _ordered(names: list[str], *preferred: list[str]) -> list[str]:
    """Reorder `names` so values from the `preferred` lists come first (in that
    order, earlier lists win), with the rest appended in their original order."""
    nameset = set(names)
    seen: set = set()
    out: list[str] = []
    for pref in preferred:
        for n in pref or []:
            if n in nameset and n not in seen:
                out.append(n)
                seen.add(n)
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def _aggregate_datasets(df: pd.DataFrame, spec: PlotSpec) -> pd.DataFrame:
    """Collapse the selected datasets into a single mean curve vs the x axis.

    `macro` averages per-dataset means (each dataset weighs equally); the
    downstream mean/std then runs *across datasets*, so the error band reflects
    the spread between datasets. `pool` keeps the rows untouched (the existing
    mean over all rows already pools them). Skipped when dataset is itself an
    axis/series.
    """
    mode = spec.aggregate_datasets
    enc = spec.encoding
    if mode == "none" or df.empty or "dataset" not in df.columns:
        return df
    if enc.x == "dataset" or enc.series == "dataset":
        return df
    if mode == "pool":
        return df
    if enc.y not in df.columns:
        return df
    keys: list[str] = []
    for c in (enc.series, enc.x, spec.facet):
        if c and c != "dataset" and c in df.columns and c not in keys:
            keys.append(c)
    if not keys:
        return df
    return df.groupby(keys + ["dataset"], dropna=False)[enc.y].mean().reset_index()


def _style_for(name: str, spec: PlotSpec, line_styles: dict, idx: int):
    ov = spec.series_styles.get(name)
    ls = (ov.linestyle if ov and ov.linestyle else None) or line_styles.get(name, "-")
    lw = (ov.width if ov and ov.width else None) or 2.0
    marker = ov.marker if (ov and ov.marker is not None) else _DEFAULT_MARKERS[idx % len(_DEFAULT_MARKERS)]
    return ls, lw, marker


def _scale(series: pd.Series, percentage: bool) -> pd.Series:
    return series * 100.0 if percentage else series


def _err(agg: pd.DataFrame, error_type: str) -> pd.Series:
    """Error measure from an aggregate with 'std' and 'count' columns:
    standard deviation (±std) or standard error of the mean (±std/sqrt(n))."""
    std = agg["std"]
    if error_type == "sem":
        cnt = agg["count"].replace(0, np.nan)
        return std / np.sqrt(cnt)
    return std


def _draw_axis(ax, fdf: pd.DataFrame, spec: PlotSpec, colors: dict, line_styles: dict,
               series_names: list[str]):
    enc = spec.encoding
    xcol, ycol, scol = enc.x, enc.y, enc.series
    pct = spec.y_axis.percentage

    if scol and scol in fdf.columns:
        gmap = {str(k): g for k, g in fdf.groupby(scol)}
        groups = [(n, gmap[n]) for n in series_names if n in gmap]
    else:
        groups = [(enc.y, fdf)]

    if spec.chart_type == "bar":
        raw_cats = fdf[xcol].dropna().unique().tolist()
        try:
            raw_cats = sorted(raw_cats, key=lambda v: (str(type(v)), v))
        except TypeError:
            raw_cats = sorted(raw_cats, key=str)
        # honor the saved/override category order while keeping original dtypes
        by_str = {str(c): c for c in raw_cats}
        order_pref = _ordered(list(by_str.keys()), spec.category_order, style.order_for(xcol))
        cats = [by_str[s] for s in order_pref]
        n_series = max(len(groups), 1)
        width = 0.8 / n_series
        x_base = np.arange(len(cats))
        show_err = spec.show_error_band
        for i, (name, g) in enumerate(groups):
            agg = g.groupby(xcol)[ycol].agg(["mean", "std", "count"]).reindex(cats)
            means = _scale(agg["mean"], pct).values
            errs = _scale(_err(agg, spec.error_type).fillna(0.0), pct).values if show_err else None
            offset = (i - (n_series - 1) / 2) * width
            ax.bar(x_base + offset, means, width=width, label=str(name),
                   color=colors.get(name), yerr=errs, capsize=3 if show_err else 0,
                   error_kw={"elinewidth": 1, "alpha": 0.6})
        ax.set_xticks(x_base)
        ax.set_xticklabels([str(c) for c in cats])
        return

    for i, (name, g) in enumerate(groups):
        ls, lw, marker = _style_for(name, spec, line_styles, i)
        color = colors.get(name)
        if spec.chart_type == "scatter":
            ax.scatter(g[xcol], _scale(g[ycol], pct), label=str(name), color=color,
                       marker=marker, s=spec.marker_size, alpha=0.8, edgecolors="none")
        else:  # line: aggregate mean +/- error over x
            agg = g.groupby(xcol)[ycol].agg(["mean", "std", "count"]).reset_index().sort_values(xcol)
            x = agg[xcol].values
            mean = _scale(agg["mean"], pct).values
            ax.plot(x, mean, label=str(name), color=color, linestyle=ls,
                    linewidth=lw, marker=marker, markersize=np.sqrt(spec.marker_size))
            err = _err(agg, spec.error_type)
            if spec.show_error_band and err.notna().any():
                e = _scale(err.fillna(0.0), pct).values
                ax.fill_between(x, mean - e, mean + e, color=color,
                                alpha=spec.error_alpha, linewidth=0)


def render_figure(spec: PlotSpec) -> tuple[bytes, str]:
    df = data.load_long_dataframe(spec.data.source, spec.data.datasets)
    df = _apply_filters(df, spec)
    df = _aggregate_datasets(df, spec)

    base = float(spec.fonts.get("base", 12.0))
    line_styles = _line_styles()

    # When collapsing across datasets, never facet by dataset (defeats the mean).
    facet = None if (spec.aggregate_datasets != "none" and spec.facet == "dataset") else spec.facet

    # facets -> side-by-side subplots
    if facet and facet in df.columns and not df.empty:
        facet_vals = sorted(df[facet].dropna().unique().tolist(), key=str)
    else:
        facet_vals = [None]

    # consistent, user-ordered color/series mapping across all facets
    scol = spec.encoding.series
    if scol and scol in df.columns:
        series_names = sorted(df[scol].dropna().astype(str).unique().tolist(), key=str)
        series_names = _ordered(series_names, spec.series_order, style.order_for(scol))
    else:
        series_names = [spec.encoding.y]
    colors = _resolve_colors(series_names, spec)

    with plt.rc_context({"font.size": base, "svg.fonttype": "none"}):
        w, h = spec.figure.size
        fig, axes = plt.subplots(1, len(facet_vals), figsize=(w, h),
                                 dpi=spec.export.dpi, squeeze=False)
        axes = axes[0]

        for ax, fval in zip(axes, facet_vals):
            fdf = df if fval is None else df[df[facet] == fval]
            if not fdf.empty:
                _draw_axis(ax, fdf, spec, colors, line_styles, series_names)

            # axes labels / limits / ticks
            xa, ya = spec.x_axis, spec.y_axis
            ax.set_xlabel(xa.label if xa.label is not None else spec.encoding.x,
                          fontsize=xa.label_fontsize)
            ax.set_ylabel(ya.label if ya.label is not None else spec.encoding.y,
                          fontsize=ya.label_fontsize)
            ax.tick_params(axis="x", labelsize=xa.tick_fontsize, rotation=xa.tick_rotation)
            ax.tick_params(axis="y", labelsize=ya.tick_fontsize)
            if xa.min is not None or xa.max is not None:
                ax.set_xlim(left=xa.min, right=xa.max)
            if ya.min is not None or ya.max is not None:
                ax.set_ylim(bottom=ya.min, top=ya.max)
            # fixed numeric tick spacing (e.g. every 2 instead of 2.5)
            if xa.tick_step and xa.tick_step > 0 and spec.chart_type != "bar":
                ax.xaxis.set_major_locator(MultipleLocator(xa.tick_step))
            if ya.tick_step and ya.tick_step > 0:
                ax.yaxis.set_major_locator(MultipleLocator(ya.tick_step))
            if spec.show_grid:
                ax.grid(True, color="#e2e8f0", linewidth=0.8, alpha=0.9)
                ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if fval is not None:
                ax.set_title(str(fval), fontsize=base + 1)

        # legend (deduplicated across facets)
        if spec.legend.show:
            handles, labels = [], []
            for ax in axes:
                for hh, ll in zip(*ax.get_legend_handles_labels()):
                    if ll not in labels:
                        handles.append(hh)
                        labels.append(ll)
            if handles:
                if len(axes) == 1:
                    axes[0].legend(handles, labels, title=spec.legend.title,
                                   loc=spec.legend.loc, fontsize=spec.legend.fontsize,
                                   title_fontsize=spec.legend.fontsize)
                else:
                    fig.legend(handles, labels, title=spec.legend.title,
                               loc="center left", bbox_to_anchor=(1.0, 0.5),
                               fontsize=spec.legend.fontsize,
                               title_fontsize=spec.legend.fontsize)

        if spec.title.text:
            fig.suptitle(spec.title.text, fontsize=spec.title.fontsize)

        fig.tight_layout()
        fmt = spec.export.format
        save_fmt = "jpeg" if fmt == "jpg" else fmt
        buf = io.BytesIO()
        face = "white" if fmt in ("png", "jpg") else "none"
        fig.savefig(buf, format=save_fmt, bbox_inches="tight",
                    facecolor=face, dpi=spec.export.dpi)
        plt.close(fig)
        return buf.getvalue(), _MIME[fmt]
