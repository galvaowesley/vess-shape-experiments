"""BI-style pivot-table builder over the evaluation results.

`render_table(spec)` turns a `TableSpec` (rows × columns × values + filters +
aggregation) into a flat display table plus ready-to-paste Markdown, CSV and
LaTeX. It reuses the same long dataframe as the chart builder, so tables and
figures always agree.
"""
from __future__ import annotations

import pandas as pd

from . import data
from .schemas import TableSpec

COUNT = "__count__"          # sentinel "value" meaning: count of runs/rows
_REDUCERS = {"mean", "std", "median", "min", "max"}


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def _filter(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    if spec.exclude_mock and "is_mock" in df.columns:
        df = df[~df["is_mock"].fillna(False)]
    if spec.model_types and "model_type" in df.columns:
        df = df[df["model_type"].isin(spec.model_types)]
    if spec.stages and "stage" in df.columns:
        df = df[df["stage"].isin(spec.stages)]
    if spec.regimes and "regime" in df.columns:
        df = df[df["regime"].isin(spec.regimes)]
    if spec.sample_filter and "num_samples" in df.columns:
        df = df[df["num_samples"].isin(spec.sample_filter)]
    return df


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _fmt(v, decimals: int, pct: bool):
    if v is None or pd.isna(v):
        return None
    if pct:
        v = v * 100.0
    return f"{v:.{decimals}f}"


def _fmt_mean_std(m, s, decimals: int, pct: bool):
    if m is None or pd.isna(m):
        return None
    mean = _fmt(m, decimals, pct)
    if s is None or pd.isna(s):
        return mean
    return f"{mean} ± {_fmt(s, decimals, pct)}"


def _map_cells(df: pd.DataFrame, fn) -> pd.DataFrame:
    """Element-wise map across every cell (applymap replacement, pandas-2.2 safe)."""
    return df.apply(lambda col: col.map(fn))


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _flat(df: pd.DataFrame, spec: TableSpec, rows: list[str]) -> pd.DataFrame:
    """rows × values (no column pivot). Supports multiple measures."""
    g = df.groupby(rows, dropna=False)
    display: dict[str, pd.Series] = {}
    numeric: dict[str, pd.Series] = {}
    for val in spec.values:
        if val == COUNT or spec.aggregation == "count":
            size = g.size()
            display["count" if val == COUNT else val] = size.astype(int)
            numeric["count" if val == COUNT else val] = size
            continue
        if val not in df.columns:
            continue
        if spec.aggregation == "mean_std":
            mean, std = g[val].mean(), g[val].std()
            display[val] = pd.Series(
                [_fmt_mean_std(a, b, spec.decimals, spec.percentage) for a, b in zip(mean, std)],
                index=mean.index,
            )
            numeric[val] = mean
        else:  # mean / std / median / min / max
            series = getattr(g[val], spec.aggregation)()
            display[val] = pd.Series(
                [_fmt(v, spec.decimals, spec.percentage) for v in series], index=series.index
            )
            numeric[val] = series

    disp = pd.DataFrame(display)
    if spec.sort_by and spec.sort_by in numeric:
        order = pd.DataFrame(numeric).sort_values(spec.sort_by, ascending=spec.ascending).index
        disp = disp.loc[order]
    return disp.reset_index()


def _pivot(df: pd.DataFrame, spec: TableSpec, rows: list[str], cols: list[str]) -> pd.DataFrame:
    """rows × (column dimension) for a single measure."""
    value = spec.values[0] if spec.values else COUNT
    if value == COUNT or spec.aggregation == "count":
        pv = df.pivot_table(index=rows, columns=cols, aggfunc="size", fill_value=0)
        out = pv.astype(int).astype(object)
    elif spec.aggregation == "mean_std":
        mean = df.pivot_table(index=rows, columns=cols, values=value, aggfunc="mean")
        std = df.pivot_table(index=rows, columns=cols, values=value, aggfunc="std")
        out = mean.copy().astype(object)
        for r in mean.index:
            for c in mean.columns:
                out.loc[r, c] = _fmt_mean_std(
                    mean.loc[r, c], std.loc[r, c], spec.decimals, spec.percentage
                )
    else:
        pv = df.pivot_table(index=rows, columns=cols, values=value, aggfunc=spec.aggregation)
        out = _map_cells(pv, lambda v: _fmt(v, spec.decimals, spec.percentage))

    out.columns = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in out.columns]
    return out.reset_index()


def _build(spec: TableSpec) -> pd.DataFrame:
    df = data.load_long_dataframe(spec.source, spec.datasets or None)
    if df.empty:
        return pd.DataFrame()
    df = _filter(df, spec)
    rows = [r for r in spec.rows if r in df.columns] or ["model_type"]
    cols = [c for c in spec.columns if c in df.columns]
    if df.empty:
        return pd.DataFrame(columns=rows)
    table = _pivot(df, spec, rows, cols) if cols else _flat(df, spec, rows)
    # tidy: NaN -> None for JSON, prettify dimension headers
    return table.where(pd.notna(table), None)


# --------------------------------------------------------------------------- #
# Exporters (kept identical to what the preview shows — WYSIWYG)
# --------------------------------------------------------------------------- #
def _cell(v) -> str:
    return "" if v is None else str(v)


def to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def to_latex(df: pd.DataFrame, caption: str) -> str:
    safe = df.copy()
    safe.columns = [str(c).replace("_", r"\_") for c in safe.columns]
    safe = _map_cells(safe, lambda v: _cell(v).replace("±", r"$\pm$").replace("_", r"\_"))
    latex = safe.to_latex(index=False, escape=False, column_format="l" + "c" * (safe.shape[1] - 1))
    if caption:
        latex = f"% {caption}\n{latex}"
    return latex


def render_table(spec: TableSpec) -> dict:
    df = _build(spec)
    caption = spec.title or _auto_caption(spec)
    columns = [str(c) for c in df.columns]
    rows = [[None if pd.isna(v) else v for v in row] for row in df.itertuples(index=False)]
    return {
        "columns": columns,
        "rows": rows,
        "caption": caption,
        "n_rows": int(len(df)),
        "markdown": to_markdown(df),
        "csv": df.to_csv(index=False),
        "latex": to_latex(df, caption),
    }


def _auto_caption(spec: TableSpec) -> str:
    measures = ", ".join("count" if v == COUNT else v for v in spec.values)
    layout = " × ".join(spec.rows + spec.columns)
    agg = spec.aggregation.replace("_", " ± ") if spec.aggregation == "mean_std" else spec.aggregation
    cap = f"{agg} of {measures} by {layout}"
    if spec.sample_filter:
        cap += f" · N ∈ {sorted(spec.sample_filter)}"
    return cap
