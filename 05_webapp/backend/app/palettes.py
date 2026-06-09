"""Seaborn named-palette catalog resolved to hex swatch lists for the UI."""
from __future__ import annotations

from functools import lru_cache

import matplotlib.colors as mcolors
import seaborn as sns

# Curated catalog grouped by kind. Names are valid `seaborn.color_palette` args.
CATALOG: dict[str, list[str]] = {
    "qualitative": ["deep", "muted", "pastel", "dark", "colorblind", "bright",
                    "Set1", "Set2", "Set3", "tab10", "husl", "hls"],
    "sequential": ["viridis", "rocket", "mako", "crest", "flare", "magma", "Blues", "Greens"],
    "diverging": ["vlag", "icefire", "coolwarm", "Spectral"],
}

_SWATCH_N = 8  # how many colors to preview per palette


def _to_hex(rgb) -> str:
    return mcolors.to_hex(rgb)


@lru_cache(maxsize=1)
def catalog_with_swatches() -> dict:
    """Return {group: [{name, colors:[hex...]}]} for the palette dropdown."""
    out: dict[str, list] = {}
    for group, names in CATALOG.items():
        entries = []
        for name in names:
            try:
                colors = [_to_hex(c) for c in sns.color_palette(name, _SWATCH_N)]
            except Exception:
                continue
            entries.append({"name": name, "colors": colors})
        out[group] = entries
    return out


def resolve(name: str | None, n: int) -> list[str] | None:
    """Resolve a named palette to `n` hex colors (cycled if needed)."""
    if not name or n <= 0:
        return None
    try:
        pal = sns.color_palette(name, n)
    except Exception:
        return None
    return [_to_hex(c) for c in pal]
