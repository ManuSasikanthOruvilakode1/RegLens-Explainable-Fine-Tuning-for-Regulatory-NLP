"""Shared matplotlib styling for the analysis notebooks.

Categorical colours are used in a fixed order (blue, orange, aqua) and were checked for colour-vision
deficiency separation. Every bar carries a direct value label and each chart is paired with a table,
so identity and value never depend on colour alone.
"""

from typing import Optional, Sequence

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
MUTED = "#a3a29c"          # reference / baseline marks
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3, fixed order


def apply_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.labelcolor": TEXT_SECONDARY, "axes.edgecolor": GRID,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY, "text.color": TEXT_PRIMARY, "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def hbar_groups(ax, categories: Sequence[str], series: dict, errors: Optional[dict] = None,
                fmt: str = "{:.1f}", xlabel: str = "", colors: Optional[Sequence[str]] = None,
                xlim: Optional[tuple] = None) -> None:
    """Horizontal grouped bars, one group per category, with value labels and optional error bars.

    series: {series_name: [value per category]} - colours follow series order, never rank.
    """
    import numpy as np
    n = len(series)
    height = 0.8 / n
    y = np.arange(len(categories))
    colors = list(colors or SERIES)
    if xlim:
        ax.set_xlim(*xlim)
    for i, (name, values) in enumerate(series.items()):
        pos = y - 0.4 + height * (i + 0.5)
        err = errors.get(name) if errors else None
        ax.barh(pos, values, height=height - 0.04, color=colors[i % len(colors)], label=name,
                xerr=err, error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1, "capsize": 2})
        # symmetric errors are one list; asymmetric (e.g. CI bounds) are [lower list, upper list]
        upper = err[1] if err is not None and len(err) == 2 and isinstance(err[0], (list, tuple)) else err
        for p, v, e in zip(pos, values, upper if upper is not None else [0] * len(values)):
            if v is None or v != v:
                continue
            ax.text(v + (e or 0) + ax.get_xlim()[1] * 0.01, p, fmt.format(v), va="center",
                    fontsize=8.5, color=TEXT_PRIMARY)
    ax.set_yticks(y, categories)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel, loc="right")
    handles, _ = ax.get_legend_handles_labels()
    if n > 1 or len(handles) > n:   # several series, or extra reference lines drawn before this call
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=len(handles), fontsize=9,
                  handlelength=1.4, columnspacing=1.4, borderaxespad=0)
