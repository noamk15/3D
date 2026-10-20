"""Shared figure style (validated categorical palette, thin marks, recessive grid)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
        "ytick.color": TEXT2, "text.color": TEXT, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.markersize": 6, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
        "figure.dpi": 110,
    })


def line(ax, x, y, k, label, yerr=None):
    ax.plot(x, y, color=SERIES[k], marker=MARKERS[k], label=label,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    if yerr is not None:
        lo, hi = yerr
        ax.fill_between(x, lo, hi, color=SERIES[k], alpha=0.12, linewidth=0)
