"""Shared matplotlib style and palette for the figure scripts.

Sequential blue (light->dark) encodes magnitude; the ordinal ramp of the same
hue encodes ordered series (larger L is darker). Validated for contrast and
colour-vision deficiency as a set.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE_CMAP = LinearSegmentedColormap.from_list("seq_blue", [SURFACE] + SEQ)
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"          # blue, orange
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8985"


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 9,
        "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "axes.titlesize": 10,
        "axes.grid": True, "grid.color": "#e7e6e2", "grid.linewidth": 0.6,
        "legend.frameon": False, "lines.linewidth": 2.0,
    })


apply()
