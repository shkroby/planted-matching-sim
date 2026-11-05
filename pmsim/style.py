"""Matplotlib style matching the figures of the write-up.

Serif Computer-Modern text, slate-grey left/bottom spines, light horizontal
grid, open circular markers, muted blue / rust / grey palette.
Set PMSIM_USETEX=1 to render with a system LaTeX.
"""

from __future__ import annotations

import os
import shutil

import matplotlib as mpl

BLUE = "#1f5f9e"
RUST = "#c0562b"
GREY = "#5a5a5a"
GREEN = "#1b7f62"
PURPLE = "#9a5ea6"
GOLD = "#c79a2e"
SLATE = "#4c5b6b"
SPINE = "#5f6577"
TICK = "#6b7080"
GRID = "#e4e4e4"
CYCLE = [BLUE, GREEN, RUST, PURPLE, GOLD, SLATE]
SEQ = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]


def apply():
    usetex = os.environ.get("PMSIM_USETEX", "0") == "1" and shutil.which("latex")
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "Latin Modern Roman",
                       "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": bool(usetex),
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "xtick.color": SPINE,
        "ytick.color": SPINE,
        "xtick.labelcolor": TICK,
        "ytick.labelcolor": TICK,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.5,
        "lines.markeredgewidth": 1.3,
        "legend.frameon": False,
        "errorbar.capsize": 2.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 110,
    })


def marker(color, ls="-", m="o"):
    """Line style kwargs: open markers with white fill, as in the write-up."""
    return dict(color=color, ls=ls, marker=m, mfc="white", mec=color)
