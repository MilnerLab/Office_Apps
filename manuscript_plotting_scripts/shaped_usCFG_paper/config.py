# manuscript_plotting_scripts/shaped_usCFG_paper/config.py
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure

# --- Locations ---------------------------------------------------------------
DATA_ROOT = Path(r"Z:\Droplets\shaped_usCFG_paper")
OFFICE_APPS = Path(__file__).resolve().parents[2]
TEMP_DIR = OFFICE_APPS / "_temp" / "shaped_usCFG_paper"
FIGURES_DIR = OFFICE_APPS.parent / "Latex" / "shaped_usCFG_paper" / "figures"
INPUTS_DIR = Path(__file__).resolve().parent / "domain" / "inputs"

XCORR_SCAN_L = DATA_ROOT / "20260825" / "XCORR_scan_L_20260825_200235.h5"
XCORR_SCAN_D = DATA_ROOT / "20260831" / "XCORR_scan_d_20260831_131421.h5"
JET_ROOT = DATA_ROOT / "Jet"
TRUNCATION_ROOT = DATA_ROOT / "truncation"

# --- Figure sizes (inches) ---------------------------------------------------
# Each figure is saved at exactly its V28 size so the manuscript layout does not move.
FIGURE_SIZES_IN: dict[str, tuple[float, float]] = {
    "cfg_arms_truncation": (6.75, 2.55),
    "fig_char_fitdemo": (3.2514, 2.1888),
    "fig_char_freqtime": (3.3797, 3.8827),
    "fig_char_scans": (6.1601, 2.9972),
    "fig_char_truncation": (3.4, 5.2),
    "fig_jet_oscillations": (3.375, 5.15),
    "fig_jet_truncation": (3.375, 2.5),
}

# --- Style (manuscript_plotting_scripts/figuremaker.py) -----------------------
mpl.rcParams.update({
    
    # --- Axes ---
    "axes.titlesize": "medium",
    "axes.labelsize": 6,
    "axes.formatter.use_mathtext": True,
    "axes.linewidth": 0.5,
    "axes.grid": True,
    "axes.grid.axis": "both",  # which axis the grid should apply to
    "axes.grid.which": "major",
    "axes.axisbelow" : True,
    "grid.alpha": 1.0,

    # --- Grid lines ---
    "grid.linewidth": 0.3,
    "grid.linestyle": "solid",
    "grid.color": "grey",

    # --- Lines ---
    "lines.linewidth": 0.5,
    "lines.marker": "o",
    "lines.markersize": 1.0,
    "hatch.linewidth": 0.25,
    "patch.antialiased": True,
    
    #---Errorbars---
    "errorbar.capsize": 1,

    # --- Ticks (X) ---
    #"xtick.top": True,
    "xtick.bottom": True,
    "xtick.major.size": 3.0,
    "xtick.minor.size": 1.5,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "xtick.direction": "in",
    "xtick.minor.visible": False,
    #"xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.bottom": True,
    "xtick.major.pad": 5.0,
    "xtick.minor.pad": 5.0,
    "xtick.labelsize": 6,

    # --- Ticks (Y) ---
    "ytick.left": True,
    #"ytick.right": True,
    "ytick.major.size": 3.0,
    #"ytick.minor.size": 1.5,
    "ytick.major.width": 0.5,
    #"ytick.minor.width": 0.5,
    "ytick.direction": "in",
    #"ytick.minor.visible": True,
    "ytick.major.left": True,
    #"ytick.major.right": True,
    #"ytick.minor.left": True,
    "ytick.major.pad": 2.0,
    #"ytick.minor.pad": 5.0,
    "ytick.labelsize": 6,
    
    
    # --- Legend ---
    "legend.frameon": True,
    "legend.fontsize": 6,
    "legend.handlelength": 1.375,
    "legend.labelspacing": 0.4,
    "legend.columnspacing": 1,
    "legend.facecolor": "white",
    "legend.edgecolor": "white",
    "legend.framealpha": 1,
    "legend.title_fontsize": 6,
 
    # --- Figure size ---
    #"figure.figsize": (3.375, 3),
    #"figure.figsize": (6.75, 3.6), #approx. 2- column fig
    "figure.subplot.left": 0.125,
    "figure.subplot.bottom": 0.175,
    "figure.subplot.top": 0.95,
    "figure.subplot.right": 0.95,

    # --- Fonts (computer modern) ---
    "font.size": 6,
    "text.usetex": True,      
    #"mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"]

})


def save_figure(fig: Figure, name: str) -> Path:
    fig.set_size_inches(*FIGURE_SIZES_IN[name])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path)
    return path


TEMP_DIR.mkdir(parents=True, exist_ok=True)
