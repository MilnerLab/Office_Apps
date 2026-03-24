from matplotlib.axes import Axes
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from _domain.plotting import plot_ScanData
from base_core.lab_specifics.base_models import C2TScanData, Points



def plot_calculated_scan(ax: Axes, data: C2TScanData,label:str = '',*,number_of_scans: int = 1) -> None:
    #if data.file_path is not None:
        #label = f"{data.file_path.stem}"
    label =  f"{number_of_scans} scans" 
    plot_ScanData(ax, data, label)

def plot_ions_square(
    ax: Axes,
    points: Points,
    *,
    color: str = "red",
    label: str | None = None,
    bins: int = 200,
    hist_size: str = "25%",   # z.B. "20%" oder "1.2in"
    pad: float = 0.08,
) -> tuple[Axes, Axes]:
    xs = points.x
    ys = points.y

    ax.scatter(xs, ys, color=color, marker=".", s=5, label=label, alpha=0.2)
    ax.set_aspect("equal", adjustable="box")

    if xs.size and ys.size:
        lim_min = float(min(xs.min(), ys.min()))
        lim_max = float(max(xs.max(), ys.max()))
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)

    # Achsen für Marginal-Histogramme anlegen
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top",   size=hist_size, pad=pad, sharex=ax)
    ax_histy = divider.append_axes("right", size=hist_size, pad=pad, sharey=ax)

    # Histogramme (Range passend zu den Scatter-Limits)
    ax_histx.hist(xs, bins=bins, range=ax.get_xlim(), color=color, alpha=0.7)
    ax_histy.hist(ys, bins=bins, range=ax.get_ylim(), orientation="horizontal",
                  color=color, alpha=0.7)

    # Optik: Tick-Labels auf den geteilten Achsen ausblenden
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # Optional: ein bisschen aufräumen
    ax_histx.grid(False)
    ax_histy.grid(False)

    return ax_histx, ax_histy