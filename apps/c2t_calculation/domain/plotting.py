from matplotlib.axes import Axes
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from _domain.models import IonData, LoadableScanData
from _domain.plotting import plot_ScanData



def plot_calculated_scan(ax: Axes, data: LoadableScanData) -> None:
    label = f"{data.file_path.stem}"
    plot_ScanData(ax, data, label)

def plot_ions_square(
    ax: Axes,
    ion_data: IonData,
    *,
    color: str = "red",
    label: str | None = None,
    bins: int = 200,
    hist_size: str = "25%",   # z.B. "20%" oder "1.2in"
    pad: float = 0.08,
) -> tuple[Axes, Axes]:
    xs = np.array([p.x for p in ion_data.points], dtype=float)
    ys = np.array([p.y for p in ion_data.points], dtype=float)

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