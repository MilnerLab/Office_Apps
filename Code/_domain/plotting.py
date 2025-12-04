from matplotlib.axes import Axes
import numpy as np

from Lab_apps._base.fitting import fit_gaussian
from Lab_apps._base.models import PlotColor, Prefix
from Lab_apps._domain.models import ScanDataBase


def plot_ScanData(ax: Axes, data: ScanDataBase, label:str, color: PlotColor = PlotColor.RED) -> None:
    x = [time.value(Prefix.PICO) for time in data.delay]
    y = np.array([c.value for c in data.c2t])
    error = np.array([c.error for c in data.c2t])

    ax.plot(
        x,
        y,
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=3,
        color=color,
        label=label,
    )

    ax.fill_between(
        x,
        y - error,
        y + error,
        color=color,
        alpha=0.2,
        linewidth=0,
    )

    ax.set_xlabel("Probe Delay (ps)")
    ax.set_ylabel(r"$\langle \cos^2 \theta_\mathrm{2D} \rangle$")
    ax.grid(True)
    
def plot_GaussianFit(ax: Axes, data: ScanDataBase) -> None:
    gauss = fit_gaussian(data.delay, [c2t.value for c2t in data.c2t])
    
    ax.plot([time.value(Prefix.PICO) for time in data.delay], [y for y in gauss.get_curve(data.delay)])
    