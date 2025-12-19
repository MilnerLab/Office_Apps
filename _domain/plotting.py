from matplotlib.axes import Axes
from base_core.fitting.functions import fit_gaussian
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
import numpy as np
from _domain.models import ScanDataBase


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
    
    resampling_const = len(data.delay) * 100
    
    y = [y for y in gauss.get_curve(np.linspace(data.delay[0], data.delay[-1], resampling_const))]
    x = np.linspace(data.delay[0].value(Prefix.PICO), data.delay[-1].value(Prefix.PICO), resampling_const)
    
    ax.plot(x, y)
    