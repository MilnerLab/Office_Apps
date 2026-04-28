from matplotlib.axes import Axes
from base_core.fitting.functions import fit_gaussian
from base_core.fitting.models import GaussianFitResult
from base_core.lab_specifics.base_models import ScanDataBase
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
import numpy as np


def plot_ScanData(ax: Axes, data: ScanDataBase, label:str, color: PlotColor = None, ecolor: PlotColor = None,marker = 'o') -> None:
    x = [time.value(Prefix.PICO) for time in data.delays]
    y = np.array([c.value for c in data.measured_values])
    error = np.array([c.error for c in data.measured_values])

  
    ax.errorbar(
        x,
        y,
        yerr=error,
        ecolor=ecolor,
        color=color,
        marker = marker,
        markersize = 1.0,
        label = label,
    )
    ax.legend(loc='upper left')
    ax.set_xlabel("Probe Delay (ps)")
    ax.set_ylabel(r"$\langle \cos^2 \theta_\mathrm{2D} \rangle$")
    
def plot_GaussianFit(ax: Axes, data: ScanDataBase) -> None:
    x1 = [t.value(Prefix.PICO) for t in data.delays]
    y1 = [c2t.value for c2t in data.measured_values]
    gauss = fit_gaussian(x1, y1)
    
    resampling_const = len(data.delays) * 100
    
    y = [y for y in gauss.get_curve(np.linspace(x1[0], x1[-1], resampling_const))]
    x = np.linspace(x1[0], x1[-1], resampling_const)
    
    ax.plot(x, y)
    ax.text(0.95,0.95, f"Center = {gauss.center:.2f} ± {gauss.center_err:.2f} ps", transform=ax.transAxes, ha='right', va='top', fontsize=10)
    #print("center = ",gauss.center, " error = ", gauss.center_err)
   