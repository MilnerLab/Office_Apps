from matplotlib.axes import Axes
from base_core.fitting.functions import fit_gaussian
from base_core.fitting.models import GaussianFitResult
from base_core.lab_specifics.averaging.models import AveragedScansData
from base_core.lab_specifics.base_models import C2TScanData, ScanDataBase
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
import numpy as np


def plot_ScanData(ax: Axes, data: ScanDataBase, label:str = None,*, number_of_scans: int,ax_twin: Axes = None, color: PlotColor = PlotColor.BLUE, ecolor: PlotColor = PlotColor.RED,marker = 'o', ion_color: PlotColor = PlotColor.GRAY) -> None:
    x = [time.value(Prefix.PICO) for time in data.delays]
    y = np.array([c.value for c in data.measured_values])
    error = np.array([c.error for c in data.measured_values])
    
    if label == None:
        label =  f"{number_of_scans} scans" 
  
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
    
    if ax_twin is not None and (isinstance(data,C2TScanData) or isinstance(data,AveragedScansData)):
        ions = np.asarray(data.ions_per_frame)
        
        ax_twin.plot(
            x,
            ions,
            linestyle="--",
            linewidth=1.0,
            color=ion_color,
            marker = marker,
        )
        ax_twin.set_ylabel("Ions per frame",rotation=270,labelpad=8)
        ax_twin.tick_params(axis="y", labelcolor=ion_color)
        ax_twin.grid(False)
    
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
   