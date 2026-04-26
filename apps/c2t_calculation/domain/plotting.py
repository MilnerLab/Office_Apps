from matplotlib.axes import Axes
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from _domain.plotting import plot_ScanData
from base_core.lab_specifics.base_models import C2TScanData, Points



def plot_calculated_scan(ax: Axes, data: C2TScanData,label:str = '',*,number_of_scans: int = 1) -> None:
    #if data.file_path is not None:
        #label = f"{data.file_path.stem}"
    if label =='':
        label = f"{number_of_scans} scans" 
    plot_ScanData(ax, data, label)

def plot_ions_square(
    ax: Axes,
    points: Points,
    *,
    color: str = "red",
    label: str | None = None,
) -> tuple[Axes, Axes]:
    xs = points.x
    ys = points.y

    ax.scatter(xs, ys, color=color, marker=".", s=5, label=label, alpha=0.2)
    