from matplotlib.axes import Axes

from _domain.plotting import plot_ScanData
from apps.scan_averaging.domain.models import AveragedScansData
from base_core.plotting.enums import PlotColor



def plot_averaged_scan(ax: Axes, data: AveragedScansData, color: PlotColor = None, ecolor: PlotColor=None, label: str = "",marker = 'o') -> None:
    n_files = len(data.file_names)
    if marker == 'o':
        label = f"{n_files} avg files" + label
        #sorry but this was the easiest way to remove the N avg files for my plot without breaking plotbot :) 
    plot_ScanData(ax, data, label,color,ecolor,marker)