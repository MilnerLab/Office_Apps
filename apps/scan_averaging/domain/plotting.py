from matplotlib.axes import Axes

from _domain.plotting import plot_ScanData
from base_core.lab_specifics.averaging.models import AveragedScansData
from base_core.plotting.enums import PlotColor



def plot_averaged_scan(ax: Axes, data: AveragedScansData, color: PlotColor = None, ecolor: PlotColor=None, label: str = None, marker = 'o', elinewidth:float = 0.5) -> None:
    n_files = len(data.run_ids)
    if marker == 'o' and label is not None:
        label = f"{n_files} avg files" + label
        #sorry but this was the easiest way to remove the N avg files for my plot without breaking plotbot :) 
    plot_ScanData(ax, data, label, number_of_scans=data.run_ids.__len__(), color=color, ecolor=ecolor, marker=marker, elinewidth=elinewidth)