from matplotlib.axes import Axes

from _domain.plotting import plot_ScanData
from apps.scan_averaging.domain.models import AveragedScansData
from base_core.plotting.enums import PlotColor



def plot_averaged_scan(ax: Axes, data: AveragedScansData, color: PlotColor = PlotColor.RED, label: str = "") -> None:
    n_files = len(data.file_names)
    label = f"{n_files} avg files" + label
    plot_ScanData(ax, data, label,color)