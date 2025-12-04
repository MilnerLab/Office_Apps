from matplotlib.axes import Axes

from _domain.plotting import plot_ScanData
from apps.scan_averaging.domain.models import AveragedScansData



def plot_averaged_scan(ax: Axes, data: AveragedScansData) -> None:
    n_files = len(data.file_names)
    label = f"{n_files} avg files"
    plot_ScanData(ax, data, label)