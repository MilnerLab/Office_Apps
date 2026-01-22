from matplotlib.axes import Axes
import numpy as np

from _domain.models import LoadableScanData
from _domain.plotting import plot_ScanData
from base_core.plotting.enums import PlotColor

#Temporary plotting function for GUI_based_plotting....

def plot_single_scan(ax: Axes, data: LoadableScanData, show_ions: bool = False, data_color: PlotColor = PlotColor.RED, ion_color: PlotColor = PlotColor.GRAY) -> None:
    label = f"{data.file_path.stem}"
    plot_ScanData(ax, data, label, data_color)

