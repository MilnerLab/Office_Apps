from xml.dom import ValidationErr
from matplotlib.axes import Axes
from base_core.lab_specifics.base_models import C2TScanData
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
import numpy as np

from _domain.plotting import plot_ScanData


def plot_single_scan(ax: Axes, data: C2TScanData, show_ions: bool = False, data_color: PlotColor = PlotColor.RED, ecolor: PlotColor = PlotColor.BLACK, ion_color: PlotColor = PlotColor.GRAY,marker='o') -> None:
    if data.file_path is not None:
        label = f"{data.file_path.stem}"
    else:
        raise ValueError('Should have a file path.')
    
    plot_ScanData(ax, data, label, show_ions, data_color, ecolor, marker)
    
    
    
    
    