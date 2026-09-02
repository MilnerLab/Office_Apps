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
        label = "Calculated Scan"
    
    ax_twin = None
    if show_ions and data.ions_per_frame is not None:
        ax_twin = ax.twinx()

    plot_ScanData(
        ax,
        data,
        label,
        ax_twin=ax_twin,
        color=data_color,
        ecolor=ecolor,
        marker=marker,
        ion_color=ion_color,
    )
    
    
    
    
    