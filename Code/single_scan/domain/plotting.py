from matplotlib.axes import Axes
import numpy as np

from Lab_apps._base.models import PlotColor, Prefix
from Lab_apps._domain.models import LoadableScanData
from Lab_apps._domain.plotting import plot_ScanData
from Lab_apps.scan_averaging.domain.models import AveragedScansData


def plot_single_scan(ax: Axes, data: LoadableScanData, show_ions: bool = False, data_color: PlotColor = PlotColor.RED, ion_color: PlotColor = PlotColor.GRAY) -> None:
    label = f"{data.file_path.stem}"
    plot_ScanData(ax, data, label, data_color)

    if data.ions_per_frame is None:
        return
    
    delay = [t.value(Prefix.PICO) for t in data.delay]
    ions = np.asarray(data.ions_per_frame)

    # zweite y-Achse rechts
    ax_ions = ax.twinx()
    ax_ions.plot(
        delay,
        ions,
        linestyle="--",
        linewidth=1.0,
        color=ion_color,
    )
    ax_ions.set_ylabel("Ions per frame")
    ax_ions.tick_params(axis="y", labelcolor=ion_color)