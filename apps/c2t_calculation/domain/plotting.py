from matplotlib.axes import Axes
import numpy as np

from Lab_apps._domain.models import IonData, LoadableScanData
from Lab_apps._domain.plotting import plot_ScanData


def plot_calculated_scan(ax: Axes, data: LoadableScanData) -> None:
    label = f"{data.file_path.stem}"
    plot_ScanData(ax, data, label)

def plot_ions_square(ax: Axes, ion_data: IonData, *, color: str = "red", label: str | None = None) -> None:
    xs = [point.x for point in ion_data.points]
    ys = [point.y for point in ion_data.points]

    ax.scatter(xs, ys, color=color, marker=".", s=5, label=label)

    ax.set_aspect("equal", adjustable="box")

    if xs and ys:
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        lim_min = min(xmin, ymin)
        lim_max = max(xmax, ymax)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)