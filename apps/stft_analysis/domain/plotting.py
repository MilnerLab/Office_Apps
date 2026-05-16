from matplotlib.axes import Axes
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from base_core.lab_specifics.averaging.models import AveragedScansData
from base_core.lab_specifics.base_models import ScanDataBase
from base_core.math.models import Range
from base_core.plotting.enums import PlotColor, PlotColorMap
from base_core.quantities.enums import Prefix
import numpy as np

from apps.stft_analysis.domain.models import SpectrogramBase
from base_core.quantities.models import Frequency

def plot_scan_and_spectrogram(
    ax: Axes,
    scan_data: AveragedScansData,
    spectrogram_data: SpectrogramBase,
    scan_color: PlotColor = PlotColor.BLUE,
    scan_ecolor: PlotColor = PlotColor.RED,
    scan_marker: str = "d",
    scan_label: str = None,
    v_range: Range[float] = Range(0, 1),
    colormap: PlotColorMap = PlotColorMap.MAGMA,
    shading: str = "auto",
):
    # Left axis: <cos^2>
    ax_scan = ax

    # Right axis: frequency / spectrogram
    ax_freq = ax_scan.twinx()
    
    # --- Spectrogram on right y-axis ---
    delay = [d.value(Prefix.PICO) for d in spectrogram_data.delay]
    frequency = [f.value(Prefix.GIGA) for f in spectrogram_data.frequency]
    P = np.array(spectrogram_data.power)

    mesh = ax_freq.pcolormesh(
        delay,
        frequency,
        P,
        shading=shading,
        cmap=colormap,
        vmin=v_range.min,
        vmax=v_range.max,
        alpha=0.85,
    )

    # --- Put scan axis visually above spectrogram axis ---
    ax_scan.set_zorder(ax_freq.get_zorder() + 1)
    ax_scan.patch.set_visible(False)

    # --- Scan on left y-axis ---
    plot_averaged_scan(
        ax_scan,
        scan_data,
        color=scan_color,
        ecolor=scan_ecolor,
        marker=scan_marker,
        label=scan_label,
        elinewidth=0
    )

    return ax_freq, mesh

def plot_Spectrogram(ax: Axes, data: SpectrogramBase, v_range: Range[float] = Range(0, 1), colormap: PlotColorMap = PlotColorMap.MAGMA,shading:str = 'gouraud') -> None:
    
    delay = [d.value(Prefix.PICO) for d in data.delay]
    frequency = [f.value(Prefix.GIGA) for f in data.frequency]
    P = np.array(data.power)       
    
    ax.pcolormesh(delay, frequency, P, shading=shading,cmap=colormap, vmin=v_range.min, vmax=v_range.max)
    ax.set_ylim(0, 150)
    ax.set_xlabel("Probe Delay (ps)")
    ax.set_ylabel("Oscillation \n Frequency (GHz)")
    ax.grid(color='grey',linewidth=0.3)
    
def plot_nyquist_frequency(ax: Axes, data: ScanDataBase) -> None:
    
    delay = [d.value(Prefix.PICO) for d in data.delays]
    frequencies = 1/np.diff(data.delays)/2
    frequency = [Frequency(f).value(Prefix.GIGA) for f in frequencies]
    ax.plot(
        delay[0:-1],
        frequency,
        linestyle="dotted",
        linewidth=1.5,
        markersize=3,
        color= PlotColor.RED,
        label="Pre-interp. Limit", )
    ax.legend(loc="upper left")        
