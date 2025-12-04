from matplotlib.axes import Axes
import numpy as np

from Lab_apps._base.models import Frequency, PlotColor, Prefix, Range
from Lab_apps._domain.models import ScanDataBase
from Lab_apps.stft_analysis.domain.models import SpectrogramBase


def plot_Spectrogram(ax: Axes, data: SpectrogramBase, v_range: Range[float] = Range(0, 0.2)) -> None:
    
    delay = [d.value(Prefix.PICO) for d in data.delay]
    frequency = [f.value(Prefix.GIGA) for f in data.frequency]
    P = np.array(data.power)       
    
    ax.pcolormesh(delay, frequency, P, shading="gouraud",cmap="viridis", vmin=v_range.min, vmax=v_range.max)
    ax.set_ylim(0, 150)
    ax.set_xlabel("Probe Delay (ps)")
    ax.set_ylabel("Oscillation \n Frequency (GHz)")
    ax.grid(color='grey',linewidth=0.3)
    
def plot_nyquist_frequency(ax: Axes, data: ScanDataBase) -> None:
    
    delay = [d.value(Prefix.PICO) for d in data.delay]
    frequencies = 1/np.diff(data.delay)/2
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
