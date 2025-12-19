from matplotlib.axes import Axes
from base_core.math.models import Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
import numpy as np

from apps.stft_analysis.domain.models import SpectrogramBase
from _domain.models import ScanDataBase
from base_core.quantities.models import Frequency


def plot_Spectrogram(ax: Axes, data: SpectrogramBase, v_range: Range[float] = Range(0, 1)) -> None:
    
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
