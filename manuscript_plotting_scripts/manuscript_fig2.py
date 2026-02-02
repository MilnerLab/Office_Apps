from ctypes.wintypes import PLONG
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_time_scans
from _domain.models import ScanDataBase
#from _domain.plotting import plot_GaussianFit
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.models import AveragedScansData
from apps.scan_averaging.domain.plotting import plot_averaged_scan

from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import calculate_averaged_spectrogram

from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Time


mpl.rcParams.update({
#copied from physrev.mplstyle file
    #"axes.prop_cycle": "(cycler('color', ['5d81b4', 'e09b24', '8eb031', 'eb6235', '8678b2', 'c46e1a', '5c9dc7', 'ffbf00', 'a5609c']) + cycler('ls', ['-', '--', '-.', (0, (1,0.85)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), ':', (4, (10, 3))]))",  
  
    # --- Axes ---
    "axes.titlesize": "medium",
    "axes.labelsize": 9,
    "axes.formatter.use_mathtext": True,
    "axes.linewidth": 0.5,
    #"axes.grid": True,
    #"axes.grid.axis": "both",  # which axis the grid should apply to
    #"axes.grid.which": "major",
    #"axes.axisbelow" : True,
    #"grid.alpha": 0.25,

    # --- Grid lines ---
    #"grid.linewidth": 0.5,
    #"grid.linestyle": "dashed",
    #"grid.color": "xkcd:light gray",

    # --- Lines ---
    "lines.linewidth": 0.5,
    "lines.marker": "o",
    "lines.markersize": 1.5,
    "hatch.linewidth": 0.25,
    "patch.antialiased": True,
    
    #---Errorbars---
    "errorbar.capsize": 0,

    # --- Ticks (X) ---
    #"xtick.top": True,
    "xtick.bottom": True,
    "xtick.major.size": 3.0,
    "xtick.minor.size": 1.5,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "xtick.direction": "out",
    "xtick.minor.visible": False,
    #"xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.bottom": True,
    "xtick.major.pad": 5.0,
    "xtick.minor.pad": 5.0,
    "xtick.labelsize": 9,

    # --- Ticks (Y) ---
    "ytick.left": True,
    #"ytick.right": True,
    "ytick.major.size": 3.0,
    #"ytick.minor.size": 1.5,
    "ytick.major.width": 0.5,
    #"ytick.minor.width": 0.5,
    "ytick.direction": "out",
    #"ytick.minor.visible": True,
    "ytick.major.left": True,
    #"ytick.major.right": True,
    #"ytick.minor.left": True,
    "ytick.major.pad": 5.0,
    #"ytick.minor.pad": 5.0,
    "ytick.labelsize": 9,
    
    
    # --- Legend ---
    "legend.frameon": True,
    "legend.fontsize": 8,
    "legend.handlelength": 1.375,
    "legend.labelspacing": 0.4,
    "legend.columnspacing": 1,
    "legend.facecolor": "white",
    "legend.edgecolor": "white",
    "legend.framealpha": 1,
    "legend.title_fontsize": 8,
 
    # --- Figure size ---
    "figure.figsize": (3.375, 4), #1- column fig
    #"figure.figsize": (6.75, 6.75), #approx. 2- column fig
    "figure.subplot.left": 0.125,
    "figure.subplot.bottom": 0.175,
    "figure.subplot.top": 0.95,
    "figure.subplot.right": 0.95,
    "figure.autolayout": True,

    # --- Fonts (computer modern) ---
    "text.usetex": True,      
    #"mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"]

})

#For Figure 2
fig_filedir = r"C:/Users/camp06/Documents/droplets_manuscript/"
fig_filename = fig_filedir + r"figure2.pdf"
folder_path = Path(r"C:\Users\camp06\Documents\droplets_manuscript\202512_Droplets(figure2)\202512_Droplets")
file_paths = DatFinder(folder_path).find_scanfiles()
scan_data = load_time_scans(file_paths)

averagedScanData = average_scans(scan_data)

config = StftAnalysisConfig(scan_data)

resampled_scans = resample_scans(scan_data, config.axis)
spectrogram = calculate_averaged_spectrogram(resampled_scans, config)

fig, (ax2a,ax2b) = plt.subplots(2,1)
plot_averaged_scan(ax2a, averagedScanData, PlotColor.BLUE)
plot_Spectrogram(ax2b, spectrogram)

ax2a.xaxis.label.set_visible(False)
ax2a.set_xticklabels("")
ax2a.set_xticks([])
ax2a.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02)) 

ylimits = {
"vmin": 0,
"vmax": 100        
}
xlimits = {
    "vmin": -200,
    "vmax": 200
}
#ax2b.xaxis.set_major_locator(mpl.ticker.MultipleLocator(100))
#ax2b.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=5,presets=xlimits))
#ax2b.yaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
#ax2b.yaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=3,presets=limits))


ax2b.grid(False) 
#ax2b.pcolormesh.cmap('magma')

ax2a.text(
    0.02, 0.95, r'\textbf{(a)}',
    transform=ax2a.transAxes,
    va='top'
)
ax2b.text(
    0.02, 0.95, r'\textbf{(b)}',
    transform=ax2b.transAxes,
    va='top',
    color='w'
)

fig.savefig(fig_filename,format='pdf')
fig.tight_layout()
plt.show()
