print('Code start!')
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib as mpl

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data, load_xcorr_means
from _data_io.dat_saver import create_save_path_for_calc_ScanFile
from _domain.plotting import plot_ScanData
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from apps.single_scan.domain.plotting import plot_single_scan
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.models import SpectrogramResult, AggregateSpectrogram
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scan, resample_scans
from apps.stft_analysis.domain.stft_calculation import StftAnalysis

from base_core.lab_specifics.base_models import C2TScanData, IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time


STFTWINDOWSIZE = Time(180,Prefix.PICO)  
EARLIEST_DELAY_PS = -550
LATEST_DELAY_PS = -EARLIEST_DELAY_PS
POSZEROSHIFT = 0 #millimetres :)
USEFONTSIZE = 10


#FUNCTION TO GENERATE THE PLOTTABLE DATA
def calculating(filepath: Path, zero_delay_position: float,prefactor) -> tuple[C2TScanData, AggregateSpectrogram]:
    xcdata = load_xcorr_means(filepath,Length(zero_delay_position,Prefix.MILLI),prefactor)
    config = StftAnalysisConfig([xcdata])
    config.stft_window_size = STFTWINDOWSIZE 
    
    resampled_scans = resample_scans([xcdata], config.axis)
    SpectrogramResult = StftAnalysis(resampled_scans, config)
    
    return (xcdata, SpectrogramResult.calculate_averaged_spectrogram())


#Path to save figure in
fig_filedir = r"C:\Users\camp06\OneDrive - UBC\Documents\droplets_manuscript\xcorr" 
fig_filename = fig_filedir + r"\\slowest_accel.png" #Name the file to save here

#Plot title on top
PlotTitle = r"Cross-correlation of CFG with a faster central frequency" 

#FIRST EXPERIMENT
# GA=0mm DA = 16.6mm
file_xcorr_1 = Path(r"Z:\Droplets\20260207\XCORR\CFG_16p6mm_0mm\202602071250_.csv")
zero_delay_position_1 = 170 + POSZEROSHIFT #mm
#SECOND EXPERIMENT
# GA=26mm DA = 16.3mm
file_xcorr_2 = Path(r"Z:\Droplets\20260207\XCORR\CFG_16p3mm_26mm\20260207905AM_.csv")
zero_delay_position_2 = 170 + POSZEROSHIFT #mm

#GA=26, DA=16.0 mm
#file_xcorr3 = Path(r"C:\Users\camp06\OneDrive - UBC\Documents\droplets_manuscript\202602131102AM_.csv")
#zero_delay_position_3 = 170 + POSZEROSHIFT #mm
#--------------------------------------------------------------------------------------------------
#Update the matplotlib settings
mpl.rcParams.update({
#copied from physrev.mplstyle file
    #"axes.prop_cycle": "(cycler('color', ['5d81b4', 'e09b24', '8eb031', 'eb6235', '8678b2', 'c46e1a', '5c9dc7', 'ffbf00', 'a5609c']) + cycler('ls', ['-', '--', '-.', (0, (1,0.85)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), ':', (4, (10, 3))]))",  
  
    # --- Axes ---
    "axes.titlesize": "medium",
    "axes.labelsize": USEFONTSIZE,
    "axes.formatter.use_mathtext": True,
    "axes.linewidth": 0.5,
    "axes.grid": True,
    "axes.grid.axis": "both",  # which axis the grid should apply to
    "axes.grid.which": "major",
    #"axes.axisbelow" : True,
    "grid.alpha": 1,

    # --- Grid lines ---
    "grid.linewidth": 0.2,
    "grid.linestyle": "solid",
    "grid.color": "grey",

    # --- Lines ---
    "lines.linewidth": 0.3,
    "lines.marker": "d",
    "lines.markersize": 0.2,
    "hatch.linewidth": 0.25,
    "patch.antialiased": True,
    
    #---Errorbars---
    "errorbar.capsize": 1,

    # --- Ticks (X) ---
    #"xtick.top": True,
    "xtick.bottom": True,
    "xtick.major.size": 3.0,
    "xtick.minor.size": 1.5,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "xtick.direction": "in",
    "xtick.minor.visible": False,
    #"xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.bottom": True,
    "xtick.major.pad": 5.0,
    "xtick.minor.pad": 5.0,
    "xtick.labelsize": USEFONTSIZE,

    # --- Ticks (Y) ---
    "ytick.left": True,
    #"ytick.right": True,
    "ytick.major.size": 3.0,
    #"ytick.minor.size": 1.5,
    "ytick.major.width": 0.5,
    #"ytick.minor.width": 0.5,
    "ytick.direction": "in",
    #"ytick.minor.visible": True,
    "ytick.major.left": True,
    #"ytick.major.right": True,
    #"ytick.minor.left": True,
    "ytick.major.pad": 5.0,
    #"ytick.minor.pad": 5.0,
    "ytick.labelsize": USEFONTSIZE,
    
    
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
    "legend.loc": "lower center", #location is loc
 
    # --- Figure size ---
    #"figure.figsize": (3.375, 3.6), #1- column fig
    #"figure.figsize": (6.75, 3.6), #approx. 2- column fig
    #"figure.figsize": (6.299,3.543), #16:9 beamer frame
    "figure.figsize": (5.67,1.772), #90% of beamer frame width
    "figure.subplot.left": 0.125,
    "figure.subplot.bottom": 0.175,
    "figure.subplot.top": 0.9,
    "figure.subplot.right": 0.95,
    "figure.autolayout": True,

    # --- Fonts (computer modern) ---
    "text.usetex": True,       #<------------------------------------------------<-----<_<_<_ LATEX 
    #"mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"],

})



#Calculations here

xcdata_1, plottable_spectrogram_1 = calculating(file_xcorr_1,zero_delay_position_1,prefactor=1000)


#xcdata_2, plottable_spectrogram_2 = calculating(file_xcorr_2,zero_delay_position_2)

#xcdata_3, plottable_spectrogram_3 = calculating(file_xcorr3,zero_delay_position_3)
#--------------------------------------------------------------------------------------------------


#Main figure
# mainfig, (axs) = plt.subplots(
#             nrows=2,
#             ncols=2,
#             figsize=(10, 8),
#             sharex=True, 
#         )

fig, (ax1,ax2) = plt.subplots(1,2,gridspec_kw={'width_ratios': [65, 35]})
textfontsize = 6

#Plot first experiment in top row
#a = axs[0,0]
plot_ScanData(ax1,xcdata_1,color = PlotColor.GRAY)
ax1.set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
ax1.set_ylabel('Intensity (mV)')

plot_Spectrogram(ax2, plottable_spectrogram_1,colormap='viridis',shading=None)
ax2.set_ylim([0,120])
ax2.text(0.05,0.78,r"$f_0 \approx 24 \; \mathrm{GHz}$" "\n" r"$\Delta\beta \approx 0.1 \frac{\mathrm{GHz}}{\mathrm{ps}}$",transform = ax2.transAxes,fontsize=textfontsize,linespacing=1.5,bbox=dict(
        facecolor='white',
        edgecolor='black',
        alpha=0.8,
        pad=0.7,           # padding around text
        boxstyle='round' # 'round', 'square', 'round4', 'rarrow', etc.
))

#a.legend(loc="upper right")
#a = axs[0,1]
#plot_Spectrogram(a, plottable_spectrogram_1,colormap='viridis')
#plot_Spectrogram(a, plottable_spectrogram_3,colormap='magma')
#a.set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
#a.set_ylim([0,120])
#a.set_ylim([0,120])
#plot_nyquist_frequency(a, plottable_nyquist_1)


#Plot second experiment in bottom row
#a = axs[1,0]
# plot_ScanData(a,xcdata_2,color = PlotColor.GRAY,label="GA=0mm, DA=15.28mm)")
# ax.set_ylabel('Photodiode Signal (V)')
# a.legend()
# a.legend(loc="upper right")
# a = axs[1,1]
# plot_Spectrogram(a, plottable_spectrogram_2,colormap='viridis')
# a.set_ylim([0,150])

#a.set_ylim([0,120])
#plot_nyquist_frequency(a, plottable_nyquist_1)
#mainfig.suptitle(PlotTitle,fontsize=USEFONTSIZE,color='BLUE')

#mainfig.savefig(fig_filename,format='png')
fig.tight_layout()
fig.savefig(fig_filename,format='png',dpi=300,bbox_inches='tight')
plt.show()
print('Code end!')