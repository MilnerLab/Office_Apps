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
from base_core.lab_specifics.averaging.models import AveragedScansData

from base_core.lab_specifics.base_models import C2TScanData, IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time


STFTWINDOWSIZE = Time(180,Prefix.PICO)  
EARLIEST_DELAY_PS = -300
LATEST_DELAY_PS = -EARLIEST_DELAY_PS
POSZEROSHIFT = -3 #millimetres 
USEFONTSIZE = 10


#FUNCTION TO GENERATE THE PLOTTABLE DATA
def calculating(folders: list[Path], configs: list[IonDataAnalysisConfig]) -> tuple[AveragedScansData, AggregateSpectrogram]:
    
    
    scans_paths = DatFinder(folders).find_datafiles() #Change this if you want a specific path rather than the Droplets folder
    raw_datas = load_ion_data(scans_paths)
    calculated_scans = run_pipeline(raw_datas, configs)
    #averagedScanData = average_scans(calculated_scans)
    averagedScanData = calculated_scans[0]
    config = StftAnalysisConfig(calculated_scans)
    config.stft_window_size = STFTWINDOWSIZE 
    resampled_scans = resample_scans(calculated_scans, config.axis)
    spectrogram = StftAnalysis(resampled_scans, config).calculate_averaged_spectrogram()
    
    return (averagedScanData, spectrogram)

def calculating_xcorr(filepath: Path, zero_delay_position: float,prefactor) -> tuple[C2TScanData, AggregateSpectrogram]:
    xcdata = load_xcorr_means(filepath,Length(zero_delay_position,Prefix.MILLI),prefactor)
    config = StftAnalysisConfig([xcdata])
    config.stft_window_size = STFTWINDOWSIZE 
    
    resampled_scans = resample_scans([xcdata], config.axis)
    SpectrogramResult = StftAnalysis(resampled_scans, config)
    
    return (xcdata, SpectrogramResult.calculate_averaged_spectrogram())


#Path to save figure in
fig_filedir = r"C:\Users\camp06\OneDrive - UBC\Documents\Presentations" 
fig_filename = fig_filedir + r"\\vmi_xcorr_comp.png" #Name the file to save here

#Plot title on top
PlotTitle = r"Cross-correlation of CFG with a faster central frequency" 

#XCorr
# GA=0mm DA = 16.6mm
file_xcorr_1 = Path(r"Z:\Droplets\20260207\XCORR\CFG_16p6mm_0mm\202602071250_.csv")
zero_delay_position_1 = 170  #mm


#Calculations here
#XCorr plottable data
xcdata_1, plottable_spectrogram_1 = calculating_xcorr(file_xcorr_1,zero_delay_position_1,prefactor=1000)

#OCS Jet Experiment
# GA=0, DA = 16.6mm
configs_2: list[IonDataAnalysisConfig] = []
folders_2: list[Path] = []

folders_2.append(Path(r"Z:\Droplets\20260207\Scan4"))
configs_2.append(IonDataAnalysisConfig(
    delay_center= Length(92.654+POSZEROSHIFT, Prefix.MILLI),
    center=Point(203, 202),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](30, 90),
    transform_parameter= 0.73))

plottable_scan_2, plottable_spectrogram_2 = calculating(folders_2, configs_2)

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
    "legend.fontsize": 7,
    "legend.handlelength": 1.375,
    "legend.labelspacing": 0.4,
    "legend.columnspacing": 1,
    "legend.facecolor": "white",
    "legend.edgecolor": "white",
    "legend.framealpha": 0.7,
    "legend.title_fontsize": 8,
    "legend.loc": "lower center", #location is loc
 
    # --- Figure size ---
    #"figure.figsize": (3.375, 3.6), #1- column fig
    #"figure.figsize": (6.75, 3.6), #approx. 2- column fig
    #"figure.figsize": (6.299,3.543), #16:9 beamer frame
    "figure.figsize": (5.67,3.24), #90% of beamer frame width
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
#------------------------------------------------------------------------------------

#Main figure
fig, (axs) = plt.subplots(
            nrows=2,
            ncols=2,
            gridspec_kw={'left':0.1,'hspace':0.2,'wspace':0.3}
        )

textfontsize = 6

#Plot Xcorr in top row
#a = axs[0,0]
plot_ScanData(axs[0,0],xcdata_1,label=r"X-corr",color = PlotColor.GRAY)
axs[0,0].set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
axs[0,0].set_ylabel('Intensity (mV)')
axs[0,0].xaxis.label.set_visible(False)
#axs[0,0].legend(loc='upper left')

plot_Spectrogram(axs[0,1], plottable_spectrogram_1,colormap='viridis',shading=None)
axs[0,1].set_ylim([0,120])
axs[0,1].set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
axs[0,1].xaxis.label.set_visible(False)
axs[0,1].yaxis.label.set_visible(False)


#Plot OCS VMI experiment in second row
a = axs[1,0]
plot_ScanData(a, plottable_scan_2,label=r"VMI data" + "\n" + r"OCS jet")
a.set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
a = axs[1,1]
plot_Spectrogram(a, plottable_spectrogram_2,colormap='viridis',shading=None)
a.set_ylim([0,120])
a.yaxis.label.set_visible(False)

xlim = a.get_xlim()
axs[0,1].set_xlim(xlim)

fig.text(0.52,0.53,r"Signal Oscillation Frequency (GHz)",rotation=90,
    va='center',
    ha='right',
    fontsize=10)
 
fig.savefig(fig_filename,format='png',dpi=300)
print('Code end!')
plt.show()