print('Code start!')
from pathlib import Path
from altair import FontWeight
import matplotlib as mpl
from matplotlib import pyplot as plt

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from _data_io.dat_saver import create_save_path_for_calc_ScanFile
from _domain.plotting import plot_GaussianFit
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from apps.single_scan.domain.plotting import plot_single_scan
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.models import AggregateSpectrogram
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from base_core.lab_specifics.averaging.models import AveragedScansData
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time

DROPLETRADIUSMIN = 40
JETMINRADIUS = 50

STFTWINDOWSIZE = Time(200,Prefix.PICO)  
EARLIEST_DELAY_PS = -250
LATEST_DELAY_PS = -EARLIEST_DELAY_PS
POSZEROSHIFT = 0 #millimetres :)

USEFONTSIZE = 16

#FUNCTION TO GENERATE THE PLOTTABLE DATA
def calculating(folders: list[Path], configs: list[IonDataAnalysisConfig]) -> tuple[AveragedScansData, AggregateSpectrogram]:
    
    
    scans_paths = DatFinder(folders).find_datafiles() #Change this if you want a specific path rather than the Droplets folder
    raw_datas = load_ion_data(scans_paths)
    calculated_scans = run_pipeline(raw_datas, configs)
    averagedScanData = average_scans(calculated_scans)
    config = StftAnalysisConfig(calculated_scans, STFTWINDOWSIZE)
    resampled_scans = resample_scans(calculated_scans, config.axis)

    spectrogram = StftAnalysis(resampled_scans, config).calculate_averaged_spectrogram()
    
    return (averagedScanData, spectrogram)
#--------------------------------------------------------------------------------------------------

#Path to save figure in

 #Name the file to save here
fig_root = Path(r"C:\Users\camp06\OneDrive - UBC\Documents\droplets_manuscript\paper_data")
fig_filename1 = fig_root / "dib_cs2jet_slow.png"
fig_filename2 = fig_root / "dib_cs2jet_fast.png"
fig_filename3 = fig_root / "dib_horizontal.png"


#Slow cfg droplets
configs1: list[IonDataAnalysisConfig] = []
folders_1 = [Path(r"Z:\Droplets\20260402\Scan4"),Path(r"Z:\Droplets\20260402\Scan5")] 
configs1.append(IonDataAnalysisConfig(
    delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
    center=Point(186, 205),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 90),
    transform_parameter= 0.81))

configs2 = [configs1[0],configs1[0]]
#Fast cfg
folders_2 = [Path(r"Z:\Droplets\20260401\Scan3")]

#Single arm horizontal
folders_3 = [Path(r"Z:\Droplets\20260331\Scan1")]

#Jet CS2 slow cfg
folders_4 = [Path(r"Z:\Droplets\20260402\Scan1")]
configs3 = [IonDataAnalysisConfig(
    delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
    center=Point(210, 194),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](40, 120),
    transform_parameter= 0.77)] #same config as jet cs2 fast cfg

#Jet CS2 fast cfg
folders_5 = [Path(r"Z:\Droplets\20260328\Scan1")]

plot_title1 = r"Same slow centrifuge in each scan. STFT Blackman window size: " + f"{STFTWINDOWSIZE.value(Prefix.PICO):.0f}" + " ps" #GA = 0mm
plot_title3 = r"Alignment of DIB in droplets with horizontally polarized pulse"
plot_title2 = r"Same fast centrifuge in each scan. STFT Blackman window size: " + f"{STFTWINDOWSIZE.value(Prefix.PICO):.0f}" + " ps" 

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#Update the matplotlib settings
mpl.rcParams.update({
  
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
    "grid.linewidth": 0.3,
    "grid.linestyle": "solid",
    "grid.color": "grey",

    # --- Lines ---
    "lines.linewidth": 0.5,
    "lines.marker": "d",
    "lines.markersize": 0.5,
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
    "legend.fontsize": 12,
    "legend.handlelength": 1.375,
    "legend.labelspacing": 0.4,
    "legend.columnspacing": 1,
    "legend.facecolor": "white",
    "legend.edgecolor": "white",
    "legend.framealpha": 1,
    "legend.title_fontsize": 12,
    "legend.loc": "lower center", #location is loc
 
    # --- Figure size ---
    "figure.figsize": (3.375, 3.6), #1- column fig
    #"figure.figsize": (6.75, 6.75), #approx. 2- column fig
    "figure.subplot.left": 0.125,
    "figure.subplot.bottom": 0.175,
    "figure.subplot.top": 0.9,
    "figure.subplot.right": 0.95,
    "figure.autolayout": True,

    # --- Fonts (computer modern) ---
    "text.usetex": False,       #<------------------------------------------------<-----<_<_<_ LATEX 
    #"mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"],

})



#Pipeline 
plottable_scan_1, plottable_spectrogram_1 = calculating(folders_1, configs2)
plottable_scan_2, plottable_spectrogram_2 = calculating(folders_2, configs1)
plottable_scan_3, plottable_spectrogram_3 = calculating(folders_4, configs3)
plottable_scan_4, plottable_spectrogram_4 = calculating(folders_5, configs3) #same config as jet cs2 slow cfg

#C2t for single arm horizontal
scans_paths = DatFinder(folders_3).find_datafiles() 
raw_datas = load_ion_data(scans_paths)
single_arm = run_pipeline(raw_datas, configs1)
single_arm_avg = average_scans(single_arm)


#Main figure
fig1, (axs) = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(12, 8),
            sharex=True,             
        )

#Plot first experiment in top row
a = axs[0,0]
plot_averaged_scan(a, plottable_scan_1, PlotColor.BLUE,ecolor=PlotColor.RED,marker='d', label = "DIB in droplets 20260402 Scan4 and Scan5")
a.grid(color='grey',linewidth=0.3)
a.set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
a.legend(loc="lower center")
a = axs[0,1]
plot_Spectrogram(a, plottable_spectrogram_1,shading="auto")
a.set_ylim([0,120])
plot_nyquist_frequency(a, plottable_scan_1)

#Plot second experiment in bottom row
a = axs[1,0]
plot_averaged_scan(a, plottable_scan_3, PlotColor.BLUE,ecolor=PlotColor.RED,marker='d',label="CS2 jet 20260402 Scan1")
a.grid(color='grey',linewidth=0.3)
a.set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
a.legend()
a = axs[1,1]
plot_Spectrogram(a, plottable_spectrogram_3,shading="auto")
a.set_ylim([0,120])
plot_nyquist_frequency(a, plottable_scan_3)
fig1.suptitle(plot_title1,fontsize=USEFONTSIZE,color='black')


fig1.savefig(fig_filename1,format='png')

fig2, (axs) = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(12,8),
    sharex=True,
)

#Plot first experiment in top row
a = axs[0,0]
plot_averaged_scan(a, plottable_scan_2, PlotColor.BLUE,ecolor=PlotColor.RED,marker='d', label = "DIB in droplets 20260401 Scan3")
a.grid(color='grey',linewidth=0.3)
a.set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
a.legend(loc="lower center")
a = axs[0,1]
plot_Spectrogram(a, plottable_spectrogram_2,shading="auto")
a.set_ylim([0,120])
plot_nyquist_frequency(a, plottable_scan_2)

#Plot second experiment in bottom row
a = axs[1,0]
plot_averaged_scan(a, plottable_scan_4, PlotColor.BLUE,ecolor=PlotColor.RED,marker='d',label="CS2 jet 20260328 Scan1")
a.grid(color='grey',linewidth=0.3)
a.set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
a.legend()
a = axs[1,1]
plot_Spectrogram(a, plottable_spectrogram_4,shading="auto")
a.set_ylim([0,120])
plot_nyquist_frequency(a, plottable_scan_4)
fig2.suptitle(plot_title2,fontsize=USEFONTSIZE,color='black')


fig2.savefig(fig_filename2,format='png')

fig3, ax = plt.subplots(figsize=(12,5))
ax.grid(color='grey',linewidth=0.3)
plot_averaged_scan(ax, single_arm_avg, PlotColor.BLUE,ecolor=PlotColor.RED,marker='d',label="20260331 Scan1")
fig3.suptitle(plot_title3,fontsize=USEFONTSIZE,color='black')
fig3.savefig(fig_filename3,format='png')

plottable_scan_3.to_csv(fig_root / "cs2jet_slow.csv")
plottable_scan_4.to_csv(fig_root / "cs2jet_fast.csv")

plt.show()