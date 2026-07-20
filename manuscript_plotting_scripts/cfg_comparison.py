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

mpl.rcParams.update({
  
    # --- Axes ---
    "axes.titlesize": "medium",
    "axes.labelsize": 9,
    "axes.formatter.use_mathtext": True,
    "axes.linewidth": 0.5,
    "axes.grid": True,
    "axes.grid.axis": "both",  # which axis the grid should apply to
    "axes.grid.which": "major",
    "axes.axisbelow" : True,
    "grid.alpha": 0.25,

    # --- Grid lines ---
    "grid.linewidth": 0.5,
    "grid.linestyle": "dashed",
    "grid.color": "xkcd:light gray",

    # --- Lines ---
    "lines.linewidth": 0.5,
    "lines.marker": "o",
    "lines.markersize": 1.5,
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
    "xtick.labelsize": 9,

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
    #"figure.figsize": (3.375, 3.6), #1- column fig
    "figure.figsize": (6.75, 6.75), #approx. 2- column fig
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


DROPLETRADIUSMIN = 60

STFTWINDOWSIZE = Time(180,Prefix.PICO)  
EARLIEST_DELAY_PS = -550
LATEST_DELAY_PS = -EARLIEST_DELAY_PS
POSZEROSHIFT = 0 #millimetres :)

#CS2 droplets 2025 data
configs_2025: list[IonDataAnalysisConfig] = []
folders_2025: list[Path] = []

folders_2025.append(Path(r"20251212\Scan4")) #Combination of 20251212 and 20251213. 
configs_2025.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.79))

folders_2025.append(Path(r"20251213\Scan1")) #Combination of 20251212 and 20251213. 
configs_2025.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.79))

folders_2025.append(Path(r"20251213\Scan2")) #Combination of 20251212 and 20251213. 
configs_2025.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.79))


folders_2025.append(Path(r"20251213\Scan3")) #Combination of 20251212 and 20251213. 
configs_2025.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.79)) 

#CS2 Droplets March data with "skewed" cfg
configs_2026: list[IonDataAnalysisConfig] = []
folders_2026: list[Path] = [] 


folders_2026.append(Path(r"Z:\Droplets\20260328\Scan3"))
configs_2026.append(IonDataAnalysisConfig(
    delay_center = Length(93.3,Prefix.MILLI),
    center=Point(180,204),
    angle=Angle(12,AngleUnit.DEG),
    analysis_zone=Range[int](40,90),
    transform_parameter=0.77,
))

plottable_scan2025,plottable_spec2025 = calculating(folders=folders_2025,configs=configs_2025)
plottable_scan2026,plottable_spec2026 = calculating(folders=folders_2026,configs=configs_2026)

axes = []
fig,axes = plt.subplots(
    nrows=2,
    ncols=2,
    gridspec_kw={'hspace': 0,'wspace' : 0.275},
)

plot_averaged_scan(axes[0,0],plottable_scan2025,PlotColor.BLUE,ecolor=PlotColor.RED,marker='d')
plot_Spectrogram(axes[1,0],plottable_spec2025,shading='auto')
axes[1,0].set_ylim([0,120])
yticks = axes[1,0].yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
axes[0,0].set_xlim([-200,100])
axes[1,0].sharex(axes[0,0])
axes[0,0].xaxis.label.set_visible(False)
axes[0,0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
axes[0,0].set_title("December 2025 Cfg without skew")


plot_averaged_scan(axes[0,1],plottable_scan2026,PlotColor.BLUE,ecolor=PlotColor.RED,marker='d')
plot_Spectrogram(axes[1,1],plottable_spec2026,shading='auto')
axes[1,1].set_ylim([0,120])
yticks = axes[1,1].yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
axes[0,1].sharex(axes[1,1])
axes[0,1].xaxis.label.set_visible(False)
axes[0,1].tick_params(axis='x',which='both', bottom=False, labelbottom=False)
axes[0,1].set_title("March 2026 Cfg with skew")


savefig_folder = r"C:\Users\camp06\OneDrive - UBC\Documents\droplets_manuscript\c2t_plots\\"
savefig_filename = savefig_folder + r"cfg_skew_vs_noskew.png"
#fig.tight_layout()
#fig.suptitle("Comparison of CS2 droplet scans. The two centrifuges have slightly different starting frequencies.")
fig.savefig(savefig_filename,format='png')
plt.show()

