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

DROPLETRADIUSMIN = 60

STFTWINDOWSIZE = Time(180,Prefix.PICO)  
EARLIEST_DELAY_PS = -350
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
fig_filedir = r"Z:\Droplets\plots" 
fig_filename = fig_filedir + r"\CS2_jet_vs_droplets_TEMP.png" #Name the file to save here

#Path to save processed data in
savedata_filedir = r"Z:\Droplets\exportdata" 
savedata_filename_1 = savedata_filedir + r"\CS2_usCFG_gas.csv" #Name the file to save here
savedata_filename_2 = savedata_filedir + r"\CS2_usCFG_droplets.csv" #Name the file to save here


#Plot on top


PlotTitle = r"CS$_2$ - STFT with 180 ps blackman window. Same centrifuge for each scan." "\n" "202512 10th + 12th + 13th"

PlotTitle = r"CS$_2$ in droplets for both - very different centrifuges"


#JET EXPERIMENT
#GA=26mm, DA=15.9mm
configs_1: list[IonDataAnalysisConfig] = []
folders_1: list[Path] = []

folders_1.append(Path(r"20251210\JSS3"))  #20251210 JSS3 is dense throughout the centrifuge
configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(195, 197),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](20, 120),
    transform_parameter= 0.79))

folders_1.append(Path(r"20251210\JSS4"))  #20251210 JSS4 is dense before the centrifuge
configs_1.append(configs_1[0]) #same config


#DROPLETS EXPERIMENT
#Same GA=26mm, DA = 15.9mm.
#20251212 Scan 4
#20251213 scan 1, 2, and 3. BUT! The last 2 scans in scan3 are bad
configs_2: list[IonDataAnalysisConfig] = []
folders_2: list[Path] = []

folders_2.append(Path(r"20251212\Scan4")) #Combination of 20251212 and 20251213. 
configs_2.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.79))

folders_2.append(Path(r"20251213\Scan1")) #Combination of 20251212 and 20251213. 
configs_2.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.79))

folders_2.append(Path(r"20251213\Scan2")) #Combination of 20251212 and 20251213. 
configs_2.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.79))


folders_2.append(Path(r"20251213\Scan3")) #Combination of 20251212 and 20251213. 
configs_2.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.79))


#FORCE DROPLETS COMPARISON
folders_1 = folders_2
configs_1 = configs_2
#APRIL CENTRIFUGE REVERSED
# GA=0, DA = 16.43mm (slower acceleration than GA=26mm data)
configs_2: list[IonDataAnalysisConfig] = []
folders_2: list[Path] = []

folders_2.append(Path(r"20260427\Scan3")) 
configs_2.append(IonDataAnalysisConfig(
    delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
    center=Point(205, 194),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.78))

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
plottable_scan_1, plottable_spectrogram_1 = calculating(folders_1, configs_1)
plottable_scan_2, plottable_spectrogram_2 = calculating(folders_2, configs_2)


#Main figure
mainfig, (axs) = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(10, 8),
            sharex=True,             
        )

#Plot first experiment in top row
a = axs[0,0]
plot_averaged_scan(a, plottable_scan_1, PlotColor.BLUE,ecolor=PlotColor.RED,marker='d', label = "120 PSI Jet")
a.grid(color='grey',linewidth=0.3)
a.set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
a.legend(loc="lower center")
a = axs[0,1]
plot_Spectrogram(a, plottable_spectrogram_1,shading="auto")
a.set_ylim([0,120])
#plot_nyquist_frequency(a, plottable_scan_1)

#Plot second experiment in bottom row
a = axs[1,0]
plot_averaged_scan(a, plottable_scan_2, PlotColor.BLUE,ecolor=PlotColor.RED,marker='d',label="30 bar / 16 K Droplets")
a.grid(color='grey',linewidth=0.3)
a.legend()
a = axs[1,1]
plot_Spectrogram(a, plottable_spectrogram_2,shading="auto")
a.set_ylim([0,120])
plot_nyquist_frequency(a, plottable_scan_2)
mainfig.suptitle(PlotTitle,fontsize=USEFONTSIZE,color='black')

#Save scans
plottable_scan_1.to_csv(savedata_filename_1)
plottable_scan_2.to_csv(savedata_filename_2)

mainfig.savefig(fig_filename,format='png')
plt.show()
print('Done!')