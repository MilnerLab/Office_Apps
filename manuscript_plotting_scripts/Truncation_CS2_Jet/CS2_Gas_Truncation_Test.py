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

DROPLETRADIUSMIN = 65

STFTWINDOWSIZE = Time(180,Prefix.PICO)  
EARLIEST_DELAY_PS = -600
LATEST_DELAY_PS = 1550
POSZEROSHIFT = 0 #millimetres :)

MINRADIUS = 30
MAXRADIUS = 60

MAJORTITLEFONTSIZE = 12
YLABELX = -0.135

#FUNCTION TO GENERATE THE PLOTTABLE DATA
def calculating(folders: list[Path], configs: list[IonDataAnalysisConfig]) -> tuple[AveragedScansData, AggregateSpectrogram]:
    
    scans_paths = DatFinder(folders).find_datafiles() #Change this if you want a specific path rather than the Droplets folder
    raw_datas = load_ion_data(scans_paths)
    calculated_scans = run_pipeline(raw_datas, configs)
    averagedScanData = average_scans(calculated_scans)
    
    return (averagedScanData)
#--------------------------------------------------------------------------------------------------

#Path to save figure in
fig_filedir = r"Z:\Droplets\plots" 
fig_filename = fig_filedir + r"\CS2_JET_TRUNCATION_TEMP.png" #Name the file to save here



PlotTitle = r"CS$_2$ in jet" 


#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []

folders.append(Path(r"20260904\Scan3_CFG")) #20260904\Scan3_CFG is full centrifuge
configs.append(IonDataAnalysisConfig(
    delay_center= Length(64.5-POSZEROSHIFT, Prefix.MILLI),
    center=Point(162, 220),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS, MAXRADIUS),
    transform_parameter=0.85))

folders_full = folders
configs_full = configs

#--------------------------------------------------------------------------------------------------------------

configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []

folders.append(Path(r"20260907\Scan2_CFG"))  #15mm truncation with 20260907\Scan2_CFG
configs.append(IonDataAnalysisConfig(
    delay_center= Length(64.5-POSZEROSHIFT, Prefix.MILLI),
    center=Point(162, 220),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS, MAXRADIUS),
    transform_parameter=0.85))

folders.append(Path(r"20260907\Scan3_CFG"))  #15mm truncation with 20260907\Scan3_CFG
configs.append(IonDataAnalysisConfig(
    delay_center= Length(64.5-POSZEROSHIFT, Prefix.MILLI),
    center=Point(162, 220),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS, MAXRADIUS),
    transform_parameter=0.85))

folders_15mm = folders
configs_15mm = configs

#--------------------------------------------------------------------------------------------------------------

configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []

folders.append(Path(r"20260904\Scan4_CFG"))  #14.5mm truncation with 20260904\Scan4_CFG
configs.append(IonDataAnalysisConfig(
    delay_center= Length(64.5-POSZEROSHIFT, Prefix.MILLI),
    center=Point(162, 220),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS, MAXRADIUS),
    transform_parameter=0.85))

folders_14p5mm = folders
configs_14p5mm = configs

#--------------------------------------------------------------------------------------------------------------

configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []

folders.append(Path(r"20260907\Scan1_CFG"))  #13.5mm truncation with 20260904\Scan1_CFG
configs.append(IonDataAnalysisConfig(
    delay_center= Length(64.5-POSZEROSHIFT, Prefix.MILLI),
    center=Point(162, 220),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS, MAXRADIUS),
    transform_parameter=0.85))

folders_13p5mm = folders
configs_13p5mm = configs





#--------------------------------------------------------------------------------------------------
#Update the matplotlib settings
plt.style.use(r"stylefiles\compare_c2t_spectrogram.mplstyle")

#Pipeline 
full  = calculating(folders_full, configs_full)
trunc_15mm = calculating(folders_15mm, configs_15mm)
trunc_14p5mm = calculating(folders_14p5mm, configs_14p5mm)
trunc_13p5mm = calculating(folders_13p5mm, configs_13p5mm)


#Main figure
mainfig, (axs) = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(6.75, 3),
            sharex=True,             
            gridspec_kw={'hspace': 0,'wspace': 0.3}
        )

#Plot first experiment in top row
a = axs
plot_averaged_scan(a, full, PlotColor.BLACK,ecolor=PlotColor.BLACK,marker='d', label = "Full Centrifuge",elinewidth=1)
plot_averaged_scan(a, trunc_15mm, PlotColor.GREEN,ecolor=PlotColor.GREEN,marker='d', label = "15 mm Truncation",elinewidth=1)
plot_averaged_scan(a, trunc_14p5mm, PlotColor.RED,ecolor=PlotColor.RED,marker='d', label = "14.5 mm Truncation",elinewidth=1)
plot_averaged_scan(a, trunc_13p5mm, PlotColor.BLUE,ecolor=PlotColor.BLUE,marker='d', label = "13.5 mm Truncation",elinewidth=1)
a.grid()
a.set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
a.set_xlabel("Probe Delay (ps)")
a.legend(loc='upper left')


#mainfig.suptitle(PlotTitle,fontsize=MAJORTITLEFONTSIZE,color='black')

mainfig.savefig(fig_filename,format='png',dpi=300)
plt.show()
print('Done!')