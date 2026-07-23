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

from apps.stft_analysis.domain.models import AggregateSpectrogram

from base_core.lab_specifics.averaging.models import AveragedScansData
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time
print('Dependencies loaded.')

DROPLETRADIUSMIN = 60
STFTWINDOWSIZE = Time(180,Prefix.PICO)  
EARLIEST_DELAY_PS = -800
LATEST_DELAY_PS = 1100
POSZEROSHIFT = 0 #millimetres, applies to all scans calculated from raw ion histograms so is probably not useful here.


#FUNCTION TO GENERATE THE PLOTTABLE DATA
def calculating(folders: list[Path], configs: list[IonDataAnalysisConfig]) -> tuple[AveragedScansData, AggregateSpectrogram]:
    print('Starting: ',folders)
    
    scans_paths = DatFinder(folders).find_datafiles() #Change this if you want a specific path rather than the Droplets folder
    print('Data found!')
    raw_datas = load_ion_data(scans_paths)
    print('Data loaded!')
    calculated_scans = run_pipeline(raw_datas, configs)
    averagedScanData = average_scans(calculated_scans)
    print('Scans resampled!')
    
    return (averagedScanData)
#--------------------------------------------------------------------------------------------------

#Path to save figure in
fig_filedir = r"Z:\Droplets\plots" 
fig_filename = fig_filedir + r"\\fig5_longlasting_TEMP.pdf" #Name the file to save here

warnings: list[str] = []



#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#OCS DATA FIRST
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#OCS plots can have tracenames array (old from previous script...)
tracenames: list[str] = []
#Trace 2
configs_1: list[IonDataAnalysisConfig] = []
folders_1: list[Path] = [Path(r"20260211\Scan1")]
tracenames.append("3mJ usCFG (polarization averaged)")
#folders_1.append(Path(r"20260211\Scan1"))  

configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(92.654-POSZEROSHIFT, Prefix.MILLI),
    center=Point(174, 206),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](50, 120),
    transform_parameter= 0.78))

#Trace 2
#folders_2: list[Path] = [Path(r"20260211\Scan2")]
#tracenames.append("1.5mJ Circular Polarization")

#Trace 3
#folders_3: list[Path] = [Path(r"20260211\Scan4")]#scan 3 also but not needed.
#tracenames.append("1.5mJ // Linear Polarization")
#Trace 4
#folders_4: list[Path] = [Path(r"20260211\Scan5")]
#tracenames.append("1.5mJ // Linear Polarization \n with droplet beam blocked.")

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#CS2 DATA
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#CS2 IN DROPLETS WITH CFG
#Can look at 20260121\Scan2_ScanFiles and 20260120 scans 1 and 2, also 20260119\Scan2_ScanFiles for direct 1-arm comparison

cs2_droplets_folders: list[IonDataAnalysisConfig] = []
cs2_droplets_configs: list[Path] = []

cs2_droplets_folders.append(Path(r"20260120\Scan1")) 
cs2_droplets_configs.append(IonDataAnalysisConfig(
    delay_center= Length(98.054-POSZEROSHIFT, Prefix.MILLI),
    center=Point(228, 193),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.75))


cs2_droplets_folders.append(Path(r"20260120\Scan2")) 
cs2_droplets_configs.append(IonDataAnalysisConfig(
    delay_center= Length(98.054-POSZEROSHIFT, Prefix.MILLI),
    center=Point(228, 193),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.75))

cs2_droplets_folders.append(Path(r"20260119\Scan2_CFG")) 
cs2_droplets_configs.append(IonDataAnalysisConfig(
    delay_center= Length(98.054-POSZEROSHIFT, Prefix.MILLI),
    center=Point(228, 193),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.75))

##--------------------------------------------------------------------------------------------------

#Update the matplotlib settings
plt.style.use(r"stylefiles/compare_c2t_spectrogram.mplstyle")
##--------------------------------------------------------------------------------------------------

#Pipeline for the usCFG data
OCS_usCFG = calculating(folders_1, configs_1) #Same config used for each

cs2_droplets_CFG = calculating(cs2_droplets_folders, cs2_droplets_configs)
print('Done importing data.')
##--------------------------------------------------------------------------------------------------

#%%
#Main figure
mainfig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(3.375, 2)
        )

plot_averaged_scan(ax, OCS_usCFG, marker='x',color='magenta',ecolor=PlotColor.RED, label = None)
plot_averaged_scan(ax, cs2_droplets_CFG, marker='d',color=PlotColor.BLUE,ecolor=PlotColor.RED, label = None)

ax.set_ylabel(None)
ax.grid()
ax.set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
ax.set_xlabel('Probe Delay (ps)')
mainfig.supylabel('$\langle \cos^2 \\theta_{\mathrm{2D}} \\rangle$\n',horizontalalignment = 'right')


mainfig.savefig(fig_filename,format='pdf',dpi=300)
print("\n\nWARNINGS:\n",warnings)
plt.show()