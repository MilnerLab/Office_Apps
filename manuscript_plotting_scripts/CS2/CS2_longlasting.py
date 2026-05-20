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
EARLIEST_DELAY_PS = -800
LATEST_DELAY_PS = 1250
POSZEROSHIFT = 0 #millimetres :)

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
fig_filename = fig_filedir + r"\CS2_longlasting_TEMP.png" #Name the file to save here

#Path to save processed data in
savedata_filedir = r"Z:\Droplets\exportdata" 
savedata_filename_1 = savedata_filedir + r"\CS2_longlasting_jet.csv" #Name the file to save here
savedata_filename_2 = savedata_filedir + r"\CS2_longlasting_droplets.csv" #Name the file to save here
savedata_filename_3 = savedata_filedir + r"\CS2_longlasting_droplets_singlearm.csv" #Name the file to save here


#Plot on top


PlotTitle = r"CS$_2$ in jet (first row) vs droplets (second row) " 


#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
# JET
configs_1: list[IonDataAnalysisConfig] = []
folders_1: list[Path] = []

# folders_1.append(Path(r"20251128\Scan4"))  #20251128\Scan4 set has a different maximum from 20260112\4+5 that looks weird mid-scan
# configs_1.append(IonDataAnalysisConfig(
#     delay_center= Length(91-POSZEROSHIFT, Prefix.MILLI),
#     center=Point(199, 203),
#     angle= Angle(12, AngleUnit.DEG),
#     analysis_zone= Range[int](25, 120),
#     transform_parameter=0.9))

folders_1.append(Path(r"20260112\JetScan4")) 
configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(98.054-POSZEROSHIFT, Prefix.MILLI),
    center=Point(214, 191),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](25, 120),
    transform_parameter=0.75))

folders_1.append(Path(r"20260112\JetScan5")) 
configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(98.054-POSZEROSHIFT, Prefix.MILLI),
    center=Point(214, 191),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](25, 120), 
    transform_parameter=0.75))


#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
# DROPLETS CFG
#Can look at 20260121\Scan2_ScanFiles and 20260120 scans 1 and 2, also 20260119\Scan2_ScanFiles for direct 1-arm comparison

configs_2: list[IonDataAnalysisConfig] = []
folders_2: list[Path] = []

folders_2.append(Path(r"20260120\Scan1")) 
configs_2.append(IonDataAnalysisConfig(
    delay_center= Length(98.054-POSZEROSHIFT, Prefix.MILLI),
    center=Point(228, 193),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.75))


folders_2.append(Path(r"20260120\Scan2")) 
configs_2.append(IonDataAnalysisConfig(
    delay_center= Length(98.054-POSZEROSHIFT, Prefix.MILLI),
    center=Point(228, 193),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.75))

folders_2.append(Path(r"20260119\Scan2_CFG")) 
configs_2.append(IonDataAnalysisConfig(
    delay_center= Length(98.054-POSZEROSHIFT, Prefix.MILLI),
    center=Point(228, 193),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.75))

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
# DROPLETS SINGLE ARM

configs_3: list[IonDataAnalysisConfig] = []
folders_3: list[Path] = []


# folders_3.append(Path(r"20260114\Scan1_HorizontalDA")) 
# configs_3.append(IonDataAnalysisConfig(
#     delay_center= Length( 98.054-POSZEROSHIFT, Prefix.MILLI),
#     center=Point(188, 200),
#     angle= Angle(12, AngleUnit.DEG),
#     analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
#     transform_parameter=0.73))

folders_3.append(Path(r"20260119\Scan1_Horizontal")) 
configs_3.append(IonDataAnalysisConfig(
    delay_center= Length(98.054-POSZEROSHIFT, Prefix.MILLI),
    center=Point(228, 193),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.75))

# folders_3.append(Path(r"20260203\Scan1_horizontal")) 
# configs_3.append(IonDataAnalysisConfig(
#     delay_center= Length(89.65-POSZEROSHIFT, Prefix.MILLI),
#     center=Point(181, 205),
#     angle= Angle(12, AngleUnit.DEG),
#     analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
#     transform_parameter=0.73))


#--------------------------------------------------------------------------------------------------
#Update the matplotlib settings
plt.style.use(r"stylefiles\compare_c2t_spectrogram.mplstyle")

#Pipeline 
plottable_scan_1  = calculating(folders_1, configs_1)
plottable_scan_2 = calculating(folders_2, configs_2)
plottable_scan_3 = calculating(folders_3, configs_3)


#Main figure
mainfig, (axs) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(6.75, 4.2),
            sharex=True,             
            gridspec_kw={'hspace': 0.1,'wspace': 0.3}
        )

#Plot first experiment in top row
a = axs[0]
plot_averaged_scan(a, plottable_scan_1, PlotColor.BLACK,ecolor=PlotColor.RED,marker='d', label = "3 mJ Polarization-Averaged Centrifuge",elinewidth=1)
a.grid()
a.set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
a.set_xlabel(None)

a.legend(loc='upper right')

#Plot second experiment in bottom row
a = axs[1]
plot_averaged_scan(a, plottable_scan_2, PlotColor.BLACK,ecolor=PlotColor.RED,marker='d',label="3 mJ Polarization-Averaged Centrifuge",elinewidth=1)
plot_averaged_scan(a, plottable_scan_3, PlotColor.GRAY,ecolor=PlotColor.GRAY,marker='s',label="1.5 mJ Linearly Polarized Field",elinewidth=1)

a.grid()
a.legend(loc='upper right')


mainfig.suptitle(PlotTitle,fontsize=MAJORTITLEFONTSIZE,color='black')

#Save scans
plottable_scan_1.to_csv(savedata_filename_1)
plottable_scan_2.to_csv(savedata_filename_2)
plottable_scan_3.to_csv(savedata_filename_3)

mainfig.savefig(fig_filename,format='png',dpi=300)
plt.show()
print('Done!')