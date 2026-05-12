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
EARLIEST_DELAY_PS = -230
LATEST_DELAY_PS = -EARLIEST_DELAY_PS
POSZEROSHIFT = 0 #millimetres :)

MAJORTITLEFONTSIZE = 16
YLABELX = -0.135

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
fig_filename = fig_filedir + r"\OCS_directionality_droplets_TEMP.png" #Name the file to save here

#Path to save processed data in
savedata_filedir = r"Z:\Droplets\exportdata" 
savedata_filename_1 = savedata_filedir + r"\OCS_accelerating_droplets.csv" #Name the file to save here
savedata_filename_2 = savedata_filedir + r"\OCS_decelerating_droplets.csv" #Name the file to save here


#Plot on top


PlotTitle = r"OCS in 30 bar / 18 K droplets"


#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------

#ACCELERATING


# configs_1: list[IonDataAnalysisConfig] = []
# folders_1: list[Path] = []

# folders_1.append(Path(r"20260426\Scan3"))  #Correct for direct comparison to 20260427\Scan2,  GA=0, DA = 15.5mm
# configs_1.append(IonDataAnalysisConfig(
#     delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
#     center=Point(205, 194),
#     angle= Angle(12, AngleUnit.DEG),
#     analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
#     transform_parameter=0.78))


configs_1: list[IonDataAnalysisConfig] = []
folders_1: list[Path] = []

folders_1.append(Path(r"20260210\Scan4")) #better older data, GA=0, DA = 16.6mm, still has ~40GHz central oscillation frequency 
configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(92.654-POSZEROSHIFT, Prefix.MILLI),
    center=Point(175, 205),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.75))


#DECELERATING
# GA=0, DA = 16.43mm
configs_2: list[IonDataAnalysisConfig] = []
folders_2: list[Path] = []

folders_2.append(Path(r"20260427\Scan2")) 
configs_2.append(IonDataAnalysisConfig(
    delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
    center=Point(205, 194),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.78))


#--------------------------------------------------------------------------------------------------
#Update the matplotlib settings
plt.style.use(r"stylefiles\compare_c2t_spectrogram.mplstyle")

#Pipeline 
plottable_scan_1, plottable_spectrogram_1 = calculating(folders_1, configs_1)
plottable_scan_2, plottable_spectrogram_2 = calculating(folders_2, configs_2)


#Main figure
mainfig, (axs) = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(6.75, 4.2),
            sharex=True,             
            gridspec_kw={'hspace': 0.1,'wspace': 0.3}
        )

#Plot first experiment in top row
a = axs[0,0]
plot_averaged_scan(a, plottable_scan_1, PlotColor.BLACK,ecolor=PlotColor.RED,marker='d', label = None,elinewidth=0)
a.grid()
a.set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
a.set_xlabel(None)

a = axs[0,1]
plot_Spectrogram(a, plottable_spectrogram_1,shading="auto",v_range=Range(0,0.6))
a.set_ylim([0,100])
a.set_ylabel('Oscillation\nFrequency (GHz)')
a.yaxis.set_label_coords(YLABELX,0.5)

#plot_nyquist_frequency(a, plottable_scan_1)
a.set_xlabel(None)

#Plot second experiment in bottom row
a = axs[1,0]
plot_averaged_scan(a, plottable_scan_2, PlotColor.BLACK,ecolor=PlotColor.RED,marker='d',label=None,elinewidth=0)
a.grid()

a = axs[1,1]
plot_Spectrogram(a, plottable_spectrogram_2,shading="auto",v_range=Range(0,0.4))
print('Colour axis is different for second row.')
a.set_ylim([0,100])
a.set_ylabel('Oscillation\nFrequency (GHz)')
a.yaxis.set_label_coords(YLABELX,0.5)
#plot_nyquist_frequency(a, plottable_scan_2)

mainfig.suptitle(PlotTitle,fontsize=MAJORTITLEFONTSIZE,color='black')

#Save scans
plottable_scan_1.to_csv(savedata_filename_1)
plottable_scan_2.to_csv(savedata_filename_2)

mainfig.savefig(fig_filename,format='png',dpi=300)
plt.show()
print('Done!')