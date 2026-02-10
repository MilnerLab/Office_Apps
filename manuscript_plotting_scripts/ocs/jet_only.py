print('Code start!')
from pathlib import Path
from matplotlib import pyplot as plt

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from _data_io.dat_saver import create_save_path_for_calc_ScanFile
from apps.c2t_calculation.domain.config import IonDataAnalysisConfig
from apps.c2t_calculation.domain.pipeline import run_pipeline
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.models import AveragedScansData
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from apps.single_scan.domain.plotting import plot_single_scan
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.models import AggregateSpectrogram
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import calculate_averaged_spectrogram
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time

def calculating(folders: list[Path], configs: list[IonDataAnalysisConfig]) -> tuple[AveragedScansData, AggregateSpectrogram]:
    scans_paths = DatFinder(folders).find_datafiles() #Change this if you want a specific path rather than the Droplets folder
    raw_datas = load_ion_data(scans_paths, configs)
    save_path = create_save_path_for_calc_ScanFile(folders[0], str(raw_datas[0].ion_datas[0].run_id))
    calculated_scans = run_pipeline(raw_datas, save_path)
    averagedScanData = average_scans(calculated_scans)
    config = StftAnalysisConfig(calculated_scans)
    resampled_scans = resample_scans(calculated_scans, config.axis)
    spectrogram = calculate_averaged_spectrogram(resampled_scans, config)
    
    return (averagedScanData, spectrogram)

#Path to save figure in
fig_filedir = r"Z:\Droplets\plots" 
fig_filename = fig_filedir + r"\\jet_only_TEMP.png" #Name the file to save here

#Plot on top
PlotTitle = r'OCS Jet with usCFG set to max/min acceleration (above/below)'

#FIRST EXPERIMENT
# GA=26, DA = 16.3mm
configs_1: list[IonDataAnalysisConfig] = []
folders_1: list[Path] = []

folders_1.append(Path(r"20260206\Scan7"))
configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(92.654, Prefix.MILLI),
    center=Point(203, 202),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](30, 90),
    transform_parameter= 0.73))


#SECOND EXPERIMENT
# GA=0, DA = 16.6mm
configs_2: list[IonDataAnalysisConfig] = []
folders_2: list[Path] = []

folders_2.append(Path(r"20260207\Scan4"))
configs_2.append(IonDataAnalysisConfig(
    delay_center= Length(92.654, Prefix.MILLI),
    center=Point(203, 202),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](30, 90),
    transform_parameter= 0.73))

#--------------------------------------------------------------------------------------------------

#Pipeline 
plottable_scan_1, plottable_spectrogram_1 = calculating(folders_1, configs_1)
plottable_scan_2, plottable_spectrogram_2 = calculating(folders_2, configs_2)

#Plot histogram to check centre (change variable here)
if False:
    TestIndex = 4 #Delay point
    plot_radius = 150
    h_shift, edgex, edgey  = raw_datas[0].ion_datas[TestIndex].get_2D_histogram(num_bins=2*plot_radius,xy_range=Range(-plot_radius,plot_radius))

    #Create figures
    ionfig, (ax_shift) = plt.subplots(
                nrows=1,
                ncols=1,
                figsize=(4, 4), 
            )

    ax_shift.pcolor(edgex,edgey,h_shift)
    ax_shift.axis('equal')

#Main figure
mainfig, (axs) = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(10, 8),
            sharex=True, 
            gridspec_kw={'hspace': 0,'wspace' : 0.275},
        )


#Plot first experiment in top row
a = axs[0,0]
plot_averaged_scan(a, plottable_scan_1, PlotColor.BLUE,ecolor=PlotColor.RED)
#a.legend(loc="upper right")
a = axs[0,1]
plot_Spectrogram(a, plottable_spectrogram_1)
a.set_ylim([0,120])
plot_nyquist_frequency(a, plottable_scan_1)


#Plot second experiment in bottom row
a = axs[1,0]
plot_averaged_scan(a, plottable_scan_2, PlotColor.BLUE,ecolor=PlotColor.RED)
#a.legend(loc="upper right")
a = axs[1,1]
plot_Spectrogram(a, plottable_spectrogram_2)
a.set_ylim([0,120])
plot_nyquist_frequency(a, plottable_scan_2)

mainfig.suptitle(PlotTitle)

mainfig.savefig(fig_filename,format='png')
plt.show()