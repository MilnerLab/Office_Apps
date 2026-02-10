print('Code start!')
from pathlib import Path
from matplotlib import pyplot as plt

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from _data_io.dat_saver import create_save_path_for_calc_ScanFile
from apps.c2t_calculation.domain.config import IonDataAnalysisConfig
from apps.c2t_calculation.domain.pipeline import run_pipeline
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from apps.single_scan.domain.plotting import plot_single_scan
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import calculate_averaged_spectrogram
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time
print('Dependencies')


# GA=26, DA = 16.3mm

configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []

folders.append(Path(r"20260206\Scan7"))
configs.append(IonDataAnalysisConfig(
    delay_center= Length(92.654, Prefix.MILLI),
    center=Point(203, 202),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](30, 90),
    transform_parameter= 0.73))

scans_paths = DatFinder(folders).find_datafiles()

raw_datas = load_ion_data(scans_paths, configs)
print('Data loaded!')
save_path = create_save_path_for_calc_ScanFile(folders[0], str(raw_datas[0].ion_datas[0].run_id))
calculated_scans = run_pipeline(raw_datas, save_path)

TestIndex = 4 #Delay point
plot_radius = 150
h_shift, edgex, edgey  = raw_datas[0].ion_datas[TestIndex].get_2D_histogram(num_bins=2*plot_radius,xy_range=Range(-plot_radius,plot_radius))


averagedScanData = average_scans(calculated_scans)
config = StftAnalysisConfig(calculated_scans)
resampled_scans = resample_scans(calculated_scans, config.axis)
spectrogram = calculate_averaged_spectrogram(resampled_scans, config)


ionfig, (ax_shift) = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(4, 4), 
        )

ax_shift.pcolor(edgex,edgey,h_shift)
ax_shift.axis('equal')


fig, (axs) = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(10, 8),
            sharex=True 
        )

a = axs[0,0]
plot_averaged_scan(a, averagedScanData, PlotColor.GREEN, label=" -> CFG randomized")
a.legend(loc="upper right")


a = axs[0,1]

plot_Spectrogram(a, spectrogram)
a.set_ylim([0,120])
plot_nyquist_frequency(a, calculated_scans[0])

plt.show()