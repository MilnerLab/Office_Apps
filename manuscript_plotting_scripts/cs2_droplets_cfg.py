from pathlib import Path
import matplotlib.pyplot as plt
from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from _data_io.dat_saver import create_save_path_for_calc_ScanFile
from _domain.plotting import plot_GaussianFit
from apps.c2t_calculation.domain.config import IonDataAnalysisConfig
from apps.c2t_calculation.domain.pipeline import run_pipeline
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import calculate_averaged_spectrogram
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length


folder_path = Path(r"/mnt/valeryshare/Droplets/20260128/Scan1_CFG")
file_paths = DatFinder(folder_path).find_datafiles()

config_1 = IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 205),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](60, 120),
    transform_parameter= 0.73)


ion_data = load_ion_data(file_paths, config_1.delay_center)
save_path = create_save_path_for_calc_ScanFile(folder_path, str(ion_data[0].run_id))
calculated_Scan_1 = run_pipeline(ion_data,config_1, save_path)




folder_path = Path(r"/mnt/valeryshare/Droplets/20260128/Scan2_CFG")
file_paths = DatFinder(folder_path).find_datafiles()

config_2 = IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 205),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](60, 120),
    transform_parameter= 0.73)


ion_data = load_ion_data(file_paths, config_2.delay_center)
save_path = create_save_path_for_calc_ScanFile(folder_path, str(ion_data[0].run_id))
calculated_Scan_2 = run_pipeline(ion_data,config_2, save_path)



folder_path = Path(r"/mnt/valeryshare/Droplets/20260129/Scan1")
file_paths = DatFinder(folder_path).find_datafiles()

config_3 = IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](60, 120),
    transform_parameter= 0.73)


ion_data = load_ion_data(file_paths, config_3.delay_center)
save_path = create_save_path_for_calc_ScanFile(folder_path, str(ion_data[0].run_id))
calculated_Scan_3 = run_pipeline(ion_data,config_3, save_path)


folder_path = Path(r"/mnt/valeryshare/Droplets/20260130/Scan2")
file_paths = DatFinder(folder_path).find_datafiles()

config_4 = IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](60, 120),
    transform_parameter= 0.73)


ion_data = load_ion_data(file_paths, config_4.delay_center)
save_path = create_save_path_for_calc_ScanFile(folder_path, str(ion_data[0].run_id))
calculated_Scan_4 = run_pipeline(ion_data,config_4, save_path)

folder_path = Path(r"/mnt/valeryshare/Droplets/20260130/Scan3")
file_paths = DatFinder(folder_path).find_datafiles()

config_5 = IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](60, 120),
    transform_parameter= 0.73)


ion_data = load_ion_data(file_paths, config_5.delay_center)
save_path = create_save_path_for_calc_ScanFile(folder_path, str(ion_data[0].run_id))
calculated_Scan_5 = run_pipeline(ion_data,config_5, save_path)

scans = [calculated_Scan_1, calculated_Scan_2, calculated_Scan_3, calculated_Scan_4, calculated_Scan_5]


config = StftAnalysisConfig(scans)

resampled_scans = resample_scans(scans, config.axis)

averagedScanData = average_scans(resampled_scans)
fig, ax = plt.subplots(figsize=(8, 4))
plot_averaged_scan(ax, averagedScanData, PlotColor.GREEN, label=" -> CFG randomized")
plot_GaussianFit(ax, averagedScanData)
fig.suptitle('Droplets', fontsize=12)
ax.legend(loc="upper left")
fig.tight_layout()
plt.show()

config = StftAnalysisConfig(scans)

resampled_scans = resample_scans(scans, config.axis)
spectrogram = calculate_averaged_spectrogram(resampled_scans, config)

fig, ax = plt.subplots(figsize=(8, 4))
plot_Spectrogram(ax, spectrogram)
plot_nyquist_frequency(ax, scans[0])
fig.tight_layout()
plt.show()
