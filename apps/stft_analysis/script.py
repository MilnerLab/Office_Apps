from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import calculate_averaged_spectrogram
from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_time_scan, load_time_scans
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.models import AveragedScansData
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Time

folder_path = Path(r"/mnt/valeryshare/Droplets/")
file_paths_avg = DatFinder().find_scanfiles()
scan_data_avg = load_time_scans(file_paths)

config = StftAnalysisConfig(scan_data)

resampled_scans = resample_scans(scan_data, config.axis)
spectrogram = calculate_averaged_spectrogram(resampled_scans, config)

fig, ax = plt.subplots(figsize=(8, 4))
plot_Spectrogram(ax, spectrogram)
plot_nyquist_frequency(ax, scan_data[0])
fig.tight_layout()
plt.show()