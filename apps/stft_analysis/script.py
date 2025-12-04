from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from apps.stft_analysis.domain.config import AnalysisConfig
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import calculate_averaged_spectrogram
from base_lib.models import Prefix, Time
from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_time_scan, load_time_scans

folder_path = Path(r"Z:\Droplets\20251127\Scan3_ScanFiles")
file_paths = DatFinder().find_scanfiles()
scan_data = load_time_scans(file_paths)

config = AnalysisConfig(scan_data)

resampled_scans = resample_scans(scan_data, config.axis)
spectrogram = calculate_averaged_spectrogram(resampled_scans, config)

fig, ax = plt.subplots(figsize=(8, 4))
plot_Spectrogram(ax, spectrogram)
plot_nyquist_frequency(ax, scan_data[0])
fig.tight_layout()
plt.show()