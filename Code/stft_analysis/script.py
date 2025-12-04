from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from Lab_apps._base.models import Prefix, Time
from Lab_apps._io.dat_finder import DatFinder
from Lab_apps._io.dat_loader import load_time_scan, load_time_scans
from Lab_apps.stft_analysis.domain.config import AnalysisConfig
from Lab_apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from Lab_apps.stft_analysis.domain.resampling import resample_scan, resample_scans
from Lab_apps.stft_analysis.domain.stft_calculation import calculate_averaged_spectrogram, calculate_spectrogram

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