from pathlib import Path
from turtle import st
from matplotlib import pyplot as plt
import numpy as np

from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_time_scan, load_time_scans
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.models import AveragedScansData
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from apps.single_scan.domain.plotting import plot_single_scan
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Time

""" folder_path1 = Path(r"Z:\Droplets\20260211\Scan1_ScanFiles")
folder_path2 = Path(r"Z:\Droplets\20260211\Scan2_ScanFiles")
folder_path4 = Path(r"Z:\Droplets\20260211\Scan4_ScanFiles")

file_paths_avg1 = DatFinder(folder_path1).find_scanfiles()
file_paths_avg2 = DatFinder(folder_path2).find_scanfiles()
file_paths_avg4 = DatFinder(folder_path4).find_scanfiles()

    
scan_data_avg1 = load_time_scans(file_paths_avg1)
scan_data_avg2 = load_time_scans(file_paths_avg2)
scan_data_avg4 = load_time_scans(file_paths_avg4)

config1 = StftAnalysisConfig(scan_data_avg1)
config2 = StftAnalysisConfig(scan_data_avg2)
config4 = StftAnalysisConfig(scan_data_avg4)
#print(config.stft_window_size)
resampled_scans1 = resample_scans(scan_data_avg1, config1.axis)
resampled_scans2 = resample_scans(scan_data_avg2, config2.axis)
resampled_scans4 = resample_scans(scan_data_avg4, config4.axis)

ax = np.empty((3,2), dtype=object)
fig,ax = plt.subplots(3,2, figsize=(10,12))
averagedScanData1 = average_scans(scan_data_avg1)
averagedScanData2 = average_scans(scan_data_avg2)
averagedScanData4 = average_scans(scan_data_avg4)
plot_averaged_scan(ax=ax[0,0],data=averagedScanData1, label="Averaged Scans", color=PlotColor.RED)
plot_averaged_scan(ax=ax[1,0],data=averagedScanData2, label="Averaged Scans", color=PlotColor.RED)
plot_averaged_scan(ax=ax[2,0],data=averagedScanData4)
spectrogram1 = calculate_averaged_spectrogram(resampled_scans1, config1)
spectrogram2 = calculate_averaged_spectrogram(resampled_scans2, config2)
spectrogram4 = calculate_averaged_spectrogram(resampled_scans4, config4)
plot_Spectrogram(ax[0,0], spectrogram1)
plot_Spectrogram(ax[0,1], spectrogram2)
plot_Spectrogram(ax[0,2], spectrogram4) """

fig_path = Path(r"C:\Users\camp06\OneDrive - UBC\Documents\droplets_manuscript\test\20260219_jet_avg.png")
file_paths = [Path(r"Z:\Droplets\20260227\Scan1_ScanFiles\20260227145137_ScanFile.dat"),Path(r"Z:\Droplets\20260227\Scan1_ScanFiles\20260227151207_ScanFile.dat"),
    Path(r"Z:\Droplets\20260227\Scan1_ScanFiles\20260227153448_ScanFile.dat"),Path(r"Z:\Droplets\20260227\Scan1_ScanFiles\20260227160304_ScanFile.dat")]
#file_paths_avg = DatFinder(folder_path).find_scanfiles()
#scan_data_avg = load_time_scans(file_paths_avg)

file_paths_avg = DatFinder(Path(r"Z:\Droplets\20260227\Scan1_ScanFiles")).find_scanfiles()
scan_data_avg = load_time_scans(file_paths_avg)

""" fig,(axs) = plt.subplots(2,3,figsize=(16,8))
i = 0
for scan in scan_data_avg:
    if i <= 2:
        plot_single_scan(axs[0,i],data=scan)
    else: 
        plot_single_scan(axs[1,i-3],data=scan)
    i += 1
fig.tight_layout()
fig.savefig(fig_path,format='png')
plt.show() """

config = StftAnalysisConfig(scan_data_avg)
resampled_scans = resample_scans(scan_data_avg,config.axis)
averaged_data = average_scans(scan_data_avg)


fig,(ax1,ax2) = plt.subplots(2,1,figsize=(8,5))

plot_averaged_scan(ax1,data=averaged_data)
spectrogram = StftAnalysis(resampled_scans,config).calculate_averaged_spectrogram()
plot_Spectrogram(ax2,spectrogram)
plot_nyquist_frequency(ax2,scan_data_avg[0])
ax1.set_xlim(-250,250)
ax2.set_xlim(-250,250)
fig.suptitle('CS2 Droplets', fontsize=12)
fig.tight_layout()
#fig.savefig(fig_path,format='png')
plt.show()