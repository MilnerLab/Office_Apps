from pathlib import Path
from turtle import st
from matplotlib import pyplot as plt
import numpy as np
from yaml import scan
from matplotlib.widgets import Button

from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.c2t_calculation.domain.plotting import plot_calculated_scan
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from apps.scan_averaging.domain.averaging import average_scans
#from apps.scan_averaging.domain.models import AveragedScansData
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from apps.single_scan.domain.plotting import plot_single_scan
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length
from base_core.math.models import Point, Angle, Range
from base_core.math.enums import AngleUnit

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

def main() -> None:
    fig_path_root = Path(r"C:\Users\camp06\OneDrive - UBC\Documents")
    folder_path = Path(r"Z:\Droplets\20260323\Scan1")
    
    
    

    config = IonDataAnalysisConfig(
        delay_center= Length(93.3, Prefix.MILLI),
        center=Point(175, 202),
        angle= Angle(12, AngleUnit.DEG),
        analysis_zone= Range[int](60, 90),
        transform_parameter=0.76)

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(8,5))
    ax1.set_xlim(-300,300)
    ax2.set_xlim(-300,300)
    fig.suptitle('CS2 Droplets', fontsize=12)
    
    button_ax = fig.add_axes((0.8, 0.05, 0.15, 0.075))
    refresh_button = Button(button_ax, "Refresh")

    def on_refresh(event):
            file_paths_avg = DatFinder(folder_path,is_full_path=True).find_datafiles()
            raw_scans = load_ion_data(file_paths_avg)
            num_scans = raw_scans[0].number_of_scans
            c2t_data = run_pipeline(raw_scans,config)
            #scan_avg = average_scans(c2t_data)
            stft_config = StftAnalysisConfig(c2t_data)
            resampled_scans = resample_scans(c2t_data,stft_config.axis)
            
            ax1.clear()
            ax2.clear()
            ax1.grid(visible=True,which='major',alpha=0.5)
            ax2.grid(visible=True,which='major',alpha=0.5)
            plot_calculated_scan(ax1,data=c2t_data[0],number_of_scans=num_scans)
            spectrogram = StftAnalysis(resampled_scans,stft_config).calculate_averaged_spectrogram()
            plot_Spectrogram(ax2,spectrogram)
            plot_nyquist_frequency(ax2,c2t_data[0])
            fig.canvas.draw_idle()

    refresh_button.on_clicked(on_refresh)
    
    #fig.savefig(fig_path,format='png')
    plt.show()

if __name__ == "__main__":
    main()
