# processing/plotter.py
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Time
import numpy as np

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_time_scans
from _domain.models import LoadableScanData
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from apps.single_scan.domain.plotting import plot_single_scan
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import calculate_averaged_spectrogram

SPECTROGRAM_THRESHOLD = Time(10, Prefix.PICO)

plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
})

class PlottingBotPlotting:
    def __init__(self,
    dat_finder: DatFinder,
    output_path: Path,
    color_cos2: PlotColor,
    color_data_ions: PlotColor) -> None:
        self.scans = load_time_scans(dat_finder.find_scanfiles(mergescans=False))
        self.current_scan = self.scans[-1]
        self.output_path = output_path
        self.color_cos2 = color_cos2
        self.color_data_ions = color_data_ions
        
        self.plot_double()
    
    def plot_double(self):
        fig, (ax_scan, ax_spec) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(10, 8),
            sharex=True 
        )
        plot_single_scan(ax_scan, self.current_scan, True, self.color_cos2, self.color_data_ions)
        ax_scan.legend(loc="upper right")

        if np.mean(np.abs(np.diff(self.current_scan.delay)))  < SPECTROGRAM_THRESHOLD:
            averagedScanData = average_scans(self.scans)
            plot_averaged_scan(ax_scan, averagedScanData, PlotColor.PURPLE)
            self.add_Spectrogram(ax_spec, self.scans)
            ax_spec.legend(loc="upper left")
        else:
            averagedScanData = average_scans(self.scans)
            plot_averaged_scan(ax_spec, averagedScanData)
            ax_spec.legend(loc="upper right")
        
        self.end(fig)
        
    def plot_double(self):
        use_spec = np.mean(np.abs(np.diff(self.current_scan.delay))) < SPECTROGRAM_THRESHOLD

        if use_spec:
            fig, (ax_scan, ax_avg, ax_spec) = plt.subplots(
                nrows=3,
                ncols=1,
                figsize=(10, 10),
                sharex=True,
                constrained_layout=True,
            )

            plot_single_scan(ax_scan, self.current_scan, True, self.color_cos2, self.color_data_ions)
            ax_scan.legend(loc="upper right")

            averagedScanData = average_scans(self.scans)
            plot_averaged_scan(ax_avg, averagedScanData, PlotColor.PURPLE)
            ax_avg.legend(loc="upper right")

            self.add_Spectrogram(ax_spec, self.scans)
            ax_spec.legend(loc="upper left")

        else:
            fig, (ax_scan, ax_avg) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(10, 8),
                sharex=True,
                constrained_layout=True,
            )

            plot_single_scan(ax_scan, self.current_scan, True, self.color_cos2, self.color_data_ions)
            ax_scan.legend(loc="upper right")

            averagedScanData = average_scans(self.scans)
            plot_averaged_scan(ax_avg, averagedScanData)
            ax_avg.legend(loc="upper right")

        self.end(fig)

    
    '''
    def plot_single(self):
        fig, ax_scan = plt.subplots(figsize=(10, 6))
        plot_single_scan(ax_scan, self.scan, True, self.color_cos2, self.color_data_ions)
        ax_scan.legend(loc="upper right")
        self.end(fig, self.output_path)
    '''

    def end(self, fig: Figure) -> None:
        fig.tight_layout()
        fig.savefig(self.output_path, dpi=50)
        plt.close(fig)
    
    def add_Spectrogram(self, ax: Axes, scans: list[LoadableScanData]) -> None:
        config = StftAnalysisConfig(scans,stft_window_size=Time(100,Prefix.PICO))
        resampled_scans = resample_scans(scans, config.axis)
        spectrogram = calculate_averaged_spectrogram(resampled_scans, config)
        plot_Spectrogram(ax, spectrogram)
        plot_nyquist_frequency(ax, self.current_scan)
        