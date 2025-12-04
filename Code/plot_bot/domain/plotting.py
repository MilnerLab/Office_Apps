# processing/plotter.py
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from Lab_apps._base.models import PlotColor, Prefix, Time
from Lab_apps._domain.models import LoadableScanData
from Lab_apps._io.dat_finder import DatFinder
from Lab_apps._io.dat_loader import load_time_scans
from Lab_apps.scan_averaging.domain.averaging import average_scans
from Lab_apps.scan_averaging.domain.plotting import plot_averaged_scan
from Lab_apps.single_scan.domain.plotting import plot_single_scan
from Lab_apps.stft_analysis.domain.config import AnalysisConfig
from Lab_apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from Lab_apps.stft_analysis.domain.resampling import resample_scan, resample_scans
from Lab_apps.stft_analysis.domain.stft_calculation import calculate_averaged_spectrogram

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
    color_ions: PlotColor) -> None:
        self.scans = load_time_scans(dat_finder.find_scanfiles())
        self.current_scan = self.scans[-1]
        self.output_path = output_path
        self.color_cos2 = color_cos2
        self.color_ions = color_ions
    
    def plot_double(self):
        fig, (ax_scan, ax_spec) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(10, 8),
            sharex=True 
        )
        plot_single_scan(ax_scan, self.current_scan, True, self.color_cos2, self.color_ions)
        ax_scan.legend(loc="upper right")

        if np.mean(np.abs(np.diff(self.current_scan.delay)))  < SPECTROGRAM_THRESHOLD:
            self.add_Spectrogram(ax_spec, self.scans)
            ax_spec.legend(loc="upper left")
        else:
            averagedScanData = average_scans(self.scans)
            plot_averaged_scan(ax_spec, averagedScanData)
        
        self.end(fig)
    
    '''
    def plot_single(self):
        fig, ax_scan = plt.subplots(figsize=(10, 6))
        plot_single_scan(ax_scan, self.scan, True, self.color_cos2, self.color_ions)
        ax_scan.legend(loc="upper right")
        self.end(fig, self.output_path)
    '''

    def end(self, fig: Figure) -> None:
        fig.tight_layout()
        fig.savefig(self.output_path, dpi=50)
        plt.close(fig)
    
    def add_Spectrogram(self, ax: Axes, scans: list[LoadableScanData]) -> None:
        config = AnalysisConfig(scans)
        resampled_scans = resample_scans(scans, config.axis)
        spectrogram = calculate_averaged_spectrogram(resampled_scans, config)
        plot_Spectrogram(ax, spectrogram)
        plot_nyquist_frequency(ax, self.current_scan)
        