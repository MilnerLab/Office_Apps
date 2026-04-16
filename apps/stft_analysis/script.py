from pathlib import Path
from turtle import st
from matplotlib import pyplot as plt
import numpy as np
from yaml import scan
from matplotlib.widgets import Button

from _domain.plotting import plot_GaussianFit
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.scan_averaging.domain import averaging
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

folder_path = Path(r"/mnt/valeryshare/Droplets/20260408/Scan1")
file_paths = DatFinder(folder_path,is_full_path=True).find_datafiles()

config = IonDataAnalysisConfig(
    delay_center= Length(93.3, Prefix.MILLI),
    center=Point(217, 194),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](50, 110),
    transform_parameter=0.75)

raw_scans = load_ion_data(file_paths)
calculated_Scan = run_pipeline(raw_scans, [config])

config = StftAnalysisConfig(calculated_Scan, Time(180, Prefix.PICO))
resampled_scans = resample_scans(calculated_Scan,config.axis)
averaged_data = average_scans(calculated_Scan)
resampled_scan = resample_scans([averaged_data],config.axis)
_, baseline = resampled_scan[0].detrend_moving_average()


fig,(ax1,ax2) = plt.subplots(2,1,figsize=(8,5))

plot_averaged_scan(ax1,data=averaged_data)
x = [time.value(Prefix.PICO) for time in resampled_scan[0].delays]
ax1.plot(x, baseline)
spectrogram = StftAnalysis(resampled_scans,config).calculate_averaged_spectrogram()
plot_Spectrogram(ax2,spectrogram)
plot_nyquist_frequency(ax2,calculated_Scan[0])
ax1.set_xlim(-250,250)
ax2.set_xlim(-250,250)
fig.suptitle('CS2 Jet - window 180ps', fontsize=12)
fig.tight_layout()
#fig.savefig(fig_path,format='png')
plt.show()