from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from _data_io.dat_saver import create_save_path_for_calc_ScanFile
from _domain.plotting import plot_GaussianFit
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.models import AggregateSpectrogram
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scan, resample_scans
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from base_core.lab_specifics.averaging.models import AveragedScansData
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time


mpl.rcParams.update({
#copied from physrev.mplstyle file
    
    # --- Axes ---
    "axes.titlesize": "medium",
    "axes.labelsize": 9,
    "axes.formatter.use_mathtext": True,
    "axes.linewidth": 0.5,
    #"axes.grid": True,
    #"axes.grid.axis": "both",  # which axis the grid should apply to
    #"axes.grid.which": "major",
    #"axes.axisbelow" : True,
    #"grid.alpha": 0.25,

    # --- Grid lines ---
    #"grid.linewidth": 0.5,
    #"grid.linestyle": "dashed",
    #"grid.color": "xkcd:light gray",

    # --- Lines ---
    "lines.linewidth": 0.5,
    "lines.marker": "o",
    "lines.markersize": 1.5,
    "hatch.linewidth": 0.25,
    "patch.antialiased": True,
    
    #---Errorbars---
    "errorbar.capsize": 0,

    # --- Ticks (X) ---
    #"xtick.top": True,
    "xtick.bottom": True,
    "xtick.major.size": 3.0,
    "xtick.minor.size": 1.5,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "xtick.direction": "out",
    "xtick.minor.visible": False,
    #"xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.bottom": True,
    "xtick.major.pad": 5.0,
    "xtick.minor.pad": 5.0,
    "xtick.labelsize": 9,

    # --- Ticks (Y) ---
    "ytick.left": True,
    #"ytick.right": True,
    "ytick.major.size": 3.0,
    #"ytick.minor.size": 1.5,
    "ytick.major.width": 0.5,
    #"ytick.minor.width": 0.5,
    "ytick.direction": "out",
    #"ytick.minor.visible": True,
    "ytick.major.left": True,
    #"ytick.major.right": True,
    #"ytick.minor.left": True,
    "ytick.major.pad": 5.0,
    #"ytick.minor.pad": 5.0,
    "ytick.labelsize": 9,
    
    
    # --- Legend ---
    "legend.frameon": True,
    "legend.fontsize": 8,
    "legend.handlelength": 1.375,
    "legend.labelspacing": 0.4,
    "legend.columnspacing": 1,
    "legend.facecolor": "white",
    "legend.edgecolor": "white",
    "legend.framealpha": 1,
    "legend.title_fontsize": 8,
 
    # --- Figure size ---
    "figure.figsize": (3.375, 4), #1- column fig
    #"figure.figsize": (6.75, 6.75), #approx. 2- column fig
    "figure.subplot.left": 0.125,
    "figure.subplot.bottom": 0.175,
    "figure.subplot.top": 0.95,
    "figure.subplot.right": 0.95,
    "figure.autolayout": True,

    # --- Fonts (computer modern) ---
    "text.usetex": True,      
    #"mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"]

})

#FUNCTION TO GENERATE THE PLOTTABLE DATA
def calculating(folders: list[Path], configs: list[IonDataAnalysisConfig]) -> tuple[AveragedScansData, AveragedScansData, AggregateSpectrogram, list[Time]]:
    print('Starting: ',folders)
    
    scans_paths = DatFinder(folders).find_datafiles() #Change this if you want a specific path rather than the Droplets folder
    print('Data loaded!')
    raw_datas = load_ion_data(scans_paths, configs)
    print('Data loaded!')
    save_path = create_save_path_for_calc_ScanFile(folders[0], str(raw_datas[0].ion_datas[0].run_id))
    calculated_scans = run_pipeline(raw_datas, save_path)
    averagedScanData = average_scans(calculated_scans)
    config = StftAnalysisConfig(calculated_scans)
    resampled_scans = resample_scans(calculated_scans, config.axis)
    print('Scans resampled!')
    spectrogram = StftAnalysis(resampled_scans, config).calculate_averaged_spectrogram()
    
    return (averagedScanData, average_scans(resampled_scans), spectrogram, config.axis)

fig_filedir = r"C:/Users/camp06/Documents/droplets_manuscript/"
fig_filedir = r"/home/soeren/Desktop/"
fig_filename = fig_filedir + r"130-140.pdf"

configs_1: list[IonDataAnalysisConfig] = []
folders_1: list[Path] = []

folders_1.append(Path(r"20260128/Scan1_CFG"))

ring = Range[int](60, 120)

configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= ring,
    transform_parameter= 0.73))


folders_1.append(Path(r"20260128/Scan2_CFG"))

configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= ring,
    transform_parameter= 0.73))



folders_1.append(Path(r"20260129/Scan1"))

configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= ring,
    transform_parameter= 0.73))


folders_1.append(Path(r"20260130/Scan2"))
configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= ring,
    transform_parameter= 0.73))

folders_1.append(Path(r"20260130/Scan3"))

configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= ring,
    transform_parameter= 0.73))


folders_1.append(Path(r"20260202\Scan1_CFG"))

configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 204),
    angle= Angle(12.0, AngleUnit.DEG),
    analysis_zone= ring,
    transform_parameter= 0.74))


folders_1.append(Path(r"20260202\Scan2_CFG"))
configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 204),
    angle= Angle(12.0, AngleUnit.DEG),
    analysis_zone= ring,
    transform_parameter= 0.74))





averagedScanData, resampled_avg, spectrogram, axis = calculating(folders_1, configs_1)
x = [time.value(Prefix.PICO) for time in axis]

fig, (ax1,ax0, ax2, ax3) = plt.subplots(4,1,sharex=True,gridspec_kw={'hspace': 0})
plot_averaged_scan(ax1, averagedScanData, PlotColor.BLUE)
y, trend = resample_scan(resampled_avg, axis).detrend()
plot_averaged_scan(ax0, resampled_avg, PlotColor.BLUE)
ax0.plot(x, trend)
#plot_GaussianFit(ax0, resampled_avg)
#dsa = resample_scan(averagedScanData, axis)
#y = np.asarray([c.value for c in dsa.measured_values], dtype=float)
#ax1.plot(y)

ax2.plot(x, y)
fig.suptitle('Droplets: Ring constant 130-140 (detrend gaussian)', fontsize=12)
#ax.legend(loc="upper left")


plot_Spectrogram(ax3, spectrogram)
ax3.grid(False) 
#ax2b.pcolormesh.cmap('magma')

ax1.text(
    0.02, 0.95, r'\textbf{(a)}',
    transform=ax1.transAxes,
    va='top'
)
ax3.text(
    0.02, 0.95, r'\textbf{(b)}',
    transform=ax2.transAxes,
    va='top',
    color='w'
)

ax3.xaxis.set_major_locator(mpl.ticker.MultipleLocator(100))
ax3.set_xlim((-300,300))
ax1.set_xlabel("")
ax1.tick_params(axis='x', direction='in',labelbottom=False)
ax3.set_ylim((0,125))

ylabels_b = ax3.get_yticklabels()
ylabels_b[-1].set_visible(False)


#ax2b.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=5,presets=xlimits))
#ax2b.yaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
#ax2b.yaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=3,presets=limits))

#plot_nyquist_frequency(ax, scans[0])
fig.savefig(fig_filename,format='pdf')
fig.tight_layout()
plt.show()
