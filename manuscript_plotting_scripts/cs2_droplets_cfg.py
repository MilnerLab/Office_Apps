from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
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
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length


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


fig_filedir = r"C:/Users/camp06/Documents/droplets_manuscript/"
fig_filename = fig_filedir + r"fig2_januarydata.pdf"

folder_path = Path(r"Z://Droplets/20260128/Scan1_CFG")
file_paths = DatFinder(folder_path).find_datafiles()

ring = Range[int](80, 120)

config_1 = IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= ring,
    transform_parameter= 0.73)


ion_data = load_ion_data(file_paths, config_1.delay_center)
save_path = create_save_path_for_calc_ScanFile(folder_path, str(ion_data[0].run_id))
calculated_Scan_1 = run_pipeline(ion_data,config_1, save_path)

folder_path = Path(r"Z://Droplets/20260128/Scan2_CFG")
file_paths = DatFinder(folder_path).find_datafiles()

config_2 = IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= ring,
    transform_parameter= 0.73)


ion_data = load_ion_data(file_paths, config_2.delay_center)
save_path = create_save_path_for_calc_ScanFile(folder_path, str(ion_data[0].run_id))
calculated_Scan_2 = run_pipeline(ion_data,config_2, save_path)



folder_path = Path(r"Z://Droplets/20260129/Scan1")
file_paths = DatFinder(folder_path).find_datafiles()

config_3 = IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= ring,
    transform_parameter= 0.73)


ion_data = load_ion_data(file_paths, config_3.delay_center)
save_path = create_save_path_for_calc_ScanFile(folder_path, str(ion_data[0].run_id))
calculated_Scan_3 = run_pipeline(ion_data,config_3, save_path)


folder_path = Path(r"Z://Droplets/20260130/Scan2")
file_paths = DatFinder(folder_path).find_datafiles()

config_4 = IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= ring,
    transform_parameter= 0.73)


ion_data = load_ion_data(file_paths, config_4.delay_center)
save_path = create_save_path_for_calc_ScanFile(folder_path, str(ion_data[0].run_id))
calculated_Scan_4 = run_pipeline(ion_data,config_4, save_path)

folder_path = Path(r"Z://Droplets/20260130/Scan3")
file_paths = DatFinder(folder_path).find_datafiles()

config_5 = IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= ring,
    transform_parameter= 0.73)


ion_data = load_ion_data(file_paths, config_5.delay_center)
save_path = create_save_path_for_calc_ScanFile(folder_path, str(ion_data[0].run_id))
calculated_Scan_5 = run_pipeline(ion_data,config_5, save_path)

folder_path = Path(r"Z:\Droplets\20260202\Scan1_CFG")
file_paths = DatFinder(folder_path).find_datafiles()

config_6 = IonDataAnalysisConfig(
    delay_center= Length(89.654, Prefix.MILLI),
    center=Point(175, 204),
    angle= Angle(12.0, AngleUnit.DEG),
    analysis_zone= ring,
    transform_parameter= 0.74)


ion_data = load_ion_data(file_paths, config_6.delay_center)
save_path = create_save_path_for_calc_ScanFile(folder_path, str(ion_data[0].run_id))
calculated_Scan_6 = run_pipeline(ion_data,config_6, save_path)

folder_path = Path(r"Z:\Droplets\20260202\Scan2_CFG")
file_paths = DatFinder(folder_path).find_datafiles()
#Scan2 from 20260202 has same config as Scan1
ion_data = load_ion_data(file_paths, config_6.delay_center)
save_path = create_save_path_for_calc_ScanFile(folder_path, str(ion_data[0].run_id))
calculated_Scan_7 = run_pipeline(ion_data,config_6, save_path)

scans = [calculated_Scan_1, calculated_Scan_2, calculated_Scan_3, calculated_Scan_4, calculated_Scan_5,calculated_Scan_6,calculated_Scan_7]


config = StftAnalysisConfig(scans)

resampled_scans = resample_scans(scans, config.axis)

averagedScanData = average_scans(scans)
fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,gridspec_kw={'hspace': 0})
plot_averaged_scan(ax1, averagedScanData, PlotColor.BLUE)
#plot_GaussianFit(ax, averagedScanData)
#fig.suptitle('Droplets', fontsize=12)
#ax.legend(loc="upper left")

spectrogram = StftAnalysis(resampled_scans, config).calculate_averaged_spectrogram()
 


plot_Spectrogram(ax2, spectrogram)
ax2.grid(False) 
#ax2b.pcolormesh.cmap('magma')

ax1.text(
    0.02, 0.95, r'\textbf{(a)}',
    transform=ax1.transAxes,
    va='top'
)
ax2.text(
    0.02, 0.95, r'\textbf{(b)}',
    transform=ax2.transAxes,
    va='top',
    color='w'
)

ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(100))
ax2.set_xlim((-300,300))
ax1.set_xlabel("")
ax1.tick_params(axis='x', direction='in',labelbottom=False)
ax2.set_ylim((0,125))

ylabels_b = ax2.get_yticklabels()
ylabels_b[-1].set_visible(False)


#ax2b.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=5,presets=xlimits))
#ax2b.yaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
#ax2b.yaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=3,presets=limits))

#plot_nyquist_frequency(ax, scans[0])
fig.savefig(fig_filename,format='pdf')
fig.tight_layout()
plt.show()
