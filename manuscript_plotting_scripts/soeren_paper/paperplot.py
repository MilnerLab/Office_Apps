print('Code start!')
from pathlib import Path
from turtle import color
from altair import FontWeight
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from _data_io.dat_saver import create_save_path_for_calc_ScanFile
from _domain.plotting import plot_GaussianFit
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from apps.single_scan.domain.plotting import plot_single_scan
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.models import AggregateSpectrogram
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency, plot_scan_and_spectrogram
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from base_core.lab_specifics.averaging.models import AveragedScansData
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time

DROPLETRADIUSMIN = 40

STFTWINDOWSIZE = Time(180,Prefix.PICO)  
EARLIEST_DELAY_PS = -250
LATEST_DELAY_PS = -EARLIEST_DELAY_PS
POSZEROSHIFT = 0 #millimetres :)

USEFONTSIZE = 16

#FUNCTION TO GENERATE THE PLOTTABLE DATA
def calculating(folders: list[Path], configs: list[IonDataAnalysisConfig]) -> tuple[AveragedScansData, AggregateSpectrogram]:
    
    
    scans_paths = DatFinder(folders).find_datafiles() #Change this if you want a specific path rather than the Droplets folder
    raw_datas = load_ion_data(scans_paths)
    calculated_scans = run_pipeline(raw_datas, configs)
    averagedScanData = average_scans(calculated_scans)
    config = StftAnalysisConfig(calculated_scans, STFTWINDOWSIZE)
    resampled_scans = resample_scans(calculated_scans, config.axis)

    spectrogram = StftAnalysis(resampled_scans, config).calculate_averaged_spectrogram()
    
    return (averagedScanData, spectrogram)
#--------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#Update the matplotlib settings
mpl.rcParams.update({
#copied from physrev.mplstyle file
    #"axes.prop_cycle": "(cycler('color', ['5d81b4', 'e09b24', '8eb031', 'eb6235', '8678b2', 'c46e1a', '5c9dc7', 'ffbf00', 'a5609c']) + cycler('ls', ['-', '--', '-.', (0, (1,0.85)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), ':', (4, (10, 3))]))",  
  
    # --- Axes ---
    "axes.titlesize": "medium",
    "axes.labelsize": USEFONTSIZE,
    "axes.formatter.use_mathtext": True,

    # --- Grid lines ---
    "grid.linewidth": 0.3,
    "grid.linestyle": "solid",
    "grid.color": "grey",
    "grid.alpha": 1,

    # --- Lines ---
    "lines.linewidth": 1,
    "lines.marker": "d",
    "lines.markersize": 0.5,
    "hatch.linewidth": 0.25,
    "patch.antialiased": True,
    
    #---Errorbars---
    "errorbar.capsize": 1,

    # --- Ticks (X) ---
    #"xtick.top": True,
    "xtick.bottom": True,
    "xtick.major.size": 3.0,
    "xtick.minor.size": 1.5,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "xtick.direction": "in",
    "xtick.minor.visible": False,
    #"xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.bottom": True,
    "xtick.major.pad": 5.0,
    "xtick.minor.pad": 5.0,
    "xtick.labelsize": USEFONTSIZE,

    # --- Ticks (Y) ---
    "ytick.left": True,
    #"ytick.right": True,
    "ytick.major.size": 3.0,
    #"ytick.minor.size": 1.5,
    "ytick.major.width": 0.5,
    #"ytick.minor.width": 0.5,
    "ytick.direction": "in",
    #"ytick.minor.visible": True,
    "ytick.major.left": True,
    #"ytick.major.right": True,
    #"ytick.minor.left": True,
    "ytick.major.pad": 5.0,
    #"ytick.minor.pad": 5.0,
    "ytick.labelsize": USEFONTSIZE,
    
    
    # --- Legend ---
    "legend.handlelength": 0.0,
    "legend.handletextpad": 0.0,
    "legend.markerscale": 0.0,
    "legend.frameon": True,
    "legend.fontsize": 12,
    "legend.framealpha": 1,
    "legend.title_fontsize": 12,
    "legend.loc": "upper left", #location is loc
 
    # --- Figure size ---
    "figure.figsize": (3.375, 3.6), #1- column fig
    #"figure.figsize": (6.75, 6.75), #approx. 2- column fig

    # --- Fonts (computer modern) ---
    "text.usetex": False,       #<------------------------------------------------<-----<_<_<_ LATEX 
    #"mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"],

})

#Path to save figure in
fig_filedir = r"/home/soeren/Documents" 
fig_filename = fig_filedir + r"/soeren_paper.png" #Name the file to save here

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#------------------------ NO Dimer
PlotTitle = r"STFT with 180 ps blackman window. Same ``fast'' centrifuge for each scan." "\n" "20260417 Scan6 and 20260417 Scan7/ 20260420 Scan1" #GA = 0mm 
#JET EXPERIMENT - OCS
configs_1: list[IonDataAnalysisConfig] = []
folders_1: list[Path] = []

folders_1.append(Path(r"20260417\Scan6"))  
configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
    center=Point(209, 180),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](40, 110),
    transform_parameter= 0.78))

#DROPLETS
configs_2: list[IonDataAnalysisConfig] = []
folders_2: list[Path] = []
folders_2.append(Path(r"20260420\Scan1"))  #NO-Dimer in droplets | second day
configs_2.append(IonDataAnalysisConfig(
    delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
    center=Point(105, 107),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](30, 60),
    transform_parameter= 0.78))
folders_2.append(Path(r"20260417\Scan7"))  #NO-Dimer in droplets | first day
configs_2.append(IonDataAnalysisConfig(
    delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
    center=Point(105, 107),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](30, 60),
    transform_parameter= 0.78))


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#------------------------ DIB 

#------------------------ FAST EXPERIMENT BELOW HERE
#JET EXPERIMENT
'''configs_3: list[IonDataAnalysisConfig] = []
folders_3: list[Path] = []

folders_3.append(Path(r"20260328\Scan1"))  
configs_3.append(IonDataAnalysisConfig(
    delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
    center=Point(210, 194),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](40, 120),
    transform_parameter= 0.77))

#DROPLETS
configs_4: list[IonDataAnalysisConfig] = []
folders_4: list[Path] = []
folders_4.append(Path(r"20260401\Scan3"))  #DIB in droplets
configs_4.append(IonDataAnalysisConfig(
    delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
    center=Point(186, 205),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 90),
    transform_parameter= 0.81))'''

#------------------------ SLOW EXPERIMENT BELOW HERE
PlotTitle = r"STFT with 180 ps blackman window. Same ``slow'' centrifuge for each scan." "\n" "20260402 Scan1 and 20260402 Scan2" #GA = 0mm

#JET EXPERIMENT
configs_3: list[IonDataAnalysisConfig] = []
folders_3: list[Path] = []

folders_3.append(Path(r"20260402\Scan1"))  
configs_3.append(IonDataAnalysisConfig(
    delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
    center=Point(210, 194),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](40, 120),
    transform_parameter= 0.77))

#DROPLETS
configs_4: list[IonDataAnalysisConfig] = []
folders_4: list[Path] = []
folders_4.append(Path(r"20260402\Scan2"))  #DIB in droplets
configs_4.append(IonDataAnalysisConfig(
    delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
    center=Point(186, 205),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 90),
    transform_parameter= 0.81))


#Pipeline 
plottable_scan_1, plottable_spectrogram_1 = calculating(folders_1, configs_1)
plottable_scan_2, plottable_spectrogram_2 = calculating(folders_2, configs_2)
plottable_scan_3, plottable_spectrogram_3 = calculating(folders_3, configs_3)
plottable_scan_4, plottable_spectrogram_4 = calculating(folders_4, configs_4)

text_only_handle = Line2D([], [], linestyle="none", marker=None)
#Main figure

mainfig, (axs) = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(6.75, 3.5),
            sharex='col'
        )

axs[0,0].xaxis.set_major_locator(MultipleLocator(150))
axs[0,0].set_xlim(-400, 400)
axs[0,1].xaxis.set_major_locator(MultipleLocator(150))
axs[0,1].set_xlim(-370, 310)
axs[0,0].yaxis.set_major_locator(MultipleLocator(0.05))
axs[0,1].yaxis.set_major_locator(MultipleLocator(0.05))
axs[1,0].yaxis.set_major_locator(MultipleLocator(0.05))
axs[1,1].yaxis.set_major_locator(MultipleLocator(0.05))

# -------------------------------------------------------------------------
#Plot NO-Dimer
a = axs[0,0]
a.grid(True, axis='x')

ax_freq, mesh = plot_scan_and_spectrogram(
    a,
    plottable_scan_1,
    plottable_spectrogram_1,
    scan_color=PlotColor.WHITE,
    scan_marker="d",
    shading="auto",
)

a.legend(handles=[text_only_handle], labels=['A1'])

ax_freq.grid(True)
ax_freq.set_ylim(-2, 105)
ax_freq.yaxis.set_major_locator(MultipleLocator(25))

a = axs[0,1]
plot_averaged_scan(a, plottable_scan_2, PlotColor.BLACK,ecolor=PlotColor.RED, marker='d')
a.legend(handles=[text_only_handle], labels=['A2'])
a.grid(True, axis='x')


# -------------------------------------------------------------------------
#Plot DIB
a = axs[1,0]
a.grid(True, axis='x')

ax_freq, mesh = plot_scan_and_spectrogram(
    a,
    plottable_scan_3,
    plottable_spectrogram_3,
    scan_color=PlotColor.WHITE,
    scan_marker="d",
    shading="auto",
)

a.legend(handles=[text_only_handle], labels=['B1'])

ax_freq.grid(True)
ax_freq.set_ylim(-2, 105)
ax_freq.yaxis.set_major_locator(MultipleLocator(25))


a = axs[1,1]
plottable_scan_4.cut(1, 114)
plot_averaged_scan(a, plottable_scan_4, PlotColor.BLACK,ecolor=PlotColor.RED, marker='d')
a.legend(handles=[text_only_handle], labels=['B2'])
a.grid(True, axis='x')


# nachdem alle Plots erzeugt wurden

# Alle normalen y-labels entfernen
for ax in axs.flat:
    ax.set_ylabel("")
    ax.set_xlabel("")

axs[0,0].set_ylabel(r"$\langle \cos^2 \theta_\mathrm{2D} \rangle$")
axs[1,0].set_ylabel(r"$\langle \cos^2 \theta_\mathrm{2D} \rangle$")
axs[1,0].set_xlabel("Probe Delay (ps)")
axs[1,1].set_xlabel("Probe Delay (ps)")


# Gemeinsames Frequency-label zwischen linker und rechter Spalte
mainfig.text(
    0.53, 0.52,
    "Oscillation Frequency (GHz)",
    rotation=90,
    va="center",
    ha="center",
    fontsize=14,
)

mainfig.subplots_adjust(
    left=0.13,    # linker Rand
    right=0.96,   # rechter Rand
    bottom=0.18,  # unterer Rand
    top=0.95,     # oberer Rand
    wspace=0.55,  # Abstand zwischen Spalten
    hspace=0.15,  # Abstand zwischen Reihen
)

mainfig.savefig(fig_filename,format='png', dpi=300)
plt.show()
print('Done!')