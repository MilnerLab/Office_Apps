from pathlib import Path
import numpy as np
from altair import FontWeight
import matplotlib as mpl
from matplotlib import pyplot as plt

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
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
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

mpl.rcParams.update({
  
    # --- Axes ---
    "axes.titlesize": "medium",
    "axes.labelsize": 9,
    "axes.formatter.use_mathtext": True,
    "axes.linewidth": 0.5,
    "axes.grid": True,
    "axes.grid.axis": "both",  # which axis the grid should apply to
    "axes.grid.which": "major",
    "axes.axisbelow" : True,
    "grid.alpha": 0.25,

    # --- Grid lines ---
    "grid.linewidth": 0.5,
    "grid.linestyle": "dashed",
    "grid.color": "xkcd:light gray",

    # --- Lines ---
    "lines.linewidth": 0.5,
    "lines.marker": "o",
    "lines.markersize": 1.5,
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
    "xtick.labelsize": 9,

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
    "figure.figsize": (3.375, 3.6), #1- column fig
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

DROPLETRADIUSMIN = 60

STFTWINDOWSIZE = Time(180,Prefix.PICO)  
EARLIEST_DELAY_PS = -550
LATEST_DELAY_PS = -EARLIEST_DELAY_PS
POSZEROSHIFT = 0 #millimetres :)

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

#---------------------------------------------------------------------------
#Figure 1 - CS2 jet and droplets 2025 data BREAKING THROUGH THE WALLLLL
savefig_folder = r"C:\Users\camp06\OneDrive - UBC\Documents\droplets_manuscript\c2t_plots\\"
savefig1_filename = savefig_folder + r"cs2-breakingwall.png"

#Loading droplet data. GA=26mm, DA = 15.9mm
configs_droplets: list[IonDataAnalysisConfig] = []
folders_droplets: list[Path] = []

folders_droplets.append(Path(r"20251212\Scan4")) #Combination of 20251212 and 20251213. 
configs_droplets.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.79))

folders_droplets.append(Path(r"20251213\Scan1")) #Combination of 20251212 and 20251213. 
configs_droplets.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.79))

folders_droplets.append(Path(r"20251213\Scan2")) #Combination of 20251212 and 20251213. 
configs_droplets.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.79))


folders_droplets.append(Path(r"20251213\Scan3")) #Combination of 20251212 and 20251213. 
configs_droplets.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.79)) 

#Loading jet data
folders_jet: list[Path]=[]
configs_jet: list[IonDataAnalysisConfig]=[]

folders_jet.append(Path(r"20251210\JSS3"))  #20251210 JSS3 is dense throughout the centrifuge
configs_jet.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(195, 197),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](20, 120),
    transform_parameter= 0.79))

folders_jet.append(Path(r"20251210\JSS4"))  #20251210 JSS4 is dense before the centrifuge
configs_jet.append(configs_jet[0]) #same config


fig1,(ax1,ax2) = plt.subplots(2,1,sharex=True,gridspec_kw={'hspace':0})

plottable_scan_drop,plottable_spec_drop = calculating(folders=folders_droplets,configs=configs_droplets)
plottable_scan_jet,plottable_spec_jet = calculating(folders_jet,configs_jet)


plot_averaged_scan(ax1,plottable_scan_drop,PlotColor.BLUE,ecolor=PlotColor.RED,marker='d')
plot_averaged_scan(ax2,plottable_scan_jet,PlotColor.BLUE,ecolor=PlotColor.RED,marker='d')

yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
#axes[0,0].set_xlim([-200,100])
ax1.xaxis.label.set_visible(False)
ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

#Get relevant scan range
xdata_cs2 = ax1.lines[-1].get_xdata()
x0,x1 = ax1.get_xlim()
xmin = np.min(xdata_cs2)
xmax = np.max(xdata_cs2)

#Vertical line
x = [-138,-138]
y = list(ax1.get_ylim())
ax1.plot(x,y,'k--',label=r"Centrifugal wall $\approx 20$ GHz")
ax1.legend(loc='upper right')

ax1.text(
    0.02, 0.95, r'\textbf{(a)}',
    transform=ax1.transAxes,
    va='top'
)
ax2.text(
    0.02, 0.95, r'\textbf{(b)}',
    transform=ax2.transAxes,
    va='top',
)

#------------------------------------------------------------------------------------------------
#Figure 2 - CS2, OCS (fast cfg), OCS (slow cfg) comparison. Does OCS break the thermalization model?
savefig2_filename = savefig_folder + r"cs2-ocs-comparison.png"

#OCS with slow CFG: GA = 0mm, DA = 16.6mm
configs_ocs_slow: list[IonDataAnalysisConfig] = []
folders_ocs_slow: list[Path] = [] 

folders_ocs_slow.append(Path(r"20260210\Scan4"))
configs_ocs_slow.append(IonDataAnalysisConfig(
    delay_center= Length(92.654-POSZEROSHIFT, Prefix.MILLI),
    center=Point(175, 205),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.75))

#OCS with fast CFG: GA = 26mm, DA = 15.2mm
configs_ocs_fast: list[IonDataAnalysisConfig] = []
folders_ocs_fast: list[Path] = [] 
folders_ocs_fast.append(Path(r"20260222\Scan3")) 
configs_ocs_fast.append(IonDataAnalysisConfig(
    delay_center= Length(94.5-POSZEROSHIFT, Prefix.MILLI),
    center=Point(174, 206),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.74))
folders_ocs_fast.append(Path(r"20260223\Scan1")) 
configs_ocs_fast.append(configs_ocs_fast[0])

ocs_slow_scan,ocs_slow_spec = calculating(folders_ocs_slow,configs_ocs_slow)
ocs_fast_scan,ocs_fast_spec = calculating(folders_ocs_fast,configs_ocs_fast)

#Fixing the artifcact in the ocs fast cfg spectrogram by filtering out the early and late times in the scan
scans_paths = DatFinder(folders_ocs_fast).find_datafiles() #Change this if you want a specific path rather than the Droplets folder
raw_datas = load_ion_data(scans_paths)
calculated_scans = run_pipeline(raw_datas, configs_ocs_fast)

for i in [0,1]:
    times = np.array([t.value(Prefix.PICO) for t in calculated_scans[i].delays])
    mask = (times > -250) & (times < 250)
    inds = np.where(mask)
    calculated_scans[i].delays = calculated_scans[i].delays[inds]
    calculated_scans[i].measured_values = calculated_scans[i].measured_values(inds)

config = StftAnalysisConfig(calculated_scans, STFTWINDOWSIZE)
resampled_scans = resample_scans(calculated_scans, config.axis)
filtered_ocs_fast_spec = StftAnalysis(resampled_scans, config).calculate_averaged_spectrogram()





axes=[]
fig2, axes = plt.subplots(3,2,sharex=True,gridspec_kw={"hspace":0})

plot_Spectrogram(axes[0,0],plottable_spec_drop,shading='auto')
plot_Spectrogram(axes[2,0],ocs_slow_spec,shading='auto')
plot_Spectrogram(axes[1,0],filtered_ocs_fast_spec,shading='auto')



axes[0,0].set_ylim([0,100])
axes[2,0].set_ylim([0,80])
axes[1,0].set_ylim([0,60])
axes[2,0].set_xlim([xmin,xmax])
axes[1,0].set_xlim([xmin,xmax])

"""
To add more black background in spectrograms that extend beyond data range
axes[0,0].axvspan(x0, xmin, color='black', alpha=1.0)
axes[0,0].axvspan(xmax,x1,color='black',alpha=1.0) """

axes[0,0].xaxis.label.set_visible(False)
axes[0,0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
axes[2,0].xaxis.label.set_visible(False)
axes[2,0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
axes[0,1].text(
    -380,0.5,r'\noindent Theoretical CS2 spectrogram'
)
axes[1,1].text(
    -380,0.5,r'\noindent Theoretical OCS spectrogram\\ with faster centrifuge'
)
axes[2,1].text(
    -380,0.5,r'\noindent Theoretical OCS spectrogram\\ with slower centrifuge'
)

#Cleaning up ticks and their labels for flush x axes
letters = ['a','b','c']
for i in [0,1,2]:
    yticks = axes[i,0].yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    axes[i,0].text(
        0.02,0.95,r'\textbf{('+letters[i]+')}',
        transform=axes[i,0].transAxes,
        va='top',
        color = 'w'
    )
    axes[i,0].yaxis.label.set_visible(False)
    axes[i,1].xaxis.set_visible(False)
    axes[i,1].yaxis.set_visible(False)
    

fig2.savefig(savefig2_filename,format='png')
fig2.supylabel(r'\langle \cos^2\theta_{2D} Oscillation Frequency (GHz)')
plt.show()
""" xdata_drop = axes[0,0].lines[-1].get_xdata()
xdata_jet = axes[1,0].lines[-1].get_xdata()

xmin = min(np.min(xdata_drop),np.min(xdata_jet))
xmax = max(np.max(xdata_drop),np.min(xdata_jet)) """


