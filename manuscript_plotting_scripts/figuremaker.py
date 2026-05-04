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
from base_core.lab_specifics.base_models import C2TScanData, IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time

mpl.rcParams.update({
    
    # --- Axes ---
    "axes.titlesize": "medium",
    "axes.labelsize": 6,
    "axes.formatter.use_mathtext": True,
    "axes.linewidth": 0.5,
    "axes.grid": True,
    "axes.grid.axis": "both",  # which axis the grid should apply to
    "axes.grid.which": "major",
    "axes.axisbelow" : True,
    "grid.alpha": 1.0,

    # --- Grid lines ---
    "grid.linewidth": 0.3,
    "grid.linestyle": "solid",
    "grid.color": "grey",

    # --- Lines ---
    "lines.linewidth": 0.5,
    "lines.marker": "o",
    "lines.markersize": 1.0,
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
    "xtick.labelsize": 6,

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
    "ytick.major.pad": 2.0,
    #"ytick.minor.pad": 5.0,
    "ytick.labelsize": 6,
    
    
    # --- Legend ---
    "legend.frameon": True,
    "legend.fontsize": 6,
    "legend.handlelength": 1.375,
    "legend.labelspacing": 0.4,
    "legend.columnspacing": 1,
    "legend.facecolor": "white",
    "legend.edgecolor": "white",
    "legend.framealpha": 1,
    "legend.title_fontsize": 6,
 
    # --- Figure size ---
    "figure.figsize": (3.375, 3.6), #1- column fig
    #"figure.figsize": (3.375, 3),
    #"figure.figsize": (6.75, 3.6), #approx. 2- column fig
    "figure.subplot.left": 0.125,
    "figure.subplot.bottom": 0.175,
    "figure.subplot.top": 0.95,
    "figure.subplot.right": 0.95,

    # --- Fonts (computer modern) ---
    "font.size": 6,
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

def calculating_comp(folders_comp: list[list[Path]], configs_comp: list[list[IonDataAnalysisConfig]],cut_start = 0, cut_end = 0) -> tuple[list[AveragedScansData],list[AggregateSpectrogram]]:
    n = len(folders_comp)
    #calculated_scans = [[] for i in range(n)]
    calculated_scans = []
    averagedScanData = []
    spectrograms = []
    for i in range(n):
        scans_paths = DatFinder(folders_comp[i]).find_datafiles() #Change this if you want a specific path rather than the Droplets folder
        raw_datas = load_ion_data(scans_paths)
        calculated_scan = run_pipeline(raw_datas, configs_comp[i])
        calculated_scans.append(calculated_scan)
    
    match_scans(calculated_scans,cut_start,cut_end)
    
    for i in range(n):
        averagedScanData.append(average_scans(calculated_scans[i]))
        config = StftAnalysisConfig(calculated_scans[i], STFTWINDOWSIZE)
        resampled_scans = resample_scans(calculated_scans[i], config.axis)
        spectrogram = StftAnalysis(resampled_scans, config).calculate_averaged_spectrogram()
        spectrograms.append(spectrogram)
    return (averagedScanData,spectrograms)
        
        
def match_scans(scan_collection: list[list[C2TScanData]],cut_start,cut_end) -> None:
    mins = []
    maxes = []
    n = len(scan_collection)
    delays = [[] for i in range(n)]
    #delays = []
    if not cut_start and not cut_end:
        for i in range(n):
            scan = scan_collection[i]
            min_count = np.inf
            max_count = -np.inf
            for c2tdata in scan:
                times = np.array([t.value(Prefix.PICO) for t in c2tdata.delays])
                delays[i].append(times)
                min_test = np.min(times)
                max_test = np.max(times)
                if min_test < min_count: 
                    min_count = min_test
                if max_test > max_count:
                    max_count = max_test
            mins.append(min_count)
            maxes.append(max_count)        
        xmin = np.max(np.array(mins))
        xmax = np.min(np.array(maxes))
    else:
        for i in range(n):
            scan = scan_collection[i]
            for c2tdata in scan:
                times = np.array([t.value(Prefix.PICO) for t in c2tdata.delays])
                delays[i].append(times)
                
        xmin = cut_start
        xmax = cut_end
    for i in range(n):
        m = len(scan_collection[i])
        for j in range(m):
            mask = (delays[i][j] > xmin) & (delays[i][j] < xmax)
            inds = np.where(mask)
            start = inds[0][0]
            end = inds[-1][-1] 
            scan_collection[i][j].cut(start=start,end=end)
    
#--------------------------------------------------------------------------------------------------
#Loading data for figures    

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

plottable_scan_drop,plottable_spec_drop = calculating(folders=folders_droplets,configs=configs_droplets)
plottable_scan_jet,plottable_spec_jet = calculating(folders_jet,configs_jet)

#For figure 2
folders_comp = [folders_droplets,folders_ocs_fast,folders_ocs_slow]
configs_comp = [configs_droplets,configs_ocs_fast,configs_ocs_slow]

#---------------------------------------------------------------------------

#Folder to save figures
savefig_folder = r"C:\Users\camp06\OneDrive - UBC\Documents\droplets_manuscript\c2t_plots\\"

#-----------------------------------------------------------------------------------
#Figure 1 - CS2 jet and droplets 2025 data BREAKING THROUGH THE WALLLLL
"""
savefig1_filename = savefig_folder + r"cs2-breakingwall.png"
fig1,(ax1,ax2) = plt.subplots(2,1,sharex=True,gridspec_kw={'hspace':0})



#Truncating so that the datasets have same delay range
times = np.array([t.value(Prefix.PICO) for t in plottable_scan_drop.delays])
mask = (times > -320)
inds = np.where(mask)
start = inds[0][0]
plottable_scan_drop.cut(start=start)

times = np.array([t.value(Prefix.PICO) for t in plottable_scan_jet.delays])
mask = (times < 200)
inds = np.where(mask)
end = inds[-1][-1]
plottable_scan_jet.cut(start=0,end=end)

#add a markersize option or a separate rcparams for plot bot and other scripts
plot_averaged_scan(ax1,plottable_scan_jet,PlotColor.BLUE,ecolor=PlotColor.RED,marker='d') 
plot_averaged_scan(ax2,plottable_scan_drop,PlotColor.BLUE,ecolor=PlotColor.RED,marker='d')

#yticks = ax2.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)
#axes[0,0].set_xlim([-200,100])
ax1.xaxis.label.set_visible(False)
ax1.tick_params(top=True, labeltop=False)
#ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax1.yaxis.label.set_visible(False)
ax2.yaxis.label.set_visible(False)
ax1.grid(True) #shouldn't have to do this. Remove the code in plotting classes that override rcparams.
ax2.grid(True)
ax1.lines[-1].set_markersize(0.75)

#Get relevant scan range
xdata_cs2 = np.array(ax2.lines[-1].get_xdata())
ydata_cs2 = np.array(ax2.lines[-1].get_ydata())
x0,x1 = ax2.get_xlim()

#ax2.set_xlim([-320,x1])
#ax1.set_xlim([-320,x1])

#Vertical line
x = [-138,-138]
y = list(ax2.get_ylim())
ax2.plot(x,y,'k--',label=r"Centrifugal wall $\approx 20$ GHz",linewidth=1)
ax2.legend(loc='upper right')

ax1.text(
    0.04, 0.95, r'\textbf{(a)}',
    transform=ax1.transAxes,
    va='top'
)
ax2.text(
    0.04, 0.95, r'\textbf{(b)}',
    transform=ax2.transAxes,
    va='top',
)

fig1.supylabel(r'$\langle \cos^2\theta_{2D}\rangle$')
fig1.savefig(savefig1_filename,format='png',dpi=300) """

#------------------------------------------------------------------------------------------------
#Figure 2 - CS2, OCS (fast cfg), OCS (slow cfg) comparison. Does OCS break the thermalization model?
savefig2_filename = savefig_folder + r"cs2-ocs-comparison_v1.png"



#c2t, spec = calculating_comp(folders_comp,configs_comp,cut_start=-300,cut_end=200) #used for fig 2 v3
c2t, spec = calculating_comp(folders_comp,configs_comp) #cut range is to fix the ocs fast cfg spectrogram artifact
#ocs_slow_scan,ocs_slow_spec = calculating(folders_ocs_slow,configs_ocs_slow)
#ocs_fast_scan,ocs_fast_spec = calculating(folders_ocs_fast,configs_ocs_fast)

#Fixing the artifact in the ocs fast cfg spectrogram by filtering out the early and late times in the scan
#The artifact may be coming from an incorrect combination of 2 scans with different sampling rate 
scans_paths = DatFinder(folders_ocs_fast).find_datafiles() #Change this if you want a specific path rather than the Droplets folder
raw_datas = load_ion_data(scans_paths)
calculated_scans = run_pipeline(raw_datas, configs_ocs_fast) 
for i in [0,1]:
    times = np.array([t.value(Prefix.PICO) for t in calculated_scans[i].delays])
    mask = (times > -400) & (times < 500)
    inds = np.where(mask)
    #inds_true = np.where(inds)
    start = inds[0][0]
    end = inds[-1][-1]
    calculated_scans[i].cut(start,end)
    #calculated_scans[i].delays = np.array(calculated_scans[i].delays)[inds]
    #calculated_scans[i].measured_values = calculated_scans[i].measured_values(inds)

config = StftAnalysisConfig(calculated_scans,STFTWINDOWSIZE)
resampled_scans = resample_scans(calculated_scans, config.axis)
filtered_ocs_fast_spec = StftAnalysis(resampled_scans, config).calculate_averaged_spectrogram() 


delays = np.array([t.value(Prefix.PICO) for t in filtered_ocs_fast_spec.delay])
xmin = np.min(delays)
xmax = np.max(np.array([t.value(Prefix.PICO) for t in spec[0].delay]))

axes=[]

fig2, axes = plt.subplots(3,2,sharex=True,gridspec_kw={"hspace":0,"wspace":0.01})

plot_Spectrogram(axes[0,0],spec[0],shading='auto')
plot_Spectrogram(axes[1,0],filtered_ocs_fast_spec,shading='auto')
plot_Spectrogram(axes[2,0],spec[2],shading='auto')
 

#Horizontal lines
x = [xmin,xmax]
y_cs2 = [20,20]
y_ocs = [35,35]

axes[0,0].plot(x,y_cs2,'r--',label = r'CS\textsubscript{2} Centrifugal Wall',lw=1)
axes[1,0].plot(x,y_ocs,'r--',label = r'OCS Centrifugal Wall',lw=1)
axes[2,0].plot(x,y_ocs,'r--',label = r'OCS Centrifugal Wall',lw=1)

axes[0,0].set_ylim([0,100])
axes[2,0].set_ylim([0,100])
axes[1,0].set_ylim([0,100])
axes[2,0].set_xlim([xmin,xmax])
axes[1,0].set_xlim([xmin,xmax])

axes[2,0].tick_params(axis='x',which='both',direction='out') 
letters = ['a','b','c']
for i in [0,1,2]:
    legend = axes[i,0].legend(loc='upper left')
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("black")
    for text in legend.get_texts():
        text.set_color("white")
    
    axes[i,0].xaxis.label.set_visible(False)
    axes[i,0].yaxis.label.set_visible(False)
    axes[i,0].tick_params(axis='y',which='both',direction='out') 
    yticks = axes[i,0].yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    
    axes[i,0].text(
        0.04,0.5,f'({letters[i]})',
        transform=axes[i,0].transAxes,
        ha = 'left',
        va='top',
        fontweight='bold',
        color = 'w',
        zorder=100,
        clip_on=False
    )
    
    axes[i,1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axes[i,1].tick_params(axis='y', which='both', left=False, labelleft=False)
    axes[i,1].grid(False)

    
    
    

#To add more black background in spectrograms that extend beyond data range
#axes[0,0].axvspan(x0, xmin, color='black', alpha=1.0)
#axes[0,0].axvspan(xmax,x1,color='black',alpha=1.0) 


#axes[0,0].tick_params(axis='y',which='both',labelleft=False)
#axes[1,0].tick_params(axis='y',which='both',labelleft=False)

axes[0,1].set_title("Theory comparison")
fig2.supxlabel(r'Probe Delay (ps)',y=0.08)
fig2.supylabel(r"$\langle\cos^2\theta_{2D}\rangle$ Oscillation Frequency (GHz)") 
fig2.savefig(savefig2_filename,format='png',dpi=300)

#------------Version 2 of fig 2
""" savefig2_filename = savefig_folder + r"cs2-ocs-comparison_v2.png"

c2t, spec = calculating_comp(folders_comp,configs_comp,cut_start=-300,cut_end=200)

fig2, axes = plt.subplots(
    3,2,sharex=True,figsize=(6.75, 3.6), #2 column width
    gridspec_kw={"hspace":0,"wspace":0.1})


plot_averaged_scan(axes[0,0],c2t[0],PlotColor.BLACK,ecolor=PlotColor.GRAY,marker='d') 
plot_averaged_scan(axes[1,0],c2t[1],PlotColor.BLACK,ecolor=PlotColor.GRAY,marker='d')
plot_averaged_scan(axes[2,0],c2t[2],PlotColor.BLACK,ecolor=PlotColor.GRAY,marker='d')

axes_twins = []
axes_twins.append(axes[0,0].twinx())
axes_twins.append(axes[1,0].twinx())
axes_twins.append(axes[2,0].twinx())

x1 = [-200,100]
y1 = [5,85]
x2 = [-190,-40]
y2 = [0,40]
x3= [-200,100]
y3 = [10,65]

axes_twins[0].plot(x1,y1,'r-')
axes_twins[1].plot(x2,y2,'r-')
axes_twins[2].plot(x3,y3,'r-')


#ax_twin.yaxis.label.set_color('red')
#ax_twin.tick_params(axis='y',color='red')
#Cleaning up ticks and their labels for flush x axes
letters = ['a','b','c']
for i in range(3):
    axes_twins[i].set_ylim([0,80])
    axes_twins[i].tick_params(axis='y',colors='red')
    axes_twins[i].grid(False)
    ticks = axes_twins[i].yaxis.get_major_ticks()
    ticks[-1].label2.set_visible(False) #label 2 refers to the right y axis in mpl while label 1 refers to left y axies (same holds for the xaxis case)
    axes[i,1].grid(False)
    axes_twins[i].grid(False)
    axes[i,0].yaxis.label.set_visible(False)
    axes[i,0].xaxis.label.set_visible(False)
    axes[i,1].tick_params(axis='y',left=False,right=True,colors='red',labelleft=False)
    
    axes[i,0].set_ylim(bottom=0.495)
    axes[i,0].text(
        0.03,0.95,f'({letters[i]})',
        transform=axes[i,0].transAxes,
        ha = 'left',
        va='top',
        fontweight='bold',
        zorder=100,
        clip_on=False
    )
    

    
#yticks = axes[1,0].yaxis.get_major_ticks()
#yticks[-2].label1.set_visible(False)
    
axes[0,1].set_title("Theory comparison")
fig2.supxlabel(r'Probe Delay (ps)')
fig2.supylabel(r"$\langle\cos^2\theta_{2D}\rangle$",x=0.005)
fig2.text(
    0.995, 0.5,
    r"$\langle\cos^2\theta_{2D}\rangle$ Oscillation Frequency (GHz)",
    rotation=-90,
    va='center',
    ha='right'
)
#fig2.tight_layout()
plt.ion()
plt.show()

#stretch = float(input("stretch: ")) #0.05
#w = float(input("width: ")) #0.2
params = fig2.subplotpars
default = dict(
    left = params.left,
    right = params.right,
    top = params.top,
    bottom = params.bottom,
    hspace = params.hspace,
    wspace = 0
)
while(True):
    cmd = input("Enter 'shift, stretch, width' or 'save': ")
    
    if cmd.lower() == "save":
        fig2.savefig(savefig2_filename,format='png',dpi=300)
        break
    fig2.subplots_adjust(**default)
    
    shift, stretch, width = map(float,cmd.split(",")) 
    fig2.subplots_adjust(left=params.left - shift - stretch, right=params.right - shift + stretch,wspace = width)
    fig2.canvas.draw_idle()
    plt.pause(0.01) """


#Version 3 of figure 2

"""
savefig2_filename = savefig_folder + r"cs2-ocs-comparison_v3.png"
axes = []
fig2, axes = plt.subplots(
    3,1,gridspec_kw={"hspace":0},sharex=True
)
#fig2.supxlabel(r'Probe Delay (ps)',y=0.2)
#fig2.supylabel(r"$\langle\cos^2\theta_{2D}\rangle$",x=0)

plot_averaged_scan(axes[0],c2t[0],PlotColor.BLACK,ecolor=PlotColor.GRAY,marker='d') 
plot_averaged_scan(axes[1],c2t[1],PlotColor.BLACK,ecolor=PlotColor.GRAY,marker='d',label='Experiment')
plot_averaged_scan(axes[2],c2t[2],PlotColor.BLACK,ecolor=PlotColor.GRAY,marker='d')

axes_twins = []
axes_twins.append(axes[0].twinx())
axes_twins.append(axes[1].twinx())
axes_twins.append(axes[2].twinx())

x1 = [-200,100]
y1 = [5,85]
x2 = [-190,-40]
y2 = [0,40]
x3= [-200,100]
y3 = [10,65]

axes[0].xaxis.label.set_visible(False)

axes_twins[0].plot(x1,y1,'r-')
axes_twins[1].plot(x2,y2,'r-')
axes_twins[2].plot(x3,y3,'r-')

letters = ['a','b','c']
for i in range(3):
    axes[i].set_ylim(bottom=0.495)
    axes[i].yaxis.label.set_visible(False)
    
    ax_right = axes_twins[i]
    ax_right.set_ylim([-5,90])
    
    fig2.canvas.draw()
    #labels = ax_right.get_yticklabels()
    #labels[-1].set_visible(False)
    
    ax_right_ticks = ax_right.yaxis.get_major_ticks()
    ax_right_ticks[-1].label2.set_visible(False)
    
    ax_right.tick_params(axis='y',colors='red')
    #ax_right.yaxis.tick_right() #THIS OVERWRITES the tick removal above??
    ax_right.grid(False)
    
    axes[i].text(
        0.02,0.95,f'({letters[i]})',
        transform=axes[i].transAxes,
        ha = 'left',
        va='top',
        fontweight='bold',
        zorder=100,
        clip_on=False
    )

axes[1].legend(loc = 'lower right')

fig2.text(
    0.01, 0.5,
    r"$\langle \cos^2 \theta_{\mathrm{2D}} \rangle$",
    rotation=90,
    va='center',
    #fontsize = 8.0
)

fig2.text(
    0.98, 0.5,
    r"$\langle\cos^2\theta_{2D}\rangle$ Oscillation Frequency (GHz)",
    rotation=-90,
    va='center',
    ha='right',
    #fontsize = 8.0
)
fig2.subplots_adjust(left=0.12,right=0.88) #left and right values denote the margin sizes"""

#fig2.savefig(savefig2_filename,format='png',dpi=300)





