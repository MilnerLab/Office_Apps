from ctypes.wintypes import PLONG
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data, load_time_scans
from _domain.models import ScanDataBase
#from _domain.plotting import plot_GaussianFit

from apps.c2t_calculation.domain.pipeline import run_pipeline
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.models import AveragedScansData
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from apps.single_scan.domain.plotting import plot_single_scan
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time


TimeStepSize = 100
PlotTitle = 'OCS'

#Data Files
#20260206 is with GA=26mm, DA=16.3mm, there's a good cross-correlation trace in 20260207 with this exact centrifuge
folder_path_a= Path(r"Z:\Droplets\20260206\Scan7_ScanFiles") #20260206 scan7 is jet data
folder_path_b = Path(r"Z:\Droplets\20260206\Scan6_ScanFiles") #20260206 scan6 is droplet data

#20260205 scans 2 and earlier are DA=16mm, more similar to the good CS2 data
#folder_path_a= Path(r"Z:\Droplets\20260204\Scan2_ScanFiles") #DA = 16mm
#20260205 scans 3 and later are the same DA, so could combine 2026/02/05 data with 2026/02/06 data. 
#folder_path_b= Path(r"Z:\Droplets\20260205\Scan4+5+6+7_ScanFiles") #DA = 16.3mm 

#Combined data
folder_path_a = Path(r"Z:\Droplets\paper_scanfiles\202602_OCS_Droplets_DA16p3")


#20260207 is with GA=0mm, DA=16.6mm, with good cross-correlation
#folder_path_a= Path(r"Z:\Droplets\20260207\Scan4_ScanFiles") #20260207 scan4 is jet data
folder_path_b = Path(r"Z:\Droplets") #20260207 scan5 is droplet data


#Path to save figure in
fig_filedir = r"Z:\Droplets\plots" 
fig_filename = fig_filedir + r"\\MISC_OCS.pdf" #Name the file to save here



'''
DATA LOCATIONS! 
2026/02 OCS data to compare is with three different centrifuges (GA=26mm, DA = 16.0mm, GA=26mm, DA=16.3mm and GA=0mm, DA=16.6mm),
and for each of those we want to plot:
cross-correlation traces with spectrograms,
jet scan with spectrogram (we don't have this for OCS with the fastest centrifuge, but that's ok),
droplet scan without spectrogram.



--------XCORR---------
FASTEST:
GA=26mm, DA = 16.0mm, essentially the CFG used for CS2 experiments:
Z:\Droplets\20260205\XCORR\CFG_2\202602051051AM_.csv
        (Z:\Droplets\20260205\XCORR\ also contains GA and DA only scans)

FAST ACCELERATION SLOW START:
GA=26mm, DA = 16.3mm, used for OCS with the "quick" centrifuge that still has a slower starting frequency than in CS2 experiments
Z:\Droplets\20260207\XCORR\CFG_16p3mm_26mm\20260207905AM_.csv

SLOWEST:
GA=0mm, DA = 16.6mm, used for OCS as the "slowest" centrifuge we can currently make
Z:\Droplets\20260207\XCORR\CFG_16p6mm_0mm\202602071250_.csv


--------JET---------
GA=26, DA = 16.3mm
Z:\Droplets\20260206\Scan7

GA=0mm, DA = 16.6mm
Z:\Droplets\20260207\Scan4

--------DROPLETS---------
GA = 26mm, DA = 16.0mm
Z:\Droplets\20260204\Scan1
Z:\Droplets\20260204\Scan2

Z:\Droplets\20260205\Scan1
Z:\Droplets\20260205\Scan2

GA=26mm, DA = 16.3mm
Z:\Droplets\20260205\Scan3
Z:\Droplets\20260205\Scan4
Z:\Droplets\20260205\Scan5
Z:\Droplets\20260205\Scan6
Z:\Droplets\20260205\Scan7

Z:\Droplets\20260206\Scan5
Z:\Droplets\20260206\Scan6

Z:\Droplets\20260207\Scan1
Z:\Droplets\20260207\Scan2
Z:\Droplets\20260207\Scan3

GA=0mm, DA = 16.6mm
Z:\Droplets\20260207\Scan5
Z:\Droplets\20260207\Scan6


'''



#Average the scan files
file_path_a = DatFinder(folder_path_a).find_scanfiles()
file_path_b = DatFinder(folder_path_b).find_scanfiles()
averagedScanData_a = average_scans(load_time_scans(file_path_a))
averagedScanData_b = average_scans(load_time_scans(file_path_b))


#Update the matplotlib settings
mpl.rcParams.update({
#copied from physrev.mplstyle file
    #"axes.prop_cycle": "(cycler('color', ['5d81b4', 'e09b24', '8eb031', 'eb6235', '8678b2', 'c46e1a', '5c9dc7', 'ffbf00', 'a5609c']) + cycler('ls', ['-', '--', '-.', (0, (1,0.85)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), ':', (4, (10, 3))]))",  
  
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
    "lines.marker": "d",
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
    "text.usetex": False,       #<------------------------------------------------<-----<_<_<_ LATEX 
    #"mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"]

})


#Create the figures and plot things!
fig3, (ax_a,ax_b) = plt.subplots(2,1,sharex=True,gridspec_kw={'hspace': 0})
plot_averaged_scan(ax_a, averagedScanData_a, color=PlotColor.BLUE,ecolor=PlotColor.RED,marker='d')
ax_a.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05)) 
ax_a.grid(which='major',axis='both')

plot_averaged_scan(ax_b,averagedScanData_b,color=PlotColor.BLUE,ecolor=PlotColor.RED,marker='d')

ax_b.xaxis.set_major_locator(mpl.ticker.MultipleLocator(TimeStepSize))
ax_b.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05)) 
ax_b.grid(which='major',axis='both')




#------------------------------------------- Change these to textbf if latex is working
ax_a.text(
    0.02, 0.95, r'$\mathbf{(a)}$',
    transform=ax_a.transAxes,
    va='top'
)
ax_b.text(
    0.02, 0.95, r'$\mathbf{(b)}$', 
    transform=ax_b.transAxes,
    va='top'
)

ax_a.set_xlabel("")
ax_a.tick_params(axis='x', direction='in',labelbottom=False)
fig3.suptitle(PlotTitle)

fig3.savefig(fig_filename,format='pdf')

plt.show()
