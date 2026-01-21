from ctypes.wintypes import PLONG
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_time_scans
from _domain.models import ScanDataBase
#from _domain.plotting import plot_GaussianFit
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.models import AveragedScansData
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Time

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
    "lines.marker": "o",
    "lines.markersize": 1,
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

#Fig3a plot of the CS2 in Jet asympytotic cos^2\theta behaviour
#3a Data includes 20260112 Scan4_Scanfiles and Scan5_Scanfiles

#stylefile = r"C:\Users\camp06\Documents\droplets_manuscript\physrev.mplstyle"
#plt.style.use(stylefile)
#folder_path_a = Path(r"C:/Users/camp06/Documents/droplets_manuscript/20251128 120psi jet - Scan4_ScanFiles(figure3a)/Scan4_ScanFiles")
#merge the following paths
folder_path_a = Path(r"Y:\Droplets\20260112\Scan4_ScanFiles")
#folder_path_a2 = Path(r"Y:\Droplets\20260112\Scan5_ScanFiles")

#just copying the same data as 3(a) for now
folder_path_b_cfg = Path(r"C:\Users\camp06\Documents\droplets_manuscript\fig3b_data\cfg")
folder_path_b_hor = Path(r"Y:\Droplets\20260119\Scan1_ScanFiles")
#folder_path_a = Path(r"Y:\Droplets\20251128") #should merge scans 2-5 to have consistent jet pressure of 120psi 
#folder_path_b = Path(r"Y:\Droplets\20251215")
fig_filedir = r"C:/Users/camp06/Documents/droplets_manuscript/"
fig_filename = fig_filedir + r"figure3.pdf"
file_path_a = DatFinder(folder_path_a).find_scanfiles()
file_path_b_cfg = DatFinder(folder_path_b_cfg).find_scanfiles()
file_path_b_hor = DatFinder(folder_path_b_hor).find_scanfiles()

averagedScanData_a = average_scans(load_time_scans(file_path_a))
averagedScanData_b_cfg = average_scans(load_time_scans(file_path_b_cfg))
averagedScanData_b_hor = average_scans(load_time_scans(file_path_b_hor))

fig3, (ax3a,ax3b) = plt.subplots(2,1)
plot_averaged_scan(ax3a, averagedScanData_a, color=PlotColor.BLUE,ecolor=PlotColor.BLUE)
plot_averaged_scan(ax3b,averagedScanData_b_cfg,color=PlotColor.GREEN,ecolor=PlotColor.GREEN)
plot_averaged_scan(ax3b,averagedScanData_b_hor,PlotColor.RED,ecolor=PlotColor.RED)


ax3a.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02)) 
ax3a.xaxis.set_major_locator(mpl.ticker.MultipleLocator(250))
ax3b.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02)) 
ax3b.xaxis.set_major_locator(mpl.ticker.MultipleLocator(250))
ax3a.text(
    0.02, 0.95, r'\textbf{(a)}',
    transform=ax3a.transAxes,
    va='top'
)
ax3b.text(
    0.02, 0.95, r'\textbf{(b)}',
    transform=ax3b.transAxes,
    va='top'
)

#fig.suptitle("120 PSI Jet")
fig3.savefig(fig_filename,format='pdf')
#plt.grid(which='major',axis='both')
plt.show()

#Fig3b of uscfg and circularly polarized pulse in droplets overlaid subplots 

#For Figure 2
#folder_path = Path(r"C:\Users\camp06\Documents\droplets_manuscript\202512_Droplets(figure2)\202512_Droplets")

