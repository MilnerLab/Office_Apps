from pathlib import Path
from matplotlib import pyplot as plt

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_time_scans
from _data_io.dat_loader import load_time_scan
from apps.single_scan.domain.plotting import plot_single_scan
from _domain.plotting import plot_GaussianFit
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Time

'''

folder_path = Path(r"/mnt/valeryshare/Droplets/20260119/Scan1_ScanFiles")
file_paths = DatFinder(folder_path).find_scanfiles()

averagedScanData = average_scans(load_time_scans(file_paths))

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12, 8))
plot_averaged_scan(ax1, averagedScanData, PlotColor.PURPLE, label="average")
plot_single_scan(ax2,lastScanData,PlotColor.BLUE)
ax1.xaxis.label.set_visible(False)

# ax.tick_params(axis="y", colors=PlotColor.BLUE)
# #plot_GaussianFit(ax, new)

folder_path = Path(r"/mnt/valeryshare/Droplets/20260120/All good cfg randomized scans")
file_paths = DatFinder(folder_path).find_scanfiles(True)

averagedScanData = average_scans(load_time_scans(file_paths))
ax_r = ax.twinx()     
plot_averaged_scan(ax_r, averagedScanData, PlotColor.GREEN, label=" -> CFG randomized")
ax_r.tick_params(axis="y", colors=PlotColor.GREEN  )
plot_GaussianFit(ax, averagedScanData)

fig.suptitle('Droplets', fontsize=12)
#ax.legend(loc="upper left")
#ax_r.legend(loc="upper right")
fig.tight_layout()
#plt.grid(True) 
plt.show()


'''



#folder_path = [Path(r"Z:\Droplets\20260210\Scan3_ScanFiles"),Path(r"Z:\Droplets\20260210\Scan4_ScanFiles")]
scan3_path = Path(r"Z:\Droplets\20260210\Scan3_ScanFiles")
scan4_path = Path(r"Z:\Droplets\20260210\Scan4_ScanFiles")
fig_folder_path = Path(r"C:\Users\camp06\OneDrive - UBC\Documents\droplets_manuscript\test\jetscan.png")
scan3_files = DatFinder(scan3_path).find_scanfiles()
#scan4_files = DatFinder(scan4_path).find_scanfiles()
scan4_files = [Path(r"Z:\Droplets\20260210\Scan4_ScanFiles\20260210150041_ScanFile.dat"),Path(r"Z:\Droplets\20260210\Scan4_ScanFiles\20260210141321_ScanFile.dat")]
averagedScan3Data = average_scans(load_time_scans(scan3_files))
averagedScan4Data = average_scans(load_time_scans(scan4_files))
fig, (ax1, ax2) = plt.subplots(2, 1)
plot_averaged_scan(ax1, averagedScan3Data, PlotColor.GREEN)
plot_averaged_scan(ax2, averagedScan4Data, PlotColor.RED)
#plot_GaussianFit(ax, averagedScanData)
#fig.suptitle('OCS Jet', fontsize=12)
ax1.legend(' OCS Jet',loc="upper left")
ax2.legend(' OCS Droplets',loc="upper left")
ax1.set_xlim(-300,300)
ax2.set_xlim(-300,300)
fig.tight_layout()
#fig.savefig(fig_folder_path,format='png')
plt.show()