from pathlib import Path
from matplotlib import pyplot as plt

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_time_scans
from _domain.models import ScanDataBase
from _domain.plotting import plot_GaussianFit
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.models import AveragedScansData
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Time


folder_path = Path(r"/mnt/valeryshare/Droplets/20260119/Scan1_ScanFiles")
file_paths = DatFinder(folder_path).find_scanfiles()

averagedScanData = average_scans(load_time_scans(file_paths))


fig, ax = plt.subplots(figsize=(8, 4))
plot_averaged_scan(ax, averagedScanData, PlotColor.BLUE, label=" -> horizontal GA")
ax.tick_params(axis="y", colors=PlotColor.BLUE)
#plot_GaussianFit(ax, new)

folder_path = Path(r"/mnt/valeryshare/Droplets/20260120/All good cfg randomized scans")
file_paths = DatFinder(folder_path).find_scanfiles(True)

averagedScanData = average_scans(load_time_scans(file_paths))
ax_r = ax.twinx()     
plot_averaged_scan(ax_r, averagedScanData, PlotColor.GREEN, label=" -> CFG randomized")
ax_r.tick_params(axis="y", colors=PlotColor.GREEN  )
plot_GaussianFit(ax, averagedScanData)

fig.suptitle('Droplets', fontsize=12)
ax.legend(loc="upper left")
ax_r.legend(loc="upper right")
fig.tight_layout()
plt.grid(True) 
plt.show()


'''

folder_path = Path(r"/home/soeren/Downloads/all cfg scans")
file_paths = DatFinder().find_scanfiles(mergescans=True)

averagedScanData = average_scans(load_time_scans(file_paths))
fig, ax = plt.subplots(figsize=(8, 4))
plot_averaged_scan(ax, averagedScanData, PlotColor.GREEN, label=" -> CFG randomized")
plot_GaussianFit(ax, averagedScanData)
fig.suptitle('Droplets', fontsize=12)
ax.legend(loc="upper left")
fig.tight_layout()
plt.show()
'''