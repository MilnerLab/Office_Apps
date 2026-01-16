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
'''
folder_path = Path(r"/mnt/valeryshare/Droplets/20260102/Scan4_Scanfiles/")
file_paths = DatFinder().find_scanfiles()

averagedScanData = average_scans(load_time_scans(file_paths))

new_delay: list[Time] = []
for t in averagedScanData.delay:
    new_delay.append(Time(t-Time(80, Prefix.PICO)))
shifted = AveragedScansData(new_delay, averagedScanData.c2t, averagedScanData.file_names)
new = ScanDataBase(shifted.delay[2:], shifted.c2t[2:])

fig, ax = plt.subplots(figsize=(8, 4))
plot_averaged_scan(ax, shifted, label=" -> circ")
ax.tick_params(axis="y", colors=PlotColor.RED)
#plot_GaussianFit(ax, new)

file_paths = DatFinder(folder_path).find_scanfiles()

averagedScanData = average_scans(load_time_scans(file_paths))
ax_r = ax.twinx()     
new = ScanDataBase(averagedScanData.delay[2:], averagedScanData.c2t[2:])
plot_averaged_scan(ax_r, averagedScanData, PlotColor.GREEN, label=" -> CFG")
ax_r.tick_params(axis="y", colors=PlotColor.GREEN  )
#plot_GaussianFit(ax, new)

fig.suptitle('Droplets', fontsize=12)
ax.legend(loc="upper left")
ax_r.legend(loc="upper right")
fig.tight_layout()
plt.show()
'''
'''

folder_path = Path(r"/home/soeren/Downloads/202512_Droplets/202512_Droplets")
file_paths = DatFinder(folder_path).find_scanfiles()

averagedScanData = average_scans(load_time_scans(file_paths))

new_delay: list[Time] = []
for t in averagedScanData.delay:
    new_delay.append(Time(t-Time(80, Prefix.PICO)))
shifted = AveragedScansData(new_delay, averagedScanData.c2t, averagedScanData.file_names)
new = ScanDataBase(shifted.delay[2:], shifted.c2t[2:])
fig, ax = plt.subplots(figsize=(8, 4))
plot_averaged_scan(ax, averagedScanData, label=" -> CFG stabilized")
#plot_GaussianFit(ax, new)

folder_path = Path(r"/mnt/valeryshare/Droplets/20251215/DScan4+5")
file_paths = DatFinder(folder_path).find_scanfiles()

averagedScanData = average_scans(load_time_scans(file_paths))
new = ScanDataBase(averagedScanData.delay[2:], averagedScanData.c2t[2:])
plot_averaged_scan(ax, averagedScanData, PlotColor.GREEN, label=" -> CFG dithering +- 30 ps")
#plot_GaussianFit(ax, new)

fig.suptitle('Droplets', fontsize=12)
ax.legend(loc="upper left")
fig.tight_layout()
plt.show()'''

folder_path = Path(r"/mnt/valeryshare/Droplets/20260102/Scan4_Scanfiles/")
file_paths = DatFinder(folder_path).find_scanfiles()

averagedScanData = average_scans(load_time_scans(file_paths))
fig, ax = plt.subplots(figsize=(8, 4))
plot_averaged_scan(ax, averagedScanData, PlotColor.GREEN, label=" -> CFG dithering +- 30 ps")
plot_GaussianFit(ax, averagedScanData)
fig.suptitle('Droplets', fontsize=12)
ax.legend(loc="upper left")
fig.tight_layout()
plt.show()