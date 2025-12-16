from pathlib import Path
from matplotlib import pyplot as plt

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_time_scans
from _domain.models import ScanDataBase
from _domain.plotting import plot_GaussianFit
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.models import AveragedScansData
from apps.scan_averaging.domain.plotting import plot_averaged_scan

folder_path = Path(r"/mnt/valeryshare/Droplets/20251215/DScan4+5")
file_paths = DatFinder(folder_path).find_scanfiles()

averagedScanData = average_scans(load_time_scans(file_paths))

new = ScanDataBase(averagedScanData.delay[4:], averagedScanData.c2t[4:])
fig, ax = plt.subplots(figsize=(8, 4))
plot_averaged_scan(ax, averagedScanData)
#plot_GaussianFit(ax, new)

fig.suptitle('Droplets', fontsize=12)
ax.legend(loc="upper right")
fig.tight_layout()
plt.show()