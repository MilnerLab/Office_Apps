from pathlib import Path
from matplotlib import pyplot as plt

from Lab_apps._domain.plotting import plot_GaussianFit
from Lab_apps._io.dat_finder import DatFinder
from Lab_apps._io.dat_loader import load_time_scans
from Lab_apps.scan_averaging.domain.averaging import average_scans
from Lab_apps.scan_averaging.domain.plotting import plot_averaged_scan

folder_path = Path(r"Z:\Droplets\20251127\Scan3_ScanFiles")
file_paths = DatFinder().find_scanfiles()

averagedScanData = average_scans(load_time_scans(file_paths))

fig, ax = plt.subplots(figsize=(8, 4))
plot_averaged_scan(ax, averagedScanData)
#plot_GaussianFit(ax, averagedScanData)

fig.suptitle('Droplets', fontsize=12)
ax.legend(loc="upper right")
fig.tight_layout()
plt.show()