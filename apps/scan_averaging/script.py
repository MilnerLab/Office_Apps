from pathlib import Path
from matplotlib import pyplot as plt

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_time_scans
from apps.single_scan.domain.plotting import plot_single_scan
from _domain.plotting import plot_GaussianFit
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from base_core.lab_specifics.base_models import RawScanData
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Time,Length
from base_core.math.models import Point, Angle, Range
from base_core.math.enums import AngleUnit

file_paths = DatFinder().find_scanfiles()
scans = load_time_scans(file_paths)

scan_avg = average_scans(scans)
fig,ax = plt.subplots()

plot_averaged_scan(ax,scan_avg,ecolor='red')
fig.suptitle("Michelson cfCFG frequency scan OCS jet", fontsize=10)
#fig.savefig(fig_filesavepath,format='pdf')
plt.show()