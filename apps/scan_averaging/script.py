from pathlib import Path
from matplotlib import pyplot as plt

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.single_scan.domain.plotting import plot_single_scan
from _domain.plotting import plot_GaussianFit
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from base_core.lab_specifics.base_models import IonDataAnalysisConfig, RawScanData
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Time,Length
from base_core.math.models import Point, Angle, Range
from base_core.math.enums import AngleUnit




""" folder_path = Path(r"/mnt/valeryshare/Droplets/20260119/Scan1_ScanFiles")
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
 """
folder_path = Path(r"Z:\Droplets\20260323\Scan1")
file_paths = DatFinder(folder_path,is_full_path=True).find_datafiles()
fig_foldersavepath = Path(r"C:\Users\camp06\OneDrive - UBC\Documents\CalibrationData")
fig_filesavepath = fig_foldersavepath / "20260316TimeZeroScanJet"
raw_scans = load_ion_data(file_paths)
#raw_scan: RawScanData = raw_scans[0]



config = IonDataAnalysisConfig(
    delay_center= Length(93.3, Prefix.MILLI),
    center=Point(175, 200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](40, 90),
    transform_parameter=0.8)

c2t_data = run_pipeline(raw_scans,config)
scan_avg = average_scans(c2t_data)
fig,ax = plt.subplots()

plot_averaged_scan(ax,scan_avg)
#plot_GaussianFit(ax,scan_avg)
fig.suptitle(f"{str(folder_path)}", fontsize=10)
#fig.savefig(fig_filesavepath,format='pdf')
plt.show()