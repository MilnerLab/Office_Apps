from pathlib import Path
import matplotlib.pyplot as plt
import datetime
import re
from _data_io.dat_finder import DatFinder
from _data_io.dat_saver import create_save_path_for_calc_ScanFile
from _data_io.dat_loader import load_ion_data
from apps.c2t_calculation.domain.plotting import plot_calculated_scan, plot_ions_square

from base_core.lab_specifics.base_models import IonDataAnalysisConfig, RawScanData
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range, Histogram2D
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length

from base_core.plotting import histogram_plotting
POSZEROSHIFT = 0 #millimetres :)

folder_path = Path(r"Z:\Droplets\20260207\Scan6")
#file_paths = DatFinder(folder_path,is_full_path=True).find_datafiles()
file_path = folder_path / '20260207185047DLY_117p5651mm.dat'

config = IonDataAnalysisConfig(
    delay_center= Length(94.5-POSZEROSHIFT, Prefix.MILLI),
    center=Point(177, 203),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](0, 50),
    transform_parameter=0.73)

raw_scans = load_ion_data([[file_path]])
#x_range = Range(config.center.x - 50, config.center.x + 50)
#y_range = Range(config.center.y - 50, config.center.y + 50)

ions = raw_scans[0]
#ions_config = copy.deepcopy(raw_scans[0])
#ions_config.apply_config()
original_center = Point(177,203)





raw_scan: RawScanData = raw_scans[0]
ion_data = raw_scan.ion_datas[0]
points = ion_data.points
points_after_config = ion_data.get_points_after_config(config)

hist = Histogram2D.compute_histogram(points,center=original_center,x_bins=50,y_bins=50)
hist_config = Histogram2D.compute_histogram(points_after_config,center=config.center,x_bins=50,y_bins=50)
fig, ax = plt.subplots(2,2,figsize=(8,8))
histogram_plotting.plot_histogram2d(ax[0,0],hist)
histogram_plotting.plot_contour(ax[0,0],hist)
plot_ions_square(ax[0,1],points)
histogram_plotting.plot_histogram2d(ax[1,0],hist_config)
histogram_plotting.plot_contour(ax[1,0],hist_config)
plot_ions_square(ax[1,1],points_after_config)
plt.show()
