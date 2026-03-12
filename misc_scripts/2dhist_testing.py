
from pathlib import Path
import matplotlib.pyplot as plt
from _data_io.dat_loader import load_ion_data

from apps.c2t_calculation.domain.plotting import plot_ions_square
from base_core.lab_specifics.base_models import IonData, IonDataAnalysisConfig, RawScanData
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range, Histogram2D
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length

from base_core.plotting import histogram_plotting
POSZEROSHIFT = 0 #millimetres :)

""" folder_path = Path(r"Z:\Droplets\20260207\Scan6")
#file_paths = DatFinder(folder_path,is_full_path=True).find_datafiles()
file_path = folder_path / '20260207185047DLY_117p5651mm.dat'

original_config = IonDataAnalysisConfig(
    delay_center=Length(92.64-POSZEROSHIFT,Prefix.MILLI),
    center=Point(177,203),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](0, 50),
    transform_parameter=0.73
)

new_config = IonDataAnalysisConfig(
    delay_center= Length(94.5-POSZEROSHIFT, Prefix.MILLI),
    center=Point(180, 202),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](0, 50),
    transform_parameter=0.77)

raw_scans = load_ion_data([[file_path]])
ions = raw_scans[0]
raw_scan: RawScanData = raw_scans[0]
ion_data = raw_scan.ion_datas[0]
points = ion_data.get_points_after_config(original_config)
points_after_config = ion_data.get_points_after_config(new_config)

hist = Histogram2D.compute_histogram(points,x_bins=25,y_bins=25)
hist_config = Histogram2D.compute_histogram(points_after_config,x_bins=50,y_bins=50)
fig, ax = plt.subplots(2,2,figsize=(8,8))
hist_plot = histogram_plotting.plot_histogram2d(ax[0,0],hist)
cs = histogram_plotting.plot_contour(ax[0,0],hist)
plot_ions_square(ax[0,1],points)
hist_config_plot = histogram_plotting.plot_histogram2d(ax[1,0],hist_config)
cs_config = histogram_plotting.plot_contour(ax[1,0],hist_config)
plot_ions_square(ax[1,1],points_after_config)
fig.colorbar(cs,location='right',drawedges=True,shrink=0.3,spacing='proportional')
fig.colorbar(hist_plot,location='bottom')
fig.colorbar(cs_config,location='right',drawedges=True,shrink=0.3)
fig.colorbar(hist_config_plot,location='bottom')
plt.show() """

file_path = Path(r"Z:\Droplets\20260310\CS2Dimer_3\20260310141250DLY___4p5000mm.dat")
config = IonDataAnalysisConfig(
    delay_center=Length(94.5-POSZEROSHIFT,Prefix.MILLI),
    center=Point(171,208),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](0, 20),
    transform_parameter=0.75
)

raw_scans = load_ion_data([[file_path]])
ions = raw_scans[0]
raw_scan: RawScanData = raw_scans[0]
ion_data = raw_scan.ion_datas[0]
points = ion_data.get_points_after_config(config)
hist = Histogram2D.compute_histogram(points,x_bins=100,y_bins=100)
fig, ax = plt.subplots()
hist_plot = histogram_plotting.plot_histogram2d(ax,hist)
cs = histogram_plotting.plot_contour(ax,hist,min_count=1000)
fig.colorbar(cs,location='right',drawedges=True,shrink=0.3,spacing='proportional')
fig.colorbar(hist_plot,location='bottom')
plt.show()