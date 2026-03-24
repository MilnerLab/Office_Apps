import copy
from pathlib import Path
import matplotlib.pyplot as plt
import datetime
import re
from _data_io.dat_loader import load_ion_data
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.c2t_calculation.domain.plotting import plot_calculated_scan, plot_ions_square
from _data_io.dat_finder import DatFinder
from _data_io.dat_saver import create_save_path_for_calc_ScanFile
from apps.scan_averaging.domain import averaging
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length
from base_core.fitting.functions import fit_gaussian

POSZEROSHIFT = 0 #millimetres :)

folder_path = Path(r"/mnt/valeryshare/Droplets/20260316/JetScan1")
file_paths = DatFinder(folder_path,is_full_path=True).find_datafiles()

config = IonDataAnalysisConfig(
    delay_center= Length(94.5-POSZEROSHIFT, Prefix.MILLI),
    center=Point(199, 202),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](40, 90),
    transform_parameter=0.76)


label = "Center = (" + str(config.center.x) + ", " + str(config.center.y) + "), Angle = " + str(round(config.angle.Deg,1)) + "\n"\
        + "Ring: (" + str(config.analysis_zone.min) + ", " + str(config.analysis_zone.max) + "), ScaleX = " + str(config.transform_parameter)
raw_scans = load_ion_data(file_paths, [config])

raw_copy = copy.deepcopy(raw_scans[0])

fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])  # oben/unten Höhe anpassen

ax1 = fig.add_subplot(gs[0, 0])   # oben links
ax2 = fig.add_subplot(gs[0, 1])   # oben rechts
ax3 = fig.add_subplot(gs[1, :])   # unten über beide Spalten

 

plot_ions_square(ax1, raw_copy.ion_datas[0])
raw_copy.apply_config()
plot_ions_square(ax2, raw_copy.ion_datas[0])

save_path = create_save_path_for_calc_ScanFile(folder_path, str(raw_scans[0].ion_datas[0].run_id))
calculated_Scan = run_pipeline(raw_scans, save_path)
scan_avg = averaging.average_scans(calculated_Scan)
#plot_calculated_scan(ax3, calculated_Scan[0],label=label)
plot_calculated_scan(ax3, scan_avg,label=label)
ax3.legend(loc="upper right")
ax3.grid(visible=True,which='both',alpha=0.5)
fig.tight_layout()

fit = fit_gaussian(scan_avg.delays,scan_avg.measured_values)

plt.show()