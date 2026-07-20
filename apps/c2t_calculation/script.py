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

folder_path = Path(r"Z:\Droplets\20260424\Scan2")
file_paths = DatFinder(folder_path,is_full_path=True).find_datafiles()

config = IonDataAnalysisConfig(
    delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
    center=Point(205, 194),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](60, 110),
    transform_parameter=0.78)

raw_scans = load_ion_data(file_paths)
calculated_Scan = run_pipeline(raw_scans, [config])
scan_avg = averaging.average_scans(calculated_Scan)

fig,ax = plt.subplots()
label = "Center = (" + str(config.center.x) + ", " + str(config.center.y) + "), Angle = " + str(round(config.angle.Deg,1)) + "\n"\
        + "Ring: (" + str(config.analysis_zone.min) + ", " + str(config.analysis_zone.max) + "), ScaleX = " + str(config.transform_parameter)
#plot_calculated_scan(ax3, calculated_Scan[0],label=label)
plot_calculated_scan(ax, scan_avg,label=label)
fig.tight_layout()

#fit = fit_gaussian(scan_avg.delays,scan_avg.measured_values)

plt.show()