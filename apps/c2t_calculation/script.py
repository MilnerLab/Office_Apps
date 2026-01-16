import copy
from pathlib import Path
import matplotlib.pyplot as plt

from _data_io.dat_loader import load_ion_data
from apps.c2t_calculation.domain.config import AnalysisConfig
from apps.c2t_calculation.domain.pipeline import run_pipeline
from apps.c2t_calculation.domain.plotting import plot_calculated_scan, plot_ions_square
from _data_io.dat_finder import DatFinder
from _data_io.dat_saver import create_save_path_for_calc_ScanFile
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range


folder_path = Path(r"/mnt/valeryshare/Droplets/20260102/Scan4")
file_paths = DatFinder(folder_path).find_datafiles()

config = AnalysisConfig(
    center=Point(189, 205),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](60, 120),
    transform_parameter= 0.75)

ion_data = load_ion_data(file_paths)

plot_ions = copy.deepcopy(ion_data[8])

fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])  # oben/unten Höhe anpassen

ax1 = fig.add_subplot(gs[0, 0])   # oben links
ax2 = fig.add_subplot(gs[0, 1])   # oben rechts
ax3 = fig.add_subplot(gs[1, :])   # unten über beide Spalten

plot_ions_square(ax1, plot_ions)
plot_ions.apply_config(config)
plot_ions_square(ax2, plot_ions)

save_path = create_save_path_for_calc_ScanFile(folder_path, str(ion_data[0].run_id))
calculated_Scan = run_pipeline(ion_data,config, save_path)
plot_calculated_scan(ax3, calculated_Scan)
ax3.legend(loc="upper right")
fig.tight_layout()
plt.show()