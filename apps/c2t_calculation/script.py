from pathlib import Path
import matplotlib.pyplot as plt

from _data_io.dat_loader import load_ion_data
from apps.c2t_calculation.domain.config import AnalysisConfig
from apps.c2t_calculation.domain.pipeline import run_pipeline_for_multiple
from apps.c2t_calculation.domain.plotting import plot_calculated_scan, plot_ions_square
from _data_io.dat_finder import DatFinder
from _data_io.dat_saver import create_save_path_for_calc_ScanFile
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range


folder_path = Path(r"/mnt/valeryshare/Droplets/20251215/DScan2")
file_paths = DatFinder(folder_path).find_datafiles()

config = AnalysisConfig(
    center=Point(206, 200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](15, 120),
    transform_parameter= 0.77)

ion_data = load_ion_data(file_paths)

fig, ax = plt.subplots(figsize=(4, 4))
plot_ions_square(ax, ion_data[0])
fig.tight_layout()
plt.show()

save_path = create_save_path_for_calc_ScanFile(folder_path, str(ion_data[0].run_id))
calculated_Scan = run_pipeline_for_multiple(ion_data, config, save_path)

fig, ax = plt.subplots(figsize=(8, 4))
plot_calculated_scan(ax, calculated_Scan)
ax.legend(loc="upper right")
fig.tight_layout()
plt.show()