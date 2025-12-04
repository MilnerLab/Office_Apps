from pathlib import Path

from matplotlib import pyplot as plt

from Lab_apps._base.models import Angle, AngleUnit, Length, Point, Range
from Lab_apps._io.dat_finder import DatFinder
from Lab_apps._io.dat_loader import load_ion_data
from Lab_apps._io.dat_saver import create_save_path_for_calc_ScanFile
from Lab_apps.c2t_calculation.domain.config import AnalysisConfig
from Lab_apps.c2t_calculation.domain.pipeline import run_pipeline_for_multiple
from Lab_apps.c2t_calculation.domain.plotting import plot_calculated_scan


folder_path = Path(r"Z:\Droplets\20251121\Scan1")
file_paths = DatFinder(folder_path).find_datafiles()

config = AnalysisConfig(
    center=Point(206, 200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](15, 120),
    transform_parameter= 0.77)

ion_data = load_ion_data(file_paths)

save_path = create_save_path_for_calc_ScanFile(folder_path, str(ion_data[0].run_id))
calculated_Scan = run_pipeline_for_multiple(ion_data, config, save_path)

fig, ax = plt.subplots(figsize=(8, 4))
plot_calculated_scan(ax, calculated_Scan)
ax.legend(loc="upper right")
fig.tight_layout()
plt.show()