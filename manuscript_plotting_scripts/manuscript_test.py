
from pathlib import Path
from matplotlib import pyplot as plt

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from _data_io.dat_saver import create_save_path_for_calc_ScanFile
from apps.c2t_calculation.domain.config import IonDataAnalysisConfig
from apps.c2t_calculation.domain.pipeline import run_pipeline
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length

configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []

folders.append(Path(r"/mnt/valeryshare/Droplets/20260207/Scan1"))
configs.append(IonDataAnalysisConfig(
    delay_center= Length(92.654, Prefix.MILLI),
    center=Point(177, 203),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](60, 120),
    transform_parameter= 0.73))

folders.append(Path(r"/mnt/valeryshare/Droplets/20260207/Scan2"))
configs.append(IonDataAnalysisConfig(
    delay_center= Length(92.654, Prefix.MILLI),
    center=Point(177, 203),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](60, 120),
    transform_parameter= 0.73))

folders.append(Path(r"/mnt/valeryshare/Droplets/20260207/Scan3"))
configs.append(IonDataAnalysisConfig(
    delay_center= Length(92.654, Prefix.MILLI),
    center=Point(177, 203),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](60, 120),
    transform_parameter= 0.73))

scans_paths = DatFinder(folders).find_datafiles()

raw_datas = load_ion_data(scans_paths, configs)
save_path = create_save_path_for_calc_ScanFile(folders[0], str(raw_datas[0].ion_datas[0].run_id))
calculated_scans = run_pipeline(raw_datas, save_path)

fig, ax = plt.subplots(figsize=(8, 4))
averagedScanData = average_scans(calculated_scans)

plot_averaged_scan(ax, averagedScanData, PlotColor.GREEN, label=" -> CFG randomized")
fig.suptitle('Droplets', fontsize=12)
ax.legend(loc="upper left")
fig.tight_layout()
plt.show()