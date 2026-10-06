from pathlib import Path
from matplotlib import pyplot as plt

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from base_core.lab_specifics.averaging.models import AveragedScansData
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length

#Update the matplotlib settings
plt.style.use(r"stylefiles/compare_c2t_spectrogram.mplstyle")

POSZEROSHIFT = 0  # millimetres :)
EARLIEST_DELAY_PS = -800
LATEST_DELAY_PS = 1100

#Folder to save figures
savefig_folder = r"Z:\Droplets\plots\\"


#FUNCTION TO GENERATE THE PLOTTABLE DATA
def calculating(folders: list[Path], configs: list[IonDataAnalysisConfig]) -> AveragedScansData:
    scans_paths = DatFinder(folders).find_datafiles()
    raw_datas = load_ion_data(scans_paths)
    calculated_scans = run_pipeline(raw_datas, configs)

    return average_scans(calculated_scans)


#====================================================================================================
#OCS vs CS2 - long-lasting revival, both in droplets. Configs from
#usCFG_Nanodroplets/OCS_CS2_longlasting.py.
#====================================================================================================

DROPLETRADIUSMIN = 60

#OCS, 3mJ usCFG (polarization averaged)
folders_ocs: list[Path] = []
configs_ocs: list[IonDataAnalysisConfig] = []

folders_ocs.append(Path(r"20260211\Scan1"))
configs_ocs.append(IonDataAnalysisConfig(
    delay_center=Length(92.654 - POSZEROSHIFT, Prefix.MILLI),
    center=Point(174, 206),
    angle=Angle(12, AngleUnit.DEG),
    analysis_zone=Range[int](50, 120),
    transform_parameter=0.78))

#CS2 in droplets, CFG
folders_cs2: list[Path] = []
configs_cs2: list[IonDataAnalysisConfig] = []

folders_cs2.append(Path(r"20260120\Scan1"))
configs_cs2.append(IonDataAnalysisConfig(
    delay_center=Length(98.054 - POSZEROSHIFT, Prefix.MILLI),
    center=Point(228, 193),
    angle=Angle(12, AngleUnit.DEG),
    analysis_zone=Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.75))

folders_cs2.append(Path(r"20260120\Scan2"))
configs_cs2.append(configs_cs2[0])  #same config

folders_cs2.append(Path(r"20260119\Scan2_CFG"))
configs_cs2.append(configs_cs2[0])  #same config

scan_ocs = calculating(folders_ocs, configs_ocs)
scan_cs2 = calculating(folders_cs2, configs_cs2)

fig, ax = plt.subplots(figsize=(6.75, 2))

with plt.rc_context({'lines.linewidth': 1.5}):
    plot_averaged_scan(ax, scan_ocs, 'darkgreen', ecolor=PlotColor.RED, marker='d', label='OCS')
    plot_averaged_scan(ax, scan_cs2, PlotColor.BLUE, ecolor=PlotColor.RED, marker='s', label='CS2')

ax.grid(True)
ax.set_xlim([EARLIEST_DELAY_PS, LATEST_DELAY_PS])
ax.set_xlabel('Probe Delay (ps)')
ax.set_ylabel(r'$\langle \cos^2\theta_{2D}\rangle$')
ax.legend()

fig.savefig(savefig_folder + r"ocs_cs2_longlasting.pdf", format='pdf', dpi=300)


plt.show()
print('Done!')
