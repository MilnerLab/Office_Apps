import sys
from pathlib import Path

#Make the repo root importable no matter where the script is launched from
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from data_paths import data_root

#Data source: "online" (Valery share), "offline" (local copy) or "auto" (share if reachable).
#None = $SOEREN_DATA, else auto.
DATA_MODE: str | None = None
DATA_ROOT = data_root("stabilized_vs_unstabilized", DATA_MODE)

#Update the matplotlib settings
plt.style.use(str(REPO_ROOT / "stylefiles" / "compare_c2t_spectrogram.mplstyle"))

POSZEROSHIFT = 0  # millimetres :)

#The figure is saved next to this script
SAVE = Path(__file__).with_suffix(".pdf")


#FUNCTION TO GENERATE THE PLOTTABLE DATA
def calculating(folders: list[Path], configs: list[IonDataAnalysisConfig]) -> AveragedScansData:
    scans_paths = DatFinder(folders).find_datafiles()
    raw_datas = load_ion_data(scans_paths)
    calculated_scans = run_pipeline(raw_datas, configs)

    return average_scans(calculated_scans)


#====================================================================================================
#20261006 - rotating stabilized vs unstabilized, both sharing one config.
#====================================================================================================

#Common config for every scan in the folder
common_config = IonDataAnalysisConfig(
    delay_center=Length(80.0 - POSZEROSHIFT, Prefix.MILLI),
    center=Point(168, 190),
    angle=Angle(-12, AngleUnit.DEG),
    analysis_zone=Range[int](30, 90),
    transform_parameter=0.920)

#One column per dataset, all sharing the common config above (day folders under DATA_ROOT)
datasets: list[Path] = [
    Path("20261006/RotatingStabilized1"),  #forward slashes: '\' is not a separator on Linux
    Path("20261006/RotatingUnstabilized1"),
]

#Absolute paths: DatFinder still caches share folders to _temp, and reads the local copy in place
scans = [calculating([DATA_ROOT / folder], [common_config]) for folder in datasets]

#Two journal columns (17.8 cm), fonts at final print size (as in single_spectrum.py)
plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "pdf.fonttype": 42})

#One shared y-axis: tick labels and y-label on the left panel only
fig, axs = plt.subplots(
    nrows=1,
    ncols=len(datasets),
    figsize=(17.8 / 2.54, 4.5 / 2.54),
    sharey=True,
    squeeze=False,
    constrained_layout=True)

for col, folder in enumerate(datasets):
    ax_scan = axs[0, col]

    #Averaged <cos^2> trace
    with plt.rc_context({'lines.linewidth': 1.5}):
        plot_averaged_scan(ax_scan, scans[col], PlotColor.BLUE, ecolor=PlotColor.RED, marker='d')

    #plot_ScanData always adds a legend ("N scans"); the panels are told apart by (a), (b)
    if ax_scan.get_legend() is not None:
        ax_scan.get_legend().remove()
    ax_scan.grid(True)
    ax_scan.set_xlabel('Probe Delay (ps)')
    #\textbf, not fontweight="bold": the style sets text.usetex, which ignores fontweight
    ax_scan.text(0.965, 0.91, rf"\textbf{{({chr(ord('a') + col)})}}", transform=ax_scan.transAxes,
                 color="k", fontsize=12, va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="0.85", edgecolor="k", lw=0.6))
    print(f"({chr(ord('a') + col)}) {folder.name}")

    if col > 0:
        ax_scan.set_ylabel('')

axs[0, 0].set_ylabel(r'$\langle \cos^2\theta_{2D}\rangle$')

fig.savefig(SAVE, format='pdf', dpi=300)
print(f"Saved {SAVE}")

plt.show()
print('Done!')
