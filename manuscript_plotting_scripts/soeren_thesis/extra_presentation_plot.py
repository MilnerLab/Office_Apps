from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.scan_averaging.domain.averaging import average_scans
from base_core.lab_specifics.averaging.models import AveragedScansData
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length

#Update the matplotlib settings
plt.style.use(r"stylefiles/compare_c2t_spectrogram.mplstyle")

#Bigger type than the thesis figures - this one is meant to be read off a projector.
#Set here rather than in the shared style file, so only this script is affected.
FONTSIZE = 13
plt.rcParams.update({
    'font.size': FONTSIZE,
    'axes.labelsize': FONTSIZE,
    'xtick.labelsize': FONTSIZE - 1,
    'ytick.labelsize': FONTSIZE - 1,
})

POSZEROSHIFT = 0  # millimetres :)
SIMULATION_COLOR = "orangered"

#Row labels - same styling as theory_comparison.py, but sitting in their own column to the
#left of the panels instead of inside them.
LABEL_KWARGS = dict(color='k', family='DejaVu Sans', usetex=False, fontsize=FONTSIZE,
                     horizontalalignment='center', verticalalignment='center',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', edgecolor='none'))
LABEL_COLUMN_WIDTH_RATIO = [1, 4]  #label column : plot column - wide enough for 'decelerating'
LABEL_COLUMN_GAP = 0.25  #wspace - keeps the label boxes clear of the y-axis labels

#Folder to save figures
savefig_folder = r"Z:\Droplets\plots\\"

#Folder with the theory group's simulated cos^2theta_2D(t) curves
simulation_folder = Path(r"/mnt/data/git/Milner_Lab/Latex/droplet_theory_paper/theory_calc/")


#FUNCTION TO GENERATE THE PLOTTABLE DATA
def calculating(folders: list[Path], configs: list[IonDataAnalysisConfig]) -> AveragedScansData:
    scans_paths = DatFinder(folders).find_datafiles()
    raw_datas = load_ion_data(scans_paths)
    calculated_scans = run_pipeline(raw_datas, configs)

    return average_scans(calculated_scans)


#Load one of the theory group's "cos2theta2D_vs_t_model_only" CSVs
def load_theory(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, names=['time', 'signal_raw', 'signal_scaled'])


#Data in black (line + markers), distinct from the simulation trace.
def plot_data_points(ax: Axes, scan: AveragedScansData, marker: str = 'd') -> None:
    x = [t.value(Prefix.PICO) for t in scan.delays]
    y = [m.value for m in scan.measured_values]
    ax.plot(x, y, color=PlotColor.BLACK, marker=marker)
    ax.set_xlabel("Probe Delay (ps)")
    ax.set_ylabel(r"$\langle \cos^2 \theta_\mathrm{2D} \rangle$")


@dataclass
class PresentationRow:
    label: str  # e.g. '(a) CS$_2$ accelerating'
    scan: AveragedScansData
    theory: pd.DataFrame
    xlim: list[float]


#FUNCTION TO BUILD THE PRESENTATION FIGURE: one row per experiment, data + simulation overlay.
#Two columns - a narrow, axis-less one carrying only the row label, and the plot itself.
#The rows keep their own x-ranges, so only the bottom panel carries the x-axis label.
def plot_presentation_figure(rows: list[PresentationRow], figsize: tuple[float, float]) -> Figure:
    fig, axs = plt.subplots(
        nrows=len(rows),
        ncols=2,
        figsize=figsize,
        gridspec_kw={'hspace': 0.25, 'wspace': LABEL_COLUMN_GAP, 'width_ratios': LABEL_COLUMN_WIDTH_RATIO},
        squeeze=False,
    )

    for row_idx, row in enumerate(rows):
        ax_label, ax = axs[row_idx, 0], axs[row_idx, 1]

        #Label column: nothing but the text, vertically centered against its panel
        ax_label.axis('off')
        ax_label.text(0.5, 0.5, row.label, transform=ax_label.transAxes, **LABEL_KWARGS)

        plot_data_points(ax, row.scan)
        ax.plot(row.theory.time, row.theory.signal_scaled, color=SIMULATION_COLOR)
        ax.grid(True)
        ax.set_xlim(row.xlim)
        ax.set_ylabel(r'$\langle \cos^2\theta_{2D}\rangle$')

        if row_idx < len(rows) - 1:
            ax.set_xlabel(None)

    return fig


#====================================================================================================
#CS2 in droplets - accelerating and decelerating.
#Configs and simulation files from theory_comparison.py.
#====================================================================================================

CS2_DROPLETRADIUSMIN = 65
CS2_EARLIEST_DELAY_PS = -200
CS2_LATEST_DELAY_PS = -CS2_EARLIEST_DELAY_PS

#Accelerating
configs_cs2_accel: list[IonDataAnalysisConfig] = []
folders_cs2_accel: list[Path] = []

folders_cs2_accel.append(Path(r"20260430\Scan3"))  #GA=0, DA=15.5, ACCELERATING
configs_cs2_accel.append(IonDataAnalysisConfig(
    delay_center=Length(93.3 - POSZEROSHIFT, Prefix.MILLI),
    center=Point(205, 194),
    angle=Angle(12, AngleUnit.DEG),
    analysis_zone=Range[int](CS2_DROPLETRADIUSMIN, 120),
    transform_parameter=0.78))

folders_cs2_accel.append(Path(r"20260501\Scan1"))  #GA=0, DA=15.5, ACCELERATING
configs_cs2_accel.append(IonDataAnalysisConfig(
    delay_center=Length(93.3 - POSZEROSHIFT, Prefix.MILLI),
    center=Point(205, 194),
    angle=Angle(12, AngleUnit.DEG),
    analysis_zone=Range[int](CS2_DROPLETRADIUSMIN, 120),
    transform_parameter=0.78))

#Decelerating
configs_cs2_decel: list[IonDataAnalysisConfig] = []
folders_cs2_decel: list[Path] = []

folders_cs2_decel.append(Path(r"20260429\Scan1"))  #GA=0, DA=16.42, DECELERATING
configs_cs2_decel.append(IonDataAnalysisConfig(
    delay_center=Length(93.3 - POSZEROSHIFT, Prefix.MILLI),
    center=Point(205, 194),
    angle=Angle(12, AngleUnit.DEG),
    analysis_zone=Range[int](CS2_DROPLETRADIUSMIN, 120),
    transform_parameter=0.78))

#Simulation
theory_cs2_accel = load_theory(simulation_folder / "CS2_accelerating_droplets_CS2_cos2theta2D_vs_t_model_only.csv")
theory_cs2_decel = load_theory(simulation_folder / "CS2_decelerating_droplets_CS2_cos2theta2D_vs_t_model_only.csv")

#Pipeline
scan_cs2_accel = calculating(folders_cs2_accel, configs_cs2_accel)
scan_cs2_decel = calculating(folders_cs2_decel, configs_cs2_decel)


#====================================================================================================
#OCS in droplets - accelerating only.
#====================================================================================================

OCS_DROPLETRADIUSMIN = 65
OCS_EARLIEST_DELAY_PS = -230
OCS_LATEST_DELAY_PS = -OCS_EARLIEST_DELAY_PS

configs_ocs_accel: list[IonDataAnalysisConfig] = []
folders_ocs_accel: list[Path] = []

folders_ocs_accel.append(Path(r"20260210\Scan4"))  #GA=0, DA=16.6mm, still has ~40GHz central oscillation frequency
configs_ocs_accel.append(IonDataAnalysisConfig(
    delay_center=Length(92.654 - POSZEROSHIFT, Prefix.MILLI),
    center=Point(175, 205),
    angle=Angle(12, AngleUnit.DEG),
    analysis_zone=Range[int](OCS_DROPLETRADIUSMIN, 120),
    transform_parameter=0.75))

theory_ocs_accel = load_theory(simulation_folder / "OCS_accelerating_droplets_OCS_cos2theta2D_vs_t_model_only.csv")

scan_ocs_accel = calculating(folders_ocs_accel, configs_ocs_accel)


#====================================================================================================
#The figure: CS2 accelerating / CS2 decelerating / OCS accelerating, stacked
#====================================================================================================

fig = plot_presentation_figure(
    rows=[
        PresentationRow(
            r'CS$_2$' + '\naccelerating', scan_cs2_accel, theory_cs2_accel,
            xlim=[CS2_EARLIEST_DELAY_PS, CS2_LATEST_DELAY_PS]),
        PresentationRow(
            r'CS$_2$' + '\ndecelerating', scan_cs2_decel, theory_cs2_decel,
            xlim=[CS2_EARLIEST_DELAY_PS, CS2_LATEST_DELAY_PS]),
        PresentationRow(
            'OCS\naccelerating', scan_ocs_accel, theory_ocs_accel,
            xlim=[OCS_EARLIEST_DELAY_PS, OCS_LATEST_DELAY_PS]),
    ],
    figsize=(10, 6),
)
fig.savefig(savefig_folder + r"presentation_theory_comparison_overview.pdf", format='pdf', dpi=300)


plt.show()
print('Done!')
