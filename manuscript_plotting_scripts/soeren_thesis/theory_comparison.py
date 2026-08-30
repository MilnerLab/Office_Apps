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
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.models import AggregateSpectrogram
from apps.stft_analysis.domain.plotting import plot_Spectrogram
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from base_core.lab_specifics.averaging.models import AveragedScansData
from base_core.lab_specifics.base_models import IonDataAnalysisConfig, Measurement, ScanDataBase
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time

#Update the matplotlib settings
plt.style.use(r"stylefiles/compare_c2t_spectrogram.mplstyle")

STFTWINDOWSIZE = Time(180, Prefix.PICO)
POSZEROSHIFT = 0  # millimetres :)
SPECTROGRAM_YLIM = [0, 110]  #headroom above 100 GHz; tick locator still lands on 0/50/100
SIMULATION_COLOR = "orangered"

#(a) (b) (c) placement etc - same as breaking_the_wall.py / deceleration.py
TEXTX, TEXTY = 0.1, 0.85
LABEL_KWARGS = dict(color='k', family='DejaVu Sans', usetex=False, fontsize=9,
                     horizontalalignment='center', verticalalignment='center',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', edgecolor='none'))

#Folder to save figures
savefig_folder = r"Z:\Droplets\plots\\"

#Folder with the theory group's simulated cos^2theta_2D(t) curves
simulation_folder = Path(r"/mnt/data/git/Milner_Lab/Latex/droplet_theory_paper/theory_calc/")


#FUNCTION TO GENERATE THE PLOTTABLE DATA (scan + STFT of the scan)
def calculating(folders: list[Path], configs: list[IonDataAnalysisConfig]) -> tuple[AveragedScansData, AggregateSpectrogram]:
    scans_paths = DatFinder(folders).find_datafiles()
    raw_datas = load_ion_data(scans_paths)
    calculated_scans = run_pipeline(raw_datas, configs)
    averagedScanData = average_scans(calculated_scans)
    config = StftAnalysisConfig(calculated_scans, STFTWINDOWSIZE)
    resampled_scans = resample_scans(calculated_scans, config.axis)

    spectrogram = StftAnalysis(resampled_scans, config).calculate_averaged_spectrogram()

    return (averagedScanData, spectrogram)


#Load one of the theory group's "cos2theta2D_vs_t_model_only" CSVs
def load_theory(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, names=['time', 'signal_raw', 'signal_scaled'])


#STFT of the (uniformly sampled) theory curve, reusing the same STFT machinery as the
#experimental data - wrap it as a ScanDataBase so StftAnalysisConfig/StftAnalysis apply unchanged.
def theory_spectrogram(theory: pd.DataFrame, stft_window_size: Time = STFTWINDOWSIZE) -> AggregateSpectrogram:
    theory_scan = ScanDataBase(
        delays=[Time(t, Prefix.PICO) for t in theory.time],
        measured_values=[Measurement(v, 0) for v in theory.signal_scaled],
        run_id=0,
    )
    config = StftAnalysisConfig([theory_scan], stft_window_size)
    resampled_scans = resample_scans([theory_scan], config.axis)

    return StftAnalysis(resampled_scans, config).calculate_averaged_spectrogram()


#Data in black (line + markers), distinct from the blue simulation trace.
def plot_data_points(ax: Axes, scan: AveragedScansData, marker: str = 'd') -> None:
    x = [t.value(Prefix.PICO) for t in scan.delays]
    y = [m.value for m in scan.measured_values]
    ax.plot(x, y, color=PlotColor.BLACK, marker=marker)
    ax.set_xlabel("Probe Delay (ps)")
    ax.set_ylabel(r"$\langle \cos^2 \theta_\mathrm{2D} \rangle$")


@dataclass
class ComparisonColumn:
    label: str  # e.g. 'a' -> (a1) (a2) (a3)
    scan: AveragedScansData
    spec: AggregateSpectrogram
    theory: pd.DataFrame
    theory_spec: AggregateSpectrogram
    spec_vrange: Range[float] = Range(0, 1)


#FUNCTION TO BUILD FIGURE 1: one column per experiment (e.g. accelerating / decelerating),
#three rows - (1) data + simulation overlay, (2) STFT of the data, (3) STFT of the simulation.
#Same panel/label styling as plot_wall_figure in breaking_the_wall.py / deceleration.py.
def plot_theory_comparison_figure(
    columns: list[ComparisonColumn],
    xlim: list[float],
    figsize: tuple[float, float],
) -> Figure:
    fig, axs = plt.subplots(
        nrows=3,
        ncols=len(columns),
        figsize=figsize,
        sharex='col',
        gridspec_kw={'hspace': 0.1, 'wspace': 0.22},
        squeeze=False,
    )

    for col_idx, col in enumerate(columns):
        ax_scan, ax_spec, ax_theory_spec = axs[0, col_idx], axs[1, col_idx], axs[2, col_idx]

        #Row 1: data + simulation
        plot_data_points(ax_scan, col.scan)
        ax_scan.plot(col.theory.time, col.theory.signal_scaled, color=SIMULATION_COLOR)
        ax_scan.grid(True)
        ax_scan.set_xlim(xlim)
        ax_scan.set_xlabel(None)
        ax_scan.set_ylabel(r'$\langle \cos^2\theta_{2D}\rangle$')
        ax_scan.text(TEXTX, TEXTY, f'({col.label}1)', transform=ax_scan.transAxes, **LABEL_KWARGS)

        #Row 2: STFT of the data
        plot_Spectrogram(ax_spec, col.spec, shading='auto', v_range=col.spec_vrange)
        ax_spec.set_ylim(SPECTROGRAM_YLIM)
        ax_spec.set_xlabel(None)
        ax_spec.text(TEXTX, TEXTY, f'({col.label}2)', transform=ax_spec.transAxes, **LABEL_KWARGS)

        #Row 3: STFT of the simulation
        plot_Spectrogram(ax_theory_spec, col.theory_spec, shading='auto', v_range=Range(0, 1))
        ax_theory_spec.set_ylim(SPECTROGRAM_YLIM)
        ax_theory_spec.text(TEXTX, TEXTY, f'({col.label}3)', transform=ax_theory_spec.transAxes, **LABEL_KWARGS)

        #Drop the (redundant) y-axis label text for every column but the first - keep the tick numbers
        if col_idx > 0:
            ax_scan.set_ylabel(None)
            ax_spec.set_ylabel(None)
            ax_theory_spec.set_ylabel(None)

    return fig


#FUNCTION TO BUILD FIGURE 2: single experiment - data + simulation on the left (one row tall,
#vertically centered), STFT of the data (top right) and STFT of the simulation (bottom right).
def plot_single_experiment_theory_figure(
    scan: AveragedScansData,
    spec: AggregateSpectrogram,
    theory: pd.DataFrame,
    theory_spec: AggregateSpectrogram,
    xlim: list[float],
    figsize: tuple[float, float],
    spec_vrange: Range[float] = Range(0, 1),
) -> Figure:
    fig = plt.figure(figsize=figsize)
    #4 sub-rows so the single-row-tall scan axis (middle 2 of 4) can be centered
    #against the two stacked spectrograms (2 of 4 each) on the right.
    gs = fig.add_gridspec(nrows=4, ncols=2, hspace=0.45, wspace=0.35, width_ratios=[3, 2])

    ax_scan = fig.add_subplot(gs[1:3, 0])
    ax_spec = fig.add_subplot(gs[0:2, 1])
    ax_theory_spec = fig.add_subplot(gs[2:4, 1], sharex=ax_spec)

    #Data + simulation (left, one row tall, vertically centered)
    plot_data_points(ax_scan, scan)
    ax_scan.plot(theory.time, theory.signal_scaled, color=SIMULATION_COLOR)
    ax_scan.grid(True)
    ax_scan.set_xlim(xlim)
    ax_scan.set_ylabel(r'$\langle \cos^2\theta_{2D}\rangle$')
    ax_scan.text(TEXTX, TEXTY, '(a)', transform=ax_scan.transAxes, **LABEL_KWARGS)

    #STFT of the data (top right) - shares its x-axis with the STFT below, so hide its own tick labels
    plot_Spectrogram(ax_spec, spec, shading='auto', v_range=spec_vrange)
    ax_spec.set_ylim(SPECTROGRAM_YLIM)
    ax_spec.set_xlim(xlim)  #propagates to ax_theory_spec via sharex
    ax_spec.set_xlabel(None)
    plt.setp(ax_spec.get_xticklabels(), visible=False)
    ax_spec.text(TEXTX, TEXTY, '(b)', transform=ax_spec.transAxes, **LABEL_KWARGS)

    #STFT of the simulation (bottom right)
    plot_Spectrogram(ax_theory_spec, theory_spec, shading='auto', v_range=Range(0, 1))
    ax_theory_spec.set_ylim(SPECTROGRAM_YLIM)
    ax_theory_spec.text(TEXTX, TEXTY, '(c)', transform=ax_theory_spec.transAxes, **LABEL_KWARGS)

    return fig


#====================================================================================================
#CS2 in droplets - accelerating (left column) vs decelerating (right column).
#Configs and simulation files from CS2/CS2_directionality_droplets.py.
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
scan_cs2_accel, spec_cs2_accel = calculating(folders_cs2_accel, configs_cs2_accel)
scan_cs2_decel, spec_cs2_decel = calculating(folders_cs2_decel, configs_cs2_decel)

theory_spec_cs2_accel = theory_spectrogram(theory_cs2_accel)
theory_spec_cs2_decel = theory_spectrogram(theory_cs2_decel)

fig_cs2 = plot_theory_comparison_figure(
    columns=[
        ComparisonColumn('a', scan_cs2_accel, spec_cs2_accel, theory_cs2_accel, theory_spec_cs2_accel, spec_vrange=Range(0, 1)),
        ComparisonColumn('b', scan_cs2_decel, spec_cs2_decel, theory_cs2_decel, theory_spec_cs2_decel, spec_vrange=Range(0, 0.6)),
    ],
    xlim=[CS2_EARLIEST_DELAY_PS, CS2_LATEST_DELAY_PS],
    figsize=(6.75, 4.2),
)
fig_cs2.savefig(savefig_folder + r"cs2_theory_comparison.pdf", format='pdf', dpi=300)


#====================================================================================================
#OCS in droplets - accelerating only.
#Configs and simulation file from ocs/OCS_accel_only_droplets.py.
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

scan_ocs_accel, spec_ocs_accel = calculating(folders_ocs_accel, configs_ocs_accel)
theory_spec_ocs_accel = theory_spectrogram(theory_ocs_accel)

fig_ocs = plot_single_experiment_theory_figure(
    scan_ocs_accel, spec_ocs_accel, theory_ocs_accel, theory_spec_ocs_accel,
    xlim=[OCS_EARLIEST_DELAY_PS, OCS_LATEST_DELAY_PS],
    figsize=(6.75, 2.8),
    spec_vrange=Range(0, 0.6),
)
fig_ocs.savefig(savefig_folder + r"ocs_theory_comparison_accel_only.pdf", format='pdf', dpi=300)


plt.show()
print('Done!')
