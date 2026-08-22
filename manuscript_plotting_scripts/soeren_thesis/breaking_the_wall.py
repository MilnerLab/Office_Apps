from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.models import AggregateSpectrogram
from apps.stft_analysis.domain.plotting import plot_Spectrogram
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from base_core.lab_specifics.averaging.models import AveragedScansData
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time

#Update the matplotlib settings
plt.style.use(r"stylefiles/compare_c2t_spectrogram.mplstyle")

STFTWINDOWSIZE = Time(180, Prefix.PICO)

#Folder to save figures
savefig_folder = r"Z:\Droplets\plots\\"


#FUNCTION TO GENERATE THE PLOTTABLE DATA
def calculating(folders: list[Path], configs: list[IonDataAnalysisConfig]) -> tuple[AveragedScansData, AggregateSpectrogram]:
    scans_paths = DatFinder(folders).find_datafiles()
    raw_datas = load_ion_data(scans_paths)
    calculated_scans = run_pipeline(raw_datas, configs)
    averagedScanData = average_scans(calculated_scans)
    config = StftAnalysisConfig(calculated_scans, STFTWINDOWSIZE)
    resampled_scans = resample_scans(calculated_scans, config.axis)

    spectrogram = StftAnalysis(resampled_scans, config).calculate_averaged_spectrogram()

    return (averagedScanData, spectrogram)


#FUNCTION TO BUILD ONE "BREAKING THE WALL" FIGURE: jet on top row, droplets on bottom row,
#scan on the left column, spectrogram on the right column - shared by CS2 and OCS below.
def plot_wall_figure(
    scan_jet: AveragedScansData,
    spec_jet: AggregateSpectrogram,
    scan_droplets: AveragedScansData,
    spec_droplets: AggregateSpectrogram,
    spectrogram_ylim: list[float],
    xlim: list[float] | None = None,
    spectrogram_vrange: Range[float] = Range(0, 1),
    droplet_vline_x: float | None = None,
    spectrogram_hline_y: float | None = None,
) -> Figure:
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(6.75, 2.8),
        sharex='col',
        gridspec_kw={'hspace': 0.1, 'wspace': 0.45, 'width_ratios': [3, 2]}
    )
    ax_scan_jet, ax_spec_jet = axs[0, 0], axs[0, 1]
    ax_scan_drop, ax_spec_drop = axs[1, 0], axs[1, 1]

    plot_averaged_scan(ax_scan_jet, scan_jet, PlotColor.BLACK, ecolor=PlotColor.RED, marker='d', label=None)
    plot_averaged_scan(ax_scan_drop, scan_droplets, PlotColor.BLACK, ecolor=PlotColor.RED, marker='d', label=None)

    plot_Spectrogram(ax_spec_jet, spec_jet, shading='auto', v_range=spectrogram_vrange)
    ax_spec_jet.set_ylim(spectrogram_ylim)

    plot_Spectrogram(ax_spec_drop, spec_droplets, shading='auto', v_range=spectrogram_vrange)
    ax_spec_drop.set_ylim(spectrogram_ylim)

    #Horizontal marker line on droplets spectrogram (e.g. the surviving oscillation frequency)
    if spectrogram_hline_y is not None:
        ax_spec_drop.axhline(spectrogram_hline_y, color='w', linestyle='--', linewidth=1)

    if xlim is not None:
        ax_scan_jet.set_xlim(xlim)
        ax_spec_jet.set_xlim(xlim)

    ax_scan_jet.grid(True)
    ax_scan_drop.grid(True)

    #Vertical marker line on droplets scan (e.g. approximate position of the wall)
    if droplet_vline_x is not None:
        y = list(ax_scan_drop.get_ylim())
        ax_scan_drop.plot([droplet_vline_x, droplet_vline_x], y, 'k--', linewidth=1)

    #(a) (b) (c) (d) placement etc
    textx, texty = 0.1, 0.85
    label_kwargs = dict(color='k', family='DejaVu Sans', usetex=False, fontsize=9,
                         horizontalalignment='center', verticalalignment='center',
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', edgecolor='none'))
    ax_scan_jet.text(textx, texty, '(a)', transform=ax_scan_jet.transAxes, **label_kwargs)
    ax_spec_jet.text(textx, texty, '(b)', transform=ax_spec_jet.transAxes, **label_kwargs)
    ax_scan_drop.text(textx, texty, '(c)', transform=ax_scan_drop.transAxes, **label_kwargs)
    ax_spec_drop.text(textx, texty, '(d)', transform=ax_spec_drop.transAxes, **label_kwargs)

    ax_scan_jet.set_xlabel(None)
    ax_spec_jet.set_xlabel(None)
    ax_scan_jet.set_ylabel(r'$\langle \cos^2\theta_{2D}\rangle$')
    ax_scan_drop.set_ylabel(r'$\langle \cos^2\theta_{2D}\rangle$')

    return fig


#====================================================================================================
#CS2 - jet vs droplets (2025 data)
#====================================================================================================

CS2_DROPLETRADIUSMIN = 60
CS2_POSZEROSHIFT = 5  # millimetres :)

#Loading droplet data. GA=26mm, DA=15.9mm
configs_droplets_cs2: list[IonDataAnalysisConfig] = []
folders_droplets_cs2: list[Path] = []

folders_droplets_cs2.append(Path(r"20251212\Scan4"))  #Combination of 20251212 and 20251213.
configs_droplets_cs2.append(IonDataAnalysisConfig(
    delay_center=Length(90.55 - CS2_POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle=Angle(12, AngleUnit.DEG),
    analysis_zone=Range[int](CS2_DROPLETRADIUSMIN, 120),
    transform_parameter=0.79))

folders_droplets_cs2.append(Path(r"20251213\Scan1"))  #Combination of 20251212 and 20251213.
configs_droplets_cs2.append(IonDataAnalysisConfig(
    delay_center=Length(90.55 - CS2_POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle=Angle(12, AngleUnit.DEG),
    analysis_zone=Range[int](CS2_DROPLETRADIUSMIN, 120),
    transform_parameter=0.79))

folders_droplets_cs2.append(Path(r"20251213\Scan2"))  #Combination of 20251212 and 20251213.
configs_droplets_cs2.append(IonDataAnalysisConfig(
    delay_center=Length(90.55 - CS2_POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle=Angle(12, AngleUnit.DEG),
    analysis_zone=Range[int](CS2_DROPLETRADIUSMIN, 120),
    transform_parameter=0.79))

folders_droplets_cs2.append(Path(r"20251213\Scan3"))  #Combination of 20251212 and 20251213.
configs_droplets_cs2.append(IonDataAnalysisConfig(
    delay_center=Length(90.55 - CS2_POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle=Angle(12, AngleUnit.DEG),
    analysis_zone=Range[int](CS2_DROPLETRADIUSMIN, 120),
    transform_parameter=0.79))

#Loading jet data
folders_jet_cs2: list[Path] = []
configs_jet_cs2: list[IonDataAnalysisConfig] = []

folders_jet_cs2.append(Path(r"20251210\JSS3"))  #20251210 JSS3 is dense throughout the centrifuge
configs_jet_cs2.append(IonDataAnalysisConfig(
    delay_center=Length(90.55 - CS2_POSZEROSHIFT, Prefix.MILLI),
    center=Point(195, 197),
    angle=Angle(12, AngleUnit.DEG),
    analysis_zone=Range[int](20, 120),
    transform_parameter=0.79))

folders_jet_cs2.append(Path(r"20251210\JSS4"))  #20251210 JSS4 is dense before the centrifuge
configs_jet_cs2.append(configs_jet_cs2[0])  #same config

scan_jet_cs2, spec_jet_cs2 = calculating(folders_jet_cs2, configs_jet_cs2)
scan_drop_cs2, spec_drop_cs2 = calculating(folders_droplets_cs2, configs_droplets_cs2)

#Truncating so that the two datasets have the same delay range
times = np.array([t.value(Prefix.PICO) for t in scan_drop_cs2.delays])
start = np.where(times > -300)[0][0]
scan_drop_cs2.cut(start=start)

times = np.array([t.value(Prefix.PICO) for t in scan_jet_cs2.delays])
end = np.where(times < 300)[0][-1]
scan_jet_cs2.cut(start=0, end=end)

fig_cs2 = plot_wall_figure(
    scan_jet_cs2, spec_jet_cs2,
    scan_drop_cs2, spec_drop_cs2,
    spectrogram_ylim=[0, 120],
    xlim=[-250, 220],
    droplet_vline_x=-85,     #approximate position of the wall
    spectrogram_hline_y=22,
)
fig_cs2.savefig(savefig_folder + r"cs2_breaking_the_wall.pdf", format='pdf', dpi=300)


#====================================================================================================
#OCS - jet vs droplets (2026/04 data)
#====================================================================================================

OCS_DROPLETRADIUSMIN = 60
OCS_POSZEROSHIFT = 0  # millimetres :)
OCS_SPECTROGRAM_MAX = 0.7

#Loading jet data. GA=0mm, DA=16.6mm
folders_jet_ocs: list[Path] = []
configs_jet_ocs: list[IonDataAnalysisConfig] = []

folders_jet_ocs.append(Path(r"20260210\Scan3")) 
configs_jet_ocs.append(IonDataAnalysisConfig(
    delay_center= Length(92.654, Prefix.MILLI),
    center=Point(203, 202),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](30, 90),
    transform_parameter= 0.78))


#Loading droplet data. GA=0mm, DA=16.6mm
folders_droplets_ocs: list[Path] = []
configs_droplets_ocs: list[IonDataAnalysisConfig] = []

folders_droplets_ocs.append(Path(r"20260210\Scan4")) #better older data, GA=0, DA = 16.6mm, still has ~40GHz central oscillation frequency 
configs_droplets_ocs.append(IonDataAnalysisConfig(
    delay_center= Length(92.654, Prefix.MILLI),
    center=Point(175, 205),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](65, 120),
    transform_parameter= 0.75))

scan_jet_ocs, spec_jet_ocs = calculating(folders_jet_ocs, configs_jet_ocs)
scan_drop_ocs, spec_drop_ocs = calculating(folders_droplets_ocs, configs_droplets_ocs)

fig_ocs = plot_wall_figure(
    scan_jet_ocs, spec_jet_ocs,
    scan_drop_ocs, spec_drop_ocs,
    spectrogram_ylim=[0, 90],
    xlim=[-300, 300],
    spectrogram_vrange=Range(0, OCS_SPECTROGRAM_MAX),
    spectrogram_hline_y=36,
    droplet_vline_x=-50,
)
fig_ocs.savefig(savefig_folder + r"ocs_breaking_the_wall.pdf", format='pdf', dpi=300)


plt.show()
print('Done!')
