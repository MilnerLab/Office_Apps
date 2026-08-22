from pathlib import Path
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
POSZEROSHIFT = 0  # millimetres :)

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


#FUNCTION TO BUILD ONE FIGURE: jet on top row, droplets on bottom row,
#scan on the left column, spectrogram on the right column - same layout as breaking_the_wall.py.
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

    if spectrogram_hline_y is not None:
        ax_spec_drop.axhline(spectrogram_hline_y, color='w', linestyle='--', linewidth=1)

    if xlim is not None:
        ax_scan_jet.set_xlim(xlim)
        ax_spec_jet.set_xlim(xlim)

    ax_scan_jet.grid(True)
    ax_scan_drop.grid(True)

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
#CS2 - decelerating jet vs decelerating droplets
#====================================================================================================

CS2_DROPLETRADIUSMIN = 65

#Decelerating jet. Config from the (commented-out) "Decelerating Jet" block in CS2/CS2_jet_vs_droplets.py - verify by hand.
folders_jet_cs2: list[Path] = []
configs_jet_cs2: list[IonDataAnalysisConfig] = []

folders_jet_cs2.append(Path(r"20260429\Scan2"))
configs_jet_cs2.append(IonDataAnalysisConfig(
    delay_center=Length(93.3 - POSZEROSHIFT, Prefix.MILLI),
    center=Point(228, 186),
    angle=Angle(12, AngleUnit.DEG),
    analysis_zone=Range[int](40, 120),
    transform_parameter=0.78))

#Decelerating droplets. GA=0, DA=16.42. Config from CS2/CS2_directionality_droplets.py.
folders_droplets_cs2: list[Path] = []
configs_droplets_cs2: list[IonDataAnalysisConfig] = []

folders_droplets_cs2.append(Path(r"20260429\Scan1"))
configs_droplets_cs2.append(IonDataAnalysisConfig(
    delay_center=Length(93.3 - POSZEROSHIFT, Prefix.MILLI),
    center=Point(205, 194),
    angle=Angle(12, AngleUnit.DEG),
    analysis_zone=Range[int](CS2_DROPLETRADIUSMIN, 120),
    transform_parameter=0.78))

scan_jet_cs2, spec_jet_cs2 = calculating(folders_jet_cs2, configs_jet_cs2)
scan_drop_cs2, spec_drop_cs2 = calculating(folders_droplets_cs2, configs_droplets_cs2)

fig_cs2 = plot_wall_figure(
    scan_jet_cs2, spec_jet_cs2,
    scan_drop_cs2, spec_drop_cs2,
    spectrogram_ylim=[0, 120],
    xlim=[-200, 200],
)
fig_cs2.savefig(savefig_folder + r"cs2_deceleration.pdf", format='pdf', dpi=300)


#====================================================================================================
#OCS - decelerating jet vs decelerating droplets
#Note: the "jet" scan here is actually the same CS2 centrifuge scan used above (20260429\Scan2) -
#there was no comparable OCS jet deceleration scan, but it still shows the centrifuge deceleration.
#====================================================================================================

OCS_DROPLETRADIUSMIN = 65

#Same decelerating "jet" (CS2) scan/config as used in the CS2 figure above.
folders_jet_ocs: list[Path] = folders_jet_cs2
configs_jet_ocs: list[IonDataAnalysisConfig] = configs_jet_cs2

#Decelerating droplets. GA=0, DA=16.43. Config from ocs/OCS_directionality_droplets.py.
folders_droplets_ocs: list[Path] = []
configs_droplets_ocs: list[IonDataAnalysisConfig] = []

folders_droplets_ocs.append(Path(r"20260427\Scan2"))
configs_droplets_ocs.append(IonDataAnalysisConfig(
    delay_center=Length(93.3 - POSZEROSHIFT, Prefix.MILLI),
    center=Point(205, 194),
    angle=Angle(12, AngleUnit.DEG),
    analysis_zone=Range[int](OCS_DROPLETRADIUSMIN, 120),
    transform_parameter=0.78))

scan_jet_ocs, spec_jet_ocs = calculating(folders_jet_ocs, configs_jet_ocs)
scan_drop_ocs, spec_drop_ocs = calculating(folders_droplets_ocs, configs_droplets_ocs)

fig_ocs = plot_wall_figure(
    scan_jet_ocs, spec_jet_ocs,
    scan_drop_ocs, spec_drop_ocs,
    spectrogram_ylim=[0, 100],
    xlim=[-230, 230],
    spectrogram_vrange=Range(0, 0.4),
)
fig_ocs.savefig(savefig_folder + r"ocs_deceleration.pdf", format='pdf', dpi=300)


plt.show()
print('Done!')
