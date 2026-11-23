from dataclasses import dataclass, field, fields as dataclass_fields, replace as dataclass_replace
from enum import Enum
import json
from pathlib import Path
import sys

import matplotlib as mpl
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colorbar import make_axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.widgets import RangeSlider

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import extract_infos_from_name, load_ion_data
from _domain.plotting import plot_ScanData
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.scan_averaging.domain import averaging
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.models import ResampledScan
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from base_core.framework.events import EventBus
from base_core.lab_specifics.base_models import C2TScanData, IonDataAnalysisConfig, Measurement, ScanDataBase
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.quantities.enums import Prefix
from base_core.framework.serialization.serialization import from_primitive, to_primitive
from base_core.quantities.models import Length, Time
from base_qt.app.dispatcher import QtDispatcher
from base_qt.ui.dirty_indicator import DirtyIndicator
from base_qt.ui import LabMainWindow, MessageLevel, Panel, PanelViewModel
from base_qt.ui.apply import install_ui
from base_qt.ui.form import AngleSpec, BoolSpec, DirtyFormWidget, EnumSpec, FieldSpec, FloatSpec, IntSpec, LengthSpec, RangeSpec, TimeSpec


DEFAULT_FOLDER = r"Z:\Droplets\20261120\Scan1_CFG"
# Floor used when switching the spectrogram colour scale to log; the power data is
# normalized to a max of 1, so this is also the smallest value the slider can reach.
LOG_COLORSCALE_VMIN = 1e-3


class BaselineMode(Enum):
    NONE = "none"
    GAUSSIAN = "gaussian"
    MOVING_AVERAGE = "moving_average"


DEFAULT_BASELINE_MODE = BaselineMode.GAUSSIAN


class SpectrogramAveragingMode(Enum):
    # Spectrogram computed per individual repeat scan, then power-averaged across scans
    # with equal weight (the existing behaviour).
    PER_SCAN_POWER = "per_scan_power"
    # Spectrogram computed once, from the scans' averaged cos^2(theta)(t) curve.
    AVERAGED_CURVE = "averaged_curve"


DEFAULT_SPECTROGRAM_AVERAGING_MODE = SpectrogramAveragingMode.PER_SCAN_POWER

# How many remembered folders the Selection tab keeps before dropping the oldest
MAX_FOLDER_HISTORY = 20
# Colour assigned to each distinct group ID, cycling if there are more groups than colours
_GROUP_PALETTE = mpl.colormaps["tab10"].colors


def _group_colour(index: int):
    return _GROUP_PALETTE[index % len(_GROUP_PALETTE)]


@dataclass
class FolderEntry:
    path: str
    include: bool = True
    group_id: str = "1"
    # Numpy-style index/slice ("2:8", "::2", "-3:", "0,2,4") selecting which of this
    # folder's individual scans to use; empty means all scans.
    scan_selection: str = ""
    # Numpy-style index/slice, same syntax as scan_selection, but applied to the delay
    # axis within each of this folder's scans (replaces the old global cut_start/cut_end);
    # empty means no cut.
    cut_selection: str = ""


def _group_entries(entries: list[FolderEntry]) -> list[tuple[str, list[tuple[Path, str, str]]]]:
    # Folders sharing a group_id are combined (like the "folder_path.append" pattern in
    # script.py) into a single averaged curve, drawn in that group's own colour.
    groups: dict[str, list[tuple[Path, str, str]]] = {}
    order: list[str] = []
    for entry in entries:
        if not entry.include:
            continue
        if entry.group_id not in groups:
            groups[entry.group_id] = []
            order.append(entry.group_id)
        groups[entry.group_id].append((Path(entry.path), entry.scan_selection, entry.cut_selection))
    return [(group_id, groups[group_id]) for group_id in order]


def _parse_scan_selection(spec: str, n: int) -> list[int]:
    """Numpy-style index/slice applied to a sequence of length n, returning the selected
    0-based positions. Comma-separated terms are unioned, and each term is either a bare
    index ("4") or its own ":"-slice ("2:8", "::2", "-3:") -- so "0:20,60:70" and "0,2,4" and
    "0:5,8,12:15" ("[...]" brackets optional) all work, rather than the whole spec being
    forced into one slice or one flat index list."""
    spec = spec.strip()
    spec = spec[1:-1].strip() if spec.startswith("[") and spec.endswith("]") else spec
    idx = np.arange(n)
    if not spec:
        return idx.tolist()
    try:
        selected: list[int] = []
        for term in spec.split(","):
            term = term.strip()
            if not term:
                continue
            if ":" in term:
                parts = [p.strip() for p in term.split(":")]
                if len(parts) > 3:
                    raise ValueError("too many ':' in slice")
                selected.extend(idx[slice(*(int(p) if p else None for p in parts))].tolist())
            else:
                selected.append(int(idx[int(term)]))
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Invalid scan selection '{spec}': use numpy-style indexing, e.g. '0:5', '::2', '-3:', "
            "'0,2,4', or '0:20,60:70'."
        ) from e
    return np.atleast_1d(np.array(selected, dtype=int)).tolist()


def _group_files_by_position(files: list[Path]) -> dict:
    """Groups a folder's data files by delay position, same key load_ion_data uses, each
    sorted by its numeric time_part prefix -- the repeat loop runs once per position in the
    same order, so the rank of a file within its position's list is that file's repeat
    index, consistent across positions. (The time_part value itself is not a repeat index
    -- it does not line up across positions -- so filtering must go through this
    per-position rank; and it must be sorted numerically, not as a filename string, since
    e.g. "10DLY..." < "4DLY..." lexicographically despite 10 > 4.)"""
    by_position: dict = {}
    for p in files:
        time_part, stage_position = extract_infos_from_name(p)
        by_position.setdefault(stage_position, []).append((time_part, p))
    return {pos: [p for _, p in sorted(entries)] for pos, entries in by_position.items()}


def _filter_scan_files(files: list[Path], selection: str) -> list[Path]:
    """Keep only the files belonging to the repeat scans `selection` picks out of this
    folder, applying the index/slice to each delay position's own repeat rank."""
    selection = selection.strip()
    if not selection:
        return files
    kept: list[Path] = []
    for position_files in _group_files_by_position(files).values():
        keep = _parse_scan_selection(selection, len(position_files))
        kept.extend(position_files[i] for i in keep)
    return kept


def _count_scans(files: list[Path]) -> int:
    """Number of repeat scans present, i.e. the largest number of files any one delay
    position has (normally the same for every position)."""
    by_position = _group_files_by_position(files)
    return max((len(v) for v in by_position.values()), default=0)


def _split_into_repeats(files: list[Path]) -> list[list[Path]]:
    """Splits one folder's files into one file-list per repeat scan (one file per delay
    position each). load_ion_data merges every file it is given into a single combined
    curve -- averaging ion counts together before cos^2(theta) is even formed -- so
    without this split, "one scan" would mean "one folder's repeats pre-merged," not one
    actual repeat, and the per-scan spectrogram averaging downstream would really be
    averaging already-time-domain-merged curves under a different name."""
    by_position = _group_files_by_position(files)
    if not by_position:
        return []
    n_repeats = max(len(v) for v in by_position.values())
    repeats: list[list[Path]] = [[] for _ in range(n_repeats)]
    for position_files in by_position.values():
        for k, f in enumerate(position_files):
            repeats[k].append(f)
    return repeats


def _apply_cut_selection(scan: ScanDataBase, spec: str) -> ScanDataBase:
    """Numpy-style index/slice (same syntax as scan_selection) applied to one scan's delay
    axis, selecting which delay points to keep. Scans are frozen dataclasses, so this
    returns a new instance rather than mutating `scan`."""
    spec = spec.strip()
    if not spec:
        return scan
    keep = _parse_scan_selection(spec, len(scan.delays))
    delays = [scan.delays[i] for i in keep]
    measured_values = [scan.measured_values[i] for i in keep]
    if isinstance(scan, C2TScanData) and scan.ions_per_frame is not None:
        ions_per_frame = [scan.ions_per_frame[i] for i in keep]
        return dataclass_replace(scan, delays=delays, measured_values=measured_values, ions_per_frame=ions_per_frame)
    return dataclass_replace(scan, delays=delays, measured_values=measured_values)


@dataclass
class StftSettings:
    delay_center: Length = Length(125, Prefix.MILLI)
    center_x: float = 174.0
    center_y: float = 202.0
    angle: Angle = Angle(12, AngleUnit.DEG)
    analysis_zone: Range[int] = Range[int](60, 120)
    transform_parameter: float = 0.73
    window_size: Time = Time(180, Prefix.PICO)
    # Whether the spectrogram is the equal-weight power average of each individual repeat
    # scan's own spectrogram, or computed once from the scans' averaged cos^2(theta)(t)
    # curve instead (see SpectrogramAveragingMode).
    spectrogram_averaging_mode: SpectrogramAveragingMode = DEFAULT_SPECTROGRAM_AVERAGING_MODE
    # Baseline subtracted from the averaged scan before the STFT (see ResampledScan.detrend*)
    baseline_mode: BaselineMode = DEFAULT_BASELINE_MODE
    # Only used while baseline_mode is MOVING_AVERAGE
    moving_average_window: Time = Time(10, Prefix.PICO)
    # Plotting only: shows the baseline that baseline_mode subtracts, instead of the scan
    # curve itself; does not affect the STFT, which always uses the detrended data.
    plot_detrend_curve: bool = False
    # Plotting only; does not affect the underlying data
    x_limits_enabled: bool = False
    x_limits: Range[float] = Range[float](-250.0, 250.0)
    y_limits_enabled: bool = False
    y_limits: Range[float] = Range[float](0.0, 120.0)
    log_colorscale: bool = False
    # Most-recent-first history of folders plotted, shown and edited in the Selection tab
    folder_entries: list[FolderEntry] = field(default_factory=list)


# A separate, user-editable settings snapshot persisted next to this script. The "Save
# config" / "Load config" GUI buttons write/read it; it does NOT replace StftSettings'
# own field defaults above.
SIDECAR_CONFIG_PATH = Path(__file__).with_name("stft_analysis_config.json")
SIDECAR_CONFIG_DEFAULT = StftSettings()


def save_settings_to_sidecar(settings: StftSettings, path: Path = SIDECAR_CONFIG_PATH) -> None:
    path.write_text(json.dumps(to_primitive(settings), indent=2), encoding="utf-8")


def load_settings_from_sidecar(path: Path = SIDECAR_CONFIG_PATH) -> StftSettings:
    return from_primitive(StftSettings, json.loads(path.read_text(encoding="utf-8")))


def build_config(settings: StftSettings) -> IonDataAnalysisConfig:
    return IonDataAnalysisConfig(
        delay_center=settings.delay_center,
        center=Point(settings.center_x, settings.center_y),
        angle=settings.angle,
        analysis_zone=settings.analysis_zone,
        transform_parameter=settings.transform_parameter,
    )


def _moving_average_points(window: Time, sample_spacing: Time) -> int:
    """detrend_moving_average() takes its window as a sample count; the UI exposes it as a
    duration, so convert using the spacing the data is actually sampled at. Time is stored
    in SI seconds, so a plain float ratio is the number of samples that duration spans."""
    return max(1, round(float(window) / float(sample_spacing)))


def _scan_sample_spacing(scan: ScanDataBase) -> Time:
    diffs = [b - a for a, b in zip(scan.delays, scan.delays[1:])]
    if not diffs:
        return Time(1, Prefix.PICO)
    return Time(min(float(d) for d in diffs))


def _baseline_curve(scan: ScanDataBase, baseline_mode: BaselineMode, moving_average_window: Time) -> ScanDataBase:
    """The curve baseline_mode would subtract from `scan`, as its own ScanDataBase (same
    delays, zero error) so it can be plotted with plot_ScanData in place of the real data."""
    if baseline_mode == BaselineMode.NONE:
        baseline = [0.0] * len(scan.delays)
    else:
        wrapped = ResampledScan(
            delays=scan.delays,
            measured_values=scan.measured_values,
            run_id=scan.run_id,
            ions_per_point=scan.ions_per_point,
        )
        if baseline_mode == BaselineMode.MOVING_AVERAGE:
            window_points = _moving_average_points(moving_average_window, _scan_sample_spacing(scan))
            _, baseline = wrapped.detrend_moving_average(window_points)
        else:
            _, baseline = wrapped.detrend()
    return ScanDataBase(
        delays=scan.delays,
        measured_values=[Measurement(b, 0.0) for b in baseline],
        run_id=scan.run_id,
    )


@dataclass
class StftComputeResult:
    # One entry per plotted group: (legend label, line colour, scan curve, baseline curve
    # or None). The baseline curve -- what baseline_mode would subtract -- is plotted
    # alongside the scan curve, not in place of it.
    group_plots: list[tuple[str, tuple, object, object | None]]
    first_group_average: object
    spectrogram: object


def compute_stft_data(
    groups: list[tuple[str, list[tuple[Path, str, str]]]],
    config: IonDataAnalysisConfig,
    window_size: Time,
    baseline_mode: BaselineMode = DEFAULT_BASELINE_MODE,
    moving_average_window: Time = Time(10, Prefix.PICO),
    plot_detrend_curve: bool = False,
    spectrogram_averaging_mode: SpectrogramAveragingMode = DEFAULT_SPECTROGRAM_AVERAGING_MODE,
    progress_cb=lambda done, total, message: None,
) -> StftComputeResult:
    """Pure data computation (file IO + numpy/scipy) -- no matplotlib/Qt calls, so this is
    safe to run on a background thread while the GUI stays responsive."""
    if not groups:
        raise ValueError("No folders selected. Check at least one folder in the Selection tab.")

    # Discover every group's repeats up front so the progress bar has a real, finely
    # grained total to work against -- the per-repeat file reads below (not this scan)
    # are the slow part, so each one completing is what should move the bar. Repeats are
    # kept split out by folder (not flattened across the whole group) since each folder
    # carries its own cut_selection, applied after that folder's own pipeline run below.
    progress_cb(0, 1, "Scanning folders...")
    group_file_paths: list[list[list[Path]]] = []
    group_folder_repeats: list[list[list[list[Path]]]] = []
    for group_id, folder_entries in groups:
        folders = [path for path, _, _ in folder_entries]
        file_paths = DatFinder(folders, is_full_path=True).find_datafiles()
        file_paths = [
            _filter_scan_files(files, selection)
            for files, (_, selection, _) in zip(file_paths, folder_entries)
        ]
        group_file_paths.append(file_paths)
        group_folder_repeats.append([_split_into_repeats(files) for files in file_paths])

    total_repeats = sum(
        len(repeats) for folder_repeats in group_folder_repeats for repeats in folder_repeats
    )
    total_steps = total_repeats + len(groups) + 1  # +1 for the final spectrogram/render step
    done = 0

    # Folders in the same group are combined into one averaged curve -- but that averaged
    # curve is for the cos^2(theta)(t) plot only. The spectrogram is computed per
    # *individual repeat scan* (across every folder, every group) and only averaged
    # afterwards, since the STFT's power is a nonlinear function of the signal: averaging
    # traces together before the transform can cancel out per-scan oscillation phase
    # differences that averaging power spectrograms after the transform would not.
    # load_ion_data merges every file it is handed into one combined curve, so each
    # folder's repeats are split out (see _split_into_repeats) and run through the c2t
    # pipeline separately -- one RawScanData/C2TScanData per repeat, not per folder --
    # otherwise "per-scan" here would really mean "per folder, repeats pre-merged."
    group_plots = []
    group_averages = []
    all_individual_scans = []
    for i, (group_id, folder_entries) in enumerate(groups):
        file_paths = group_file_paths[i]
        folder_repeats = group_folder_repeats[i]
        # Counts repeat scans actually used (after scan_selection filtering).
        num_scans = sum(_count_scans(files) for files in file_paths)

        group_repeat_scans: list[C2TScanData] = []
        for (_, _, cut_selection), repeat_file_lists in zip(folder_entries, folder_repeats):
            raw_scans = []
            for j, repeat in enumerate(repeat_file_lists):
                progress_cb(
                    done, total_steps,
                    f"Loading group {group_id}: scan {j + 1}/{len(repeat_file_lists)}...",
                )
                raw_scans.extend(load_ion_data([repeat]))
                done += 1

            calculated_scans = run_pipeline(raw_scans, config)
            group_repeat_scans.extend(
                _apply_cut_selection(c2t_scan, cut_selection) for c2t_scan in calculated_scans
            )

        progress_cb(done, total_steps, f"Processing group {group_id} ({i + 1}/{len(groups)})...")
        averaged_data = averaging.average_scans(group_repeat_scans)
        group_averages.append(averaged_data)
        done += 1

        colour = _group_colour(i)
        label = f"ID {group_id} ({num_scans} scans)"
        baseline_data = _baseline_curve(averaged_data, baseline_mode, moving_average_window) if plot_detrend_curve else None
        group_plots.append((label, colour, averaged_data, baseline_data))
        all_individual_scans.extend(group_repeat_scans)

    progress_cb(done, total_steps, "Computing spectrogram...")
    stft_config = StftAnalysisConfig(all_individual_scans, window_size)
    # PER_SCAN_POWER: one spectrogram per individual repeat scan, equal-weight power
    # averaged afterwards (see the note above on why that order matters).
    # AVERAGED_CURVE: scans are averaged into a single cos^2(theta)(t) curve first (same
    # averaging.average_scans used for the plotted curve), and the STFT runs once on that.
    # calculate_averaged_spectrogram handles both: with a single input scan, "averaging"
    # is a no-op, so it still applies the same max-normalisation as the multi-scan case.
    if spectrogram_averaging_mode == SpectrogramAveragingMode.AVERAGED_CURVE:
        scans_for_stft = averaging.average_scans(all_individual_scans)
    else:
        scans_for_stft = all_individual_scans
    resampled_scans = resample_scans(scans_for_stft, stft_config.axis)
    moving_average_points = _moving_average_points(moving_average_window, stft_config.resample_time)
    spectrogram = StftAnalysis(resampled_scans, stft_config).calculate_averaged_spectrogram(
        detrend_mode=baseline_mode.value,
        moving_average_window=moving_average_points,
    )
    progress_cb(total_steps, total_steps, "Rendering...")
    return StftComputeResult(group_plots, group_averages[0], spectrogram)


def render_stft_plot(ax_scan, ax_spec, ax_ions, result: StftComputeResult):
    """Matplotlib-only: must run on the GUI thread, after compute_stft_data finishes."""
    ax_ions.yaxis.tick_right()
    ax_ions.yaxis.set_label_position("right")

    for label, colour, averaged_data, baseline_data in result.group_plots:
        plot_ScanData(ax_scan, data=averaged_data, label=label, ax_twin=None, color=colour, ecolor=colour)
        if baseline_data is not None:
            x = [t.value(Prefix.PICO) for t in baseline_data.delays]
            y = [v.value for v in baseline_data.measured_values]
            ax_scan.plot(x, y, linestyle="--", color=colour, label=f"{label} baseline")

    ax_scan.legend(loc="best")
    ax_scan.grid(visible=True, which="major", alpha=1.0)

    mesh = plot_Spectrogram(ax_spec, result.spectrogram, v_range=Range(0, 1), shading="auto")
    plot_nyquist_frequency(ax_spec, result.first_group_average)

    ax_spec.set_axisbelow(False)
    ax_spec.grid(visible=True, color="grey", linewidth=0.3, zorder=5)

    ax_ions.tick_params(axis="y", left=False, right=True, labelleft=False)
    ax_ions.grid(False)
    return mesh


class _StftWorker(QThread):
    """Runs compute_stft_data off the GUI thread and reports progress back to it."""

    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(object)

    def __init__(
        self, groups, config, window_size, baseline_mode, moving_average_window,
        plot_detrend_curve, spectrogram_averaging_mode, parent=None,
    ) -> None:
        super().__init__(parent)
        self._groups = groups
        self._config = config
        self._window_size = window_size
        self._baseline_mode = baseline_mode
        self._moving_average_window = moving_average_window
        self._plot_detrend_curve = plot_detrend_curve
        self._spectrogram_averaging_mode = spectrogram_averaging_mode

    def run(self) -> None:
        try:
            result = compute_stft_data(
                self._groups,
                self._config,
                self._window_size,
                self._baseline_mode,
                self._moving_average_window,
                self._plot_detrend_curve,
                self._spectrogram_averaging_mode,
                progress_cb=lambda done, total, message: self.progress.emit(done, total, message),
            )
        except Exception as e:  # noqa: BLE001 -- reported to the GUI thread, not swallowed
            self.failed.emit(e)
        else:
            self.succeeded.emit(result)


# ============================================================
# GUI
# ============================================================
class FolderSelectionWidget(QWidget):
    """One row per remembered folder: include checkbox, path, and a colour-group ID.

    Edits are staged on a local clone of the entries and only written back to the real
    config (via commit_pending) when the form's Apply button is pressed -- otherwise every
    checkbox toggle or ID edit would trigger an immediate, expensive replot.
    """

    def __init__(self, on_dirty, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._on_dirty = on_dirty
        self._entries: list[FolderEntry] = []
        self._indicators: list[DirtyIndicator] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        hint = QLabel(
            "Check to include a folder; folders sharing an ID are combined into one curve of "
            "that ID's colour. \"Scans\" picks specific scans from a folder with numpy-style "
            "indexing (e.g. 0:5, ::2, -3:, 0,2,4); leave blank for all scans. \"Cut\" uses the "
            "same syntax to pick which delay points within each of this folder's scans to keep. "
            "Press Apply to update the plot."
        )
        hint.setWordWrap(True)
        outer.addWidget(hint)

        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._rows_container)
        outer.addWidget(scroll, stretch=1)

    def set_entries(self, entries: list[FolderEntry]) -> None:
        # Clone so edits don't touch the real config until commit_pending() is called.
        self._entries = [
            FolderEntry(e.path, e.include, e.group_id, e.scan_selection, e.cut_selection) for e in entries
        ]
        self._indicators = []
        while self._rows_layout.count() > 1:  # keep the trailing stretch
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for entry in self._entries:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            indicator = DirtyIndicator()
            self._indicators.append(indicator)

            checkbox = QCheckBox()
            checkbox.setChecked(entry.include)
            checkbox.toggled.connect(lambda checked, e=entry, ind=indicator: self._set_include(e, checked, ind))

            label = QLabel()
            label.setFixedWidth(120)
            label.setText(QFontMetrics(label.font()).elidedText(entry.path, Qt.TextElideMode.ElideMiddle, label.width()))
            label.setToolTip(entry.path)

            id_edit = QLineEdit(entry.group_id)
            id_edit.setFixedWidth(32)
            id_edit.setToolTip("Colour-group ID: folders sharing an ID are combined")
            id_edit.editingFinished.connect(lambda e=entry, w=id_edit, ind=indicator: self._set_group_id(e, w.text(), ind))

            scan_edit = QLineEdit(entry.scan_selection)
            scan_edit.setFixedWidth(50)
            scan_edit.setPlaceholderText("all")
            scan_edit.setToolTip("Numpy-style scan index/slice, e.g. 0:5, ::2, -3:, 0,2,4")
            scan_edit.editingFinished.connect(lambda e=entry, w=scan_edit, ind=indicator: self._set_scan_selection(e, w.text(), ind))

            cut_edit = QLineEdit(entry.cut_selection)
            cut_edit.setFixedWidth(50)
            cut_edit.setPlaceholderText("all")
            cut_edit.setToolTip("Numpy-style delay-point index/slice within each scan, e.g. 5:-5, ::2")
            cut_edit.editingFinished.connect(lambda e=entry, w=cut_edit, ind=indicator: self._set_cut_selection(e, w.text(), ind))

            row_layout.addWidget(indicator)
            row_layout.addWidget(checkbox)
            row_layout.addWidget(label)
            row_layout.addStretch(1)
            row_layout.addWidget(id_edit)
            row_layout.addWidget(scan_edit)
            row_layout.addWidget(cut_edit)
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)

    def _set_include(self, entry: FolderEntry, checked: bool, indicator: DirtyIndicator) -> None:
        if entry.include != checked:
            entry.include = checked
            indicator.set_dirty(True)
            self._on_dirty()

    def _set_group_id(self, entry: FolderEntry, text: str, indicator: DirtyIndicator) -> None:
        text = text.strip() or "1"
        if entry.group_id != text:
            entry.group_id = text
            indicator.set_dirty(True)
            self._on_dirty()

    def _set_scan_selection(self, entry: FolderEntry, text: str, indicator: DirtyIndicator) -> None:
        text = text.strip()
        if entry.scan_selection != text:
            entry.scan_selection = text
            indicator.set_dirty(True)
            self._on_dirty()

    def _set_cut_selection(self, entry: FolderEntry, text: str, indicator: DirtyIndicator) -> None:
        text = text.strip()
        if entry.cut_selection != text:
            entry.cut_selection = text
            indicator.set_dirty(True)
            self._on_dirty()

    def commit_pending(self, config_entries: list[FolderEntry]) -> None:
        """Write staged include/group_id/scan_selection/cut_selection edits back onto the
        real entries, matched by path."""
        staged_by_path = {e.path: e for e in self._entries}
        for entry in config_entries:
            staged = staged_by_path.get(entry.path)
            if staged is not None:
                entry.include = staged.include
                entry.group_id = staged.group_id
                entry.scan_selection = staged.scan_selection
                entry.cut_selection = staged.cut_selection
        for indicator in self._indicators:
            indicator.set_dirty(False)


class StftForm(DirtyFormWidget):
    _specs: dict[str, FieldSpec] = {
        "delay_center": LengthSpec("Delay center", Prefix.MILLI, extra_decimals=1),
        "center_x": FloatSpec("Center x", min=-10_000.0, max=10_000.0, decimals=2, step=1.0),
        "center_y": FloatSpec("Center y", min=-10_000.0, max=10_000.0, decimals=2, step=1.0),
        "angle": AngleSpec("Angle"),
        "analysis_zone": RangeSpec("Zone", IntSpec("", min=0, max=10_000)),
        "transform_parameter": FloatSpec("ScaleX", min=0.0, max=10.0, decimals=3, step=0.01),
        "window_size": TimeSpec("STFT window", Prefix.PICO, min=0.0, max=10_000.0),
        "spectrogram_averaging_mode": EnumSpec("Spectrogram from", SpectrogramAveragingMode),
        "baseline_mode": EnumSpec("Baseline subtraction", BaselineMode),
        "moving_average_window": TimeSpec("Moving avg. window", Prefix.PICO, min=0.0, max=10_000.0),
        "plot_detrend_curve": BoolSpec("Plot subtracted baseline"),
        "x_limits_enabled": BoolSpec("Limit x axis"),
        "x_limits": RangeSpec("x range (ps)", FloatSpec("", min=-1_000_000.0, max=1_000_000.0, decimals=1, step=1.0)),
        "y_limits_enabled": BoolSpec("Limit y axis"),
        "y_limits": RangeSpec("y range (GHz)", FloatSpec("", min=0.0, max=1_000_000.0, decimals=1, step=1.0)),
        "log_colorscale": BoolSpec("Logarithmic colour scale"),
    }
    _groups = [
        ("Analysis config", ["delay_center", "center_x", "center_y", "angle", "analysis_zone", "transform_parameter"]),
        ("STFT window", ["window_size", "spectrogram_averaging_mode"]),
        ("Baseline subtraction", ["baseline_mode", "moving_average_window", "plot_detrend_curve"]),
        ("Plotting", ["x_limits_enabled", "x_limits", "y_limits_enabled", "y_limits", "log_colorscale"]),
    ]
    # Tabs group the sections above: (tab title, group titles shown on that tab)
    _tabs = [
        ("Analysis config", ["Analysis config"]),
        ("STFT config", ["STFT window", "Baseline subtraction"]),
        ("Plotting", ["Plotting"]),
    ]

    def _build(self) -> None:
        super()._build()
        # super()._build() added one QGroupBox per _groups entry, in order, directly to
        # form_layout; take them back out and sort them into tabs instead.
        containers = {}
        group_titles = [title for title, _ in self._groups]
        for title in group_titles:
            container = self.form_layout.takeAt(0).widget()
            container.setStyleSheet("QGroupBox { background-color: #e8e8e8; }")
            containers[title] = container

        tabs = QTabWidget()
        tabs.setStyleSheet("QTabBar::tab { color: red; }")
        for tab_title, titles in self._tabs:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(4, 4, 4, 4)
            for title in titles:
                page_layout.addWidget(containers[title])
            page_layout.addStretch(1)
            tabs.addTab(page, tab_title)

        # Not spec-driven like the tabs above (its rows are dynamic, added as folders are
        # plotted); edits there are staged locally and only committed to the config below,
        # when Apply is pressed, same as every other field on this form.
        self._selection_widget = FolderSelectionWidget(on_dirty=lambda: None)
        self._selection_widget.set_entries(self._config.folder_entries)
        tabs.addTab(self._selection_widget, "Selection")

        self.form_layout.addWidget(tabs)

    def refresh_selection(self) -> None:
        self._selection_widget.set_entries(self._config.folder_entries)

    def __init__(self, settings: StftSettings, on_apply, parent: QWidget | None = None) -> None:
        super().__init__(settings, parent)
        self._on_apply = on_apply

    def on_apply(self) -> None:
        self._selection_widget.commit_pending(self._config.folder_entries)
        self._on_apply()


class StftViewModel(PanelViewModel):
    def __init__(self, bus: EventBus, dispatcher: QtDispatcher, settings: StftSettings, folder: Path) -> None:
        super().__init__(bus, dispatcher)
        self.settings = settings
        self.folder = folder

    def report_error(self, e: Exception) -> None:
        self._msg(f"Error: {e}", MessageLevel.ERROR)

    # ============================================================
    # Sidecar config
    # ============================================================
    def save_sidecar_config(self) -> None:
        save_settings_to_sidecar(self.settings)
        self._msg(f"Saved config to: {SIDECAR_CONFIG_PATH}")

    def load_sidecar_config(self) -> frozenset[str]:
        loaded = load_settings_from_sidecar()
        names = [f.name for f in dataclass_fields(StftSettings)]
        for name in names:
            setattr(self.settings, name, getattr(loaded, name))
        self._msg(f"Loaded config from: {SIDECAR_CONFIG_PATH}")
        return frozenset(names)


class StftPanel(Panel):
    def __init__(self, vm: StftViewModel, parent: QWidget | None = None) -> None:
        super().__init__("STFT analysis", vm, parent)

    def setup(self) -> None:
        self._fig = Figure(figsize=(8, 5))
        self._ax_scan, self._ax_spec = self._fig.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0})
        self._fig.subplots_adjust(top=0.94, left=0.08, right=0.97, bottom=0.12)
        self._ax_scan.xaxis.label.set_visible(False)
        self._ax_scan.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        self._ax_spec.tick_params(axis="x", which="both", direction="out")
        # A colour-limit slider stands where the colorbar would. Its axes is made once,
        # after subplots_adjust, so replots do not keep shrinking the plot to make room.
        # make_axes only narrows ax_spec, so ax_scan must be narrowed to match or its
        # (wider) x-axis no longer lines up with ax_spec's despite sharing x values.
        # fraction/pad are shrunk from their (generous) defaults, which otherwise reserve
        # a wide gap between the spectrogram and the colorbar for no visible benefit here.
        self._cax, _ = make_axes(self._ax_spec, fraction=0.08, pad=0.015)
        self._cax.set_axis_off()
        pos_spec = self._ax_spec.get_position()
        pos_scan = self._ax_scan.get_position()
        self._ax_scan.set_position([pos_spec.x0, pos_scan.y0, pos_spec.width, pos_scan.height])
        self._ax_ions = self._ax_scan.twinx()
        self._clim_slider: RangeSlider | None = None
        # Handle positions as fractions of the colour range, kept across replots
        self._clim_frac: tuple[float, float] = (0.0, 1.0)
        self._canvas = FigureCanvasQTAgg(self._fig)

        plot_col = QVBoxLayout()
        plot_col.addWidget(NavigationToolbar2QT(self._canvas, self))
        plot_col.addWidget(self._canvas, stretch=1)

        # Shown only while a replot's data loading/computation is running in the background
        # (see _StftWorker), so a slow load reads as "in progress" rather than a hang.
        self._progress_label = QLabel()
        self._progress_label.setVisible(False)
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        progress_row = QHBoxLayout()
        progress_row.addWidget(self._progress_label)
        progress_row.addWidget(self._progress_bar, stretch=1)
        plot_col.addLayout(progress_row)

        self._worker: _StftWorker | None = None
        self._replot_pending = False
        self._active_groups: list[tuple[str, list[tuple[Path, str]]]] = []
        self._active_settings: StftSettings | None = None

        # Seed the Selection tab with the starting folder, so the initial plot has
        # something checked in before the form (which reads folder_entries) is built.
        self._add_folder_to_history(self.vm.folder)

        # Edits stay pending (dirty) until Apply, which writes them to the settings and replots
        self._form = StftForm(self.vm.settings, self._replot)
        save_config_button = QPushButton("Save config")
        save_config_button.clicked.connect(self._on_save_config)
        load_config_button = QPushButton("Load config")
        load_config_button.clicked.connect(self._on_load_config)
        apply_row = QHBoxLayout()
        apply_row.addStretch(1)
        apply_row.addWidget(save_config_button)
        apply_row.addWidget(load_config_button)
        apply_row.addWidget(self._form.apply_button)
        side_panel = QWidget()
        side_col = QVBoxLayout(side_panel)
        side_col.setContentsMargins(0, 0, 0, 0)
        side_col.addWidget(self._form)
        side_col.addLayout(apply_row)
        side_col.addStretch(1)
        # Fixed, because the spin boxes' huge ranges make their size hints far too wide
        side_panel.setFixedWidth(400)

        plot_row = QHBoxLayout()
        plot_row.addLayout(plot_col, stretch=1)
        plot_row.addWidget(side_panel)

        self._folder_edit = QLineEdit(str(self.vm.folder))
        self._folder_edit.textChanged.connect(self._on_folder_changed)
        self._folder_edit.returnPressed.connect(self._commit_folder)
        self._browse_button = QPushButton("Browse")
        self._browse_button.clicked.connect(self._on_browse_folder)
        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(self._commit_folder)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(QLabel("Folder"))
        bottom_row.addWidget(self._folder_edit, stretch=1)
        bottom_row.addWidget(self._browse_button)
        bottom_row.addWidget(self._refresh_button)

        self.body_layout.addLayout(plot_row, stretch=1)
        self.body_layout.addLayout(bottom_row)
        self._replot()

    def _show_error(self, e: Exception) -> None:
        self._ax_scan.set_title(f"Error: {e}", color="r")
        self._canvas.draw_idle()
        self.vm.report_error(e)

    def _clear_clim_slider(self) -> None:
        if self._clim_slider is not None:
            self._clim_slider.disconnect_events()
            self._clim_slider = None
        self._cax.clear()
        self._cax.set_axis_off()

    def _build_clim_slider(self, mesh) -> None:
        # A colorbar whose two handles set the mesh's clim. The gradient shares the mesh's
        # norm, so it shows the same clipping as the map while the handles move.
        lo, hi = float(mesh.norm.vmin), float(mesh.norm.vmax)
        span = hi - lo
        self._cax.imshow(
            np.linspace(lo, hi, 256).reshape(-1, 1),
            extent=(0.0, 1.0, lo, hi),
            origin="lower",
            aspect="auto",
            cmap=mesh.cmap,
            norm=mesh.norm,
        )
        frac = self._clim_frac
        init = (lo + frac[0] * span, lo + frac[1] * span)
        slider = RangeSlider(
            self._cax, "", lo, hi,
            valinit=init,
            orientation="vertical",
            valfmt="%.3g",
            track_color="none",
            facecolor="none",
            edgecolor="white",
            linewidth=1.0,
        )
        # RangeSlider hides its axes; bring back a colorbar-style value axis on the right
        self._cax.set_axis_on()
        self._cax.set_xlim(0.0, 1.0)
        self._cax.set_xticks([])
        self._cax.yaxis.set_ticks_position("right")
        self._cax.yaxis.set_label_position("right")
        self._cax.set_ylabel("power")

        def on_changed(val) -> None:
            vmin, vmax = val
            mesh.set_clim(vmin, vmax)
            if span > 0:
                self._clim_frac = ((vmin - lo) / span, (vmax - lo) / span)

        slider.on_changed(on_changed)
        mesh.set_clim(*slider.val)
        # Keep a reference, or the slider is garbage collected and stops responding
        self._clim_slider = slider

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._progress_label.setVisible(busy)
        self._progress_bar.setVisible(busy)
        self._progress_label.setText(message)
        if not busy:
            self._progress_bar.setValue(0)
        self._form.setEnabled(not busy)
        self._browse_button.setEnabled(not busy)
        self._refresh_button.setEnabled(not busy)
        self._folder_edit.setEnabled(not busy)

    def _replot(self) -> None:
        if self._worker is not None:
            # A load/compute is already running; run again once it finishes.
            self._replot_pending = True
            return

        settings = self.vm.settings
        groups = _group_entries(settings.folder_entries)
        if not groups:
            self._show_error(ValueError("No folders selected. Check at least one folder in the Selection tab."))
            return

        self._active_groups = groups
        self._active_settings = settings
        self._set_busy(True, "Starting...")

        worker = _StftWorker(
            groups,
            build_config(settings),
            settings.window_size,
            settings.baseline_mode,
            settings.moving_average_window,
            settings.plot_detrend_curve,
            settings.spectrogram_averaging_mode,
            parent=self,
        )
        worker.progress.connect(self._on_compute_progress)
        worker.succeeded.connect(self._on_compute_succeeded)
        worker.failed.connect(self._on_compute_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_compute_progress(self, done: int, total: int, message: str) -> None:
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(done)
        self._progress_label.setText(message)

    def _on_compute_succeeded(self, result: StftComputeResult) -> None:
        groups = self._active_groups
        settings = self._active_settings
        self._clear_clim_slider()
        self._ax_scan.clear()
        self._ax_spec.clear()
        self._ax_ions.clear()
        # clear() resets tick/label visibility too, so the top and bottom plots must be
        # re-pinned together (shared x-axis, no gap) after every replot, not just in setup
        self._ax_scan.xaxis.label.set_visible(False)
        self._ax_scan.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        self._ax_spec.tick_params(axis="x", which="both", direction="out")
        mesh = render_stft_plot(self._ax_scan, self._ax_spec, self._ax_ions, result)
        if settings.x_limits_enabled:
            self._ax_spec.set_xlim(settings.x_limits.min, settings.x_limits.max)
        if settings.y_limits_enabled:
            self._ax_spec.set_ylim(settings.y_limits.min, settings.y_limits.max)
        mesh.set_norm(LogNorm(vmin=LOG_COLORSCALE_VMIN, vmax=1.0) if settings.log_colorscale else Normalize(vmin=0.0, vmax=1.0))
        self._build_clim_slider(mesh)
        self._fig.suptitle(", ".join(f"ID {group_id}" for group_id, _ in groups), fontsize=10)
        self._canvas.draw_idle()

    def _on_compute_failed(self, e: Exception) -> None:
        self._show_error(e)

    def _on_worker_finished(self) -> None:
        self._worker = None
        self._set_busy(False)
        if self._replot_pending:
            self._replot_pending = False
            self._replot()

    def _on_save_config(self) -> None:
        try:
            self.vm.save_sidecar_config()
        except Exception as e:
            self._show_error(e)

    def _on_load_config(self) -> None:
        try:
            names = self.vm.load_sidecar_config()
        except Exception as e:
            self._show_error(e)
            return
        self._form.refresh_fields(names)
        self._form.refresh_selection()
        self._replot()

    def _on_folder_changed(self, text: str) -> None:
        self.vm.folder = Path(text.strip())

    def _on_browse_folder(self) -> None:
        folder = self.vm.folder
        initial_dir = str(folder) if folder.exists() else ""
        selected = QFileDialog.getExistingDirectory(self, "Select scan folder", initial_dir)
        if not selected:
            return
        self._folder_edit.setText(str(Path(selected)))
        self._commit_folder()

    def _commit_folder(self) -> None:
        # Enter in the path field, Browse, or Refresh: all three mean "plot this folder
        # too", so each remembers it in the Selection tab's history before replotting.
        self._add_folder_to_history(self.vm.folder)
        self._form.refresh_selection()
        self._replot()

    def _add_folder_to_history(self, folder: Path) -> None:
        entries = self.vm.settings.folder_entries
        path_str = str(folder)
        existing = next((e for e in entries if e.path == path_str), None)
        if existing is not None:
            # Already known: just bring it back to the top, keeping its include/ID as-is
            entries.remove(existing)
            entries.insert(0, existing)
        else:
            used_ids = {e.group_id for e in entries}
            next_id = 1
            while str(next_id) in used_ids:
                next_id += 1
            entries.insert(0, FolderEntry(path=path_str, include=True, group_id=str(next_id)))
        del entries[MAX_FOLDER_HISTORY:]


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    install_ui(app)
    bus = EventBus()
    dispatcher = QtDispatcher()

    window = LabMainWindow("STFT analysis", bus, dispatcher)
    vm = StftViewModel(bus, dispatcher, StftSettings(), Path(DEFAULT_FOLDER))
    window.setCentralWidget(StftPanel(vm))
    window.resize(1400, 700)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
