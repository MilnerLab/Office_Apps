from dataclasses import dataclass, field, fields as dataclass_fields
from enum import Enum
import json
from pathlib import Path
import sys

import matplotlib as mpl
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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

from _data_io.dat_finder import MOST_RECENT_FOLDER, DatFinder
from _data_io.dat_loader import load_ion_data
from _domain.plotting import plot_ScanData
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.scan_averaging.domain import averaging
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from base_core.framework.events import EventBus
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.quantities.enums import Prefix
from base_core.framework.serialization.serialization import from_primitive, to_primitive
from base_core.quantities.models import Length, Time
from base_qt.app.dispatcher import QtDispatcher
from base_qt.ui import LabMainWindow, MessageLevel, Panel, PanelViewModel
from base_qt.ui.apply import install_ui
from base_qt.ui.form import AngleSpec, BoolSpec, DirtyFormWidget, EnumSpec, FieldSpec, FloatSpec, IntSpec, LengthSpec, RangeSpec, TimeSpec


DEFAULT_FOLDER = r"Z:\Droplets\20261119\Scan7_CFG"
# Floor used when switching the spectrogram colour scale to log; the power data is
# normalized to a max of 1, so this is also the smallest value the slider can reach.
LOG_COLORSCALE_VMIN = 1e-3


class BaselineMode(Enum):
    NONE = "none"
    GAUSSIAN = "gaussian"
    MOVING_AVERAGE = "moving_average"


DEFAULT_BASELINE_MODE = BaselineMode.GAUSSIAN

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


def _group_entries(entries: list[FolderEntry]) -> list[tuple[str, list[Path]]]:
    # Folders sharing a group_id are combined (like the "folder_path.append" pattern in
    # script.py) into a single averaged curve, drawn in that group's own colour.
    groups: dict[str, list[Path]] = {}
    order: list[str] = []
    for entry in entries:
        if not entry.include:
            continue
        if entry.group_id not in groups:
            groups[entry.group_id] = []
            order.append(entry.group_id)
        groups[entry.group_id].append(Path(entry.path))
    return [(group_id, groups[group_id]) for group_id in order]


@dataclass
class StftSettings:
    delay_center: Length = Length(125, Prefix.MILLI)
    center_x: float = 174.0
    center_y: float = 202.0
    angle: Angle = Angle(12, AngleUnit.DEG)
    analysis_zone: Range[int] = Range[int](60, 120)
    transform_parameter: float = 0.73
    window_size: Time = Time(180, Prefix.PICO)
    # Indices into the averaged scan's delays; 0 means no cut on that side (see ScanDataBase.cut)
    cut_start: int = 0
    cut_end: int = 0
    # Baseline subtracted from the averaged scan before the STFT (see ResampledScan.detrend*)
    baseline_mode: BaselineMode = DEFAULT_BASELINE_MODE
    # Only used while baseline_mode is MOVING_AVERAGE
    moving_average_window: int = 10
    # Plotting only; does not affect the underlying data
    x_limits_enabled: bool = False
    x_limits: Range[float] = Range[float](-100.0, 100.0)
    y_limits_enabled: bool = False
    y_limits: Range[float] = Range[float](0.0, 150.0)
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


def get_scan_number(folder_path: Path = MOST_RECENT_FOLDER) -> int:
    scan_files = DatFinder(folder_path).find_scanfiles()
    return len(scan_files)


def plot_stft_analysis(ax_scan, ax_spec, ax_ions, groups: list[tuple[str, list[Path]]], config: IonDataAnalysisConfig, window_size: Time, cut_start: int, cut_end: int, baseline_mode: BaselineMode = DEFAULT_BASELINE_MODE, moving_average_window: int = 10):
    if not groups:
        raise ValueError("No folders selected. Check at least one folder in the Selection tab.")

    ax_ions.yaxis.tick_right()
    ax_ions.yaxis.set_label_position("right")

    # Folders in the same group are loaded together (one combined, averaged curve per
    # group); the STFT below then averages across groups the same way, by treating each
    # group's curve as one of several scans, exactly as calculate_averaged_spectrogram
    # already does for multiple raw scans.
    group_averages = []
    for i, (group_id, folders) in enumerate(groups):
        file_paths = DatFinder(folders, is_full_path=True).find_datafiles()
        num_scans = get_scan_number(folders)
        raw_scans = load_ion_data(file_paths)
        calculated_scans = run_pipeline(raw_scans, config)
        truncated_scans = [c2t_scan.cut(cut_start, cut_end) for c2t_scan in calculated_scans]
        averaged_data = averaging.average_scans(truncated_scans)
        group_averages.append(averaged_data)

        colour = _group_colour(i)
        label = f"ID {group_id} ({num_scans} scans)"
        plot_ScanData(ax_scan, data=averaged_data, label=label, ax_twin=None, color=colour, ecolor=colour)

    ax_scan.grid(visible=True, which="major", alpha=1.0)

    stft_config = StftAnalysisConfig(group_averages, window_size)
    resampled_scans = resample_scans(group_averages, stft_config.axis)
    spectrogram = StftAnalysis(resampled_scans, stft_config).calculate_averaged_spectrogram(
        detrend_mode=baseline_mode.value,
        moving_average_window=moving_average_window,
    )
    mesh = plot_Spectrogram(ax_spec, spectrogram, v_range=Range(0, 1), shading="auto")
    plot_nyquist_frequency(ax_spec, group_averages[0])

    ax_spec.set_axisbelow(False)
    ax_spec.grid(visible=True, color="grey", linewidth=0.3, zorder=5)

    ax_ions.tick_params(axis="y", left=False, right=True, labelleft=False)
    ax_ions.grid(False)
    return mesh


# ============================================================
# GUI
# ============================================================
class FolderSelectionWidget(QWidget):
    """One row per remembered folder: include checkbox, path, and a colour-group ID.

    Changes apply immediately (no Apply button), since there is nothing to stage -- unlike
    the spec-driven tabs, a row's checkbox/ID is the one copy of that state.
    """

    def __init__(self, on_changed, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._on_changed = on_changed
        self._entries: list[FolderEntry] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Check to include a folder; folders sharing an ID are combined into one curve of that ID's colour.")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._rows_container)
        outer.addWidget(scroll, stretch=1)

    def set_entries(self, entries: list[FolderEntry]) -> None:
        self._entries = entries
        while self._rows_layout.count() > 1:  # keep the trailing stretch
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for entry in entries:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            checkbox = QCheckBox()
            checkbox.setChecked(entry.include)
            checkbox.toggled.connect(lambda checked, e=entry: self._set_include(e, checked))

            display = entry.path if len(entry.path) <= 42 else "…" + entry.path[-41:]
            label = QLabel(display)
            label.setToolTip(entry.path)

            id_edit = QLineEdit(entry.group_id)
            id_edit.setFixedWidth(50)
            id_edit.editingFinished.connect(lambda e=entry, w=id_edit: self._set_group_id(e, w.text()))

            row_layout.addWidget(checkbox)
            row_layout.addWidget(label, stretch=1)
            row_layout.addWidget(id_edit)
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)

    def _set_include(self, entry: FolderEntry, checked: bool) -> None:
        entry.include = checked
        self._on_changed()

    def _set_group_id(self, entry: FolderEntry, text: str) -> None:
        text = text.strip() or "1"
        if entry.group_id != text:
            entry.group_id = text
            self._on_changed()


class StftForm(DirtyFormWidget):
    _specs: dict[str, FieldSpec] = {
        "delay_center": LengthSpec("Delay center", Prefix.MILLI, extra_decimals=1),
        "center_x": FloatSpec("Center x", min=-10_000.0, max=10_000.0, decimals=2, step=1.0),
        "center_y": FloatSpec("Center y", min=-10_000.0, max=10_000.0, decimals=2, step=1.0),
        "angle": AngleSpec("Angle"),
        "analysis_zone": RangeSpec("Zone", IntSpec("", min=0, max=10_000)),
        "transform_parameter": FloatSpec("ScaleX", min=0.0, max=10.0, decimals=3, step=0.01),
        "window_size": TimeSpec("STFT window", Prefix.PICO, min=0.0, max=10_000.0),
        "cut_start": IntSpec("Cut start index", min=0, max=100_000),
        "cut_end": IntSpec("Cut end index", min=0, max=100_000),
        "baseline_mode": EnumSpec("Baseline subtraction", BaselineMode),
        "moving_average_window": IntSpec("Moving avg. window", min=1, max=10_000),
        "x_limits_enabled": BoolSpec("Limit x axis"),
        "x_limits": RangeSpec("x range (ps)", FloatSpec("", min=-1_000_000.0, max=1_000_000.0, decimals=1, step=1.0)),
        "y_limits_enabled": BoolSpec("Limit y axis"),
        "y_limits": RangeSpec("y range (GHz)", FloatSpec("", min=0.0, max=1_000_000.0, decimals=1, step=1.0)),
        "log_colorscale": BoolSpec("Logarithmic colour scale"),
    }
    _groups = [
        ("Analysis config", ["delay_center", "center_x", "center_y", "angle", "analysis_zone", "transform_parameter"]),
        ("STFT window", ["window_size", "cut_start", "cut_end"]),
        ("Baseline subtraction", ["baseline_mode", "moving_average_window"]),
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
        # plotted), so it is built and wired up directly rather than through _groups/_specs.
        self._selection_widget = FolderSelectionWidget(on_changed=lambda: self._on_apply())
        self._selection_widget.set_entries(self._config.folder_entries)
        tabs.addTab(self._selection_widget, "Selection")

        self.form_layout.addWidget(tabs)

    def refresh_selection(self) -> None:
        self._selection_widget.set_entries(self._config.folder_entries)

    def __init__(self, settings: StftSettings, on_apply, parent: QWidget | None = None) -> None:
        super().__init__(settings, parent)
        self._on_apply = on_apply

    def on_apply(self) -> None:
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
        self._fig.subplots_adjust(top=0.94, left=0.08, right=0.92, bottom=0.12)
        self._ax_scan.xaxis.label.set_visible(False)
        self._ax_scan.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        self._ax_spec.tick_params(axis="x", which="both", direction="out")
        # A colour-limit slider stands where the colorbar would. Its axes is made once,
        # after subplots_adjust, so replots do not keep shrinking the plot to make room.
        # make_axes only narrows ax_spec, so ax_scan must be narrowed to match or its
        # (wider) x-axis no longer lines up with ax_spec's despite sharing x values.
        self._cax, _ = make_axes(self._ax_spec)
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
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._on_browse_folder)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._commit_folder)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(QLabel("Folder"))
        bottom_row.addWidget(self._folder_edit, stretch=1)
        bottom_row.addWidget(browse_button)
        bottom_row.addWidget(refresh_button)

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

    def _replot(self) -> None:
        self._clear_clim_slider()
        self._ax_scan.clear()
        self._ax_spec.clear()
        self._ax_ions.clear()
        # clear() resets tick/label visibility too, so the top and bottom plots must be
        # re-pinned together (shared x-axis, no gap) after every replot, not just in setup
        self._ax_scan.xaxis.label.set_visible(False)
        self._ax_scan.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        self._ax_spec.tick_params(axis="x", which="both", direction="out")
        settings = self.vm.settings
        groups = _group_entries(settings.folder_entries)
        try:
            mesh = plot_stft_analysis(
                self._ax_scan,
                self._ax_spec,
                self._ax_ions,
                groups,
                build_config(settings),
                settings.window_size,
                settings.cut_start,
                settings.cut_end,
                settings.baseline_mode,
                settings.moving_average_window,
            )
        except Exception as e:
            self._show_error(e)
            return
        if settings.x_limits_enabled:
            self._ax_spec.set_xlim(settings.x_limits.min, settings.x_limits.max)
        if settings.y_limits_enabled:
            self._ax_spec.set_ylim(settings.y_limits.min, settings.y_limits.max)
        mesh.set_norm(LogNorm(vmin=LOG_COLORSCALE_VMIN, vmax=1.0) if settings.log_colorscale else Normalize(vmin=0.0, vmax=1.0))
        self._build_clim_slider(mesh)
        self._fig.suptitle(", ".join(f"ID {group_id}" for group_id, _ in groups), fontsize=10)
        self._canvas.draw_idle()

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
