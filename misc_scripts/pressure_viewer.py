from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.transforms import blended_transform_factory

from base_core.framework.events import EventBus
from base_qt.app.dispatcher import QtDispatcher
from base_qt.ui import LabMainWindow, MessageLevel, Panel, PanelViewModel
from base_qt.ui.apply import install_ui


# ============================================================
# Defaults (all editable in the window)
# ============================================================
DEFAULT_PATH = r"Z:\Droplets\20261109\20261109_Pressures_TEST.txt"
# How many consecutive raw rows are averaged into one plotted point; keeps the plot
# readable without the GUI choking on files with tens of thousands of rows.
DEFAULT_SMOOTHING_POINTS = 10
# How close the mouse has to be (in pixels, horizontally only - see _on_hover) to a note's
# marker/guide line before its entry highlights in the notes box
HOVER_RADIUS_PX = 25.0

# Combo box label -> settings value
TIME_UNIT_LABELS = {"Seconds": "s", "Minutes": "min", "Hours": "hour"}
# Settings value -> seconds per unit
SECONDS_PER_TIME_UNIT = {"s": 1.0, "min": 60.0, "hour": 3600.0}

# Column names for the data columns (everything after the time column), in file order.
# Plain text for checkboxes/identification; "_" marks a subscript in the file's naming
# convention, rendered properly (via mathtext) in plot legends by CHANNEL_MATH_LABELS below.
CHANNEL_NAMES = [
    "T_N (K)",
    "T_S (K)",
    "P_S (Torr)",
    "P_D (Torr)",
    "P_diff (Torr)",
    "P_VMI (Torr)",
    "RGA_A (Torr)",
    "RGA_B (Torr)",
]
CHANNEL_MATH_LABELS = [
    "T$_N$ (K)",
    "T$_S$ (K)",
    "P$_S$ (Torr)",
    "P$_D$ (Torr)",
    "P$_{diff}$ (Torr)",
    "P$_{VMI}$ (Torr)",
    "RGA$_A$ (Torr)",
    "RGA$_B$ (Torr)",
]
# Indices (into the data columns, excluding time) that are temperatures rather than
# pressures; plotted on their own small axis up top instead of the main log-scale one
TEMPERATURE_CHANNEL_INDICES = {0, 1}


@dataclass
class PressureData:
    time: np.ndarray  # (n,) seconds
    columns: np.ndarray  # (n, n_channels), everything but the time column
    column_names: list[str]  # plain text, for checkboxes
    column_labels: list[str]  # mathtext, for plot legends


@dataclass
class Note:
    time: float
    text: str


def notes_path_for(data_path: Path) -> Path:
    return data_path.with_name(f"{data_path.stem}_notes{data_path.suffix}")


def load_pressure_data(path: Path) -> PressureData:
    raw = np.loadtxt(path, delimiter="\t")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 2:
        raise ValueError(f"Expected a time column plus at least one data column in: {path}")
    n_channels = raw.shape[1] - 1
    # Fall back to a generic label for any channel beyond the known names (e.g. a file
    # with extra columns), rather than erroring out
    column_names = [
        CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f"Channel {i + 1}" for i in range(n_channels)
    ]
    column_labels = [
        CHANNEL_MATH_LABELS[i] if i < len(CHANNEL_MATH_LABELS) else f"Channel {i + 1}" for i in range(n_channels)
    ]
    return PressureData(time=raw[:, 0], columns=raw[:, 1:], column_names=column_names, column_labels=column_labels)


def load_notes(path: Path) -> list[Note]:
    # Tab-separated "time, note"; blank/trailing/malformed lines are skipped
    if not path.exists():
        return []
    notes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 1)
        if len(parts) < 2:
            continue
        try:
            time = float(parts[0])
        except ValueError:
            continue
        text = parts[1].strip()
        if text:
            notes.append(Note(time=time, text=text))
    return notes


def smooth_resample(time: np.ndarray, values: np.ndarray, bin_points: int) -> tuple[np.ndarray, np.ndarray]:
    # Block-average every `bin_points` consecutive raw samples into one point, so the
    # plotted resolution stays reasonable regardless of the file's native sample rate
    if bin_points <= 1 or len(time) < bin_points * 2:
        return time, values
    n_bins = len(time) // bin_points
    trimmed = n_bins * bin_points
    time_binned = time[:trimmed].reshape(n_bins, bin_points).mean(axis=1)
    values_binned = values[:trimmed].reshape(n_bins, bin_points).mean(axis=1)
    return time_binned, values_binned


@dataclass
class PressureSettings:
    smoothing_points: int = DEFAULT_SMOOTHING_POINTS
    log_y: bool = True
    time_unit: str = "s"
    # Manual x-axis bounds, in raw seconds (unit-independent); None means autoscale.
    # Set via the "Set as min/max" buttons, from the currently selected note's time.
    x_min_seconds: float | None = None
    x_max_seconds: float | None = None


# ============================================================
# GUI
# ============================================================
class PressureViewModel(PanelViewModel):
    def __init__(self, bus: EventBus, dispatcher: QtDispatcher, settings: PressureSettings, path: Path) -> None:
        super().__init__(bus, dispatcher)
        self.settings = settings
        self.path = path
        # Which data-column indices (0-based, excluding time) are plotted; filled in once
        # the first file loads, since the column count isn't known up front
        self.selected_columns: set[int] = set()

    def load(self) -> tuple[PressureData, list[Note]]:
        data = load_pressure_data(self.path)
        notes = load_notes(notes_path_for(self.path))
        return data, notes

    def report_error(self, e: Exception) -> None:
        self._msg(f"Error: {e}", MessageLevel.ERROR)

    def save(self, fig: Figure) -> Path:
        out_dir = self.path.parent / "pressure_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.path.stem}.png"
        fig.savefig(out_path, dpi=150)
        self._msg(f"Saved plot to: {out_path}")
        return out_path


class PressurePanel(Panel):
    def __init__(self, vm: PressureViewModel, parent: QWidget | None = None) -> None:
        self._known_path: Path | None = None
        self._notes: list[Note] = []
        self._note_display_times: list[float] = []  # note times, converted to the current axis unit
        self._note_display_x: np.ndarray | None = None  # (n_notes,) pixel x, cached per draw
        self._selected_note_index: int | None = None  # set via the Prev/Next buttons
        self._hovered_note_index: int | None = None  # set via mouse hover
        super().__init__("Pressure log", vm, parent)

    def setup(self) -> None:
        self._fig = Figure(figsize=(12, 6))
        # Axes are (re)built in _replot, since whether the temperature subplot exists at
        # all depends on which columns are selected
        self._ax_temp = None
        self._ax = None
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._canvas.mpl_connect("draw_event", self._on_draw)

        plot_col = QVBoxLayout()
        plot_col.addWidget(NavigationToolbar2QT(self._canvas, self))
        plot_col.addWidget(self._canvas, stretch=1)

        # Column checkboxes are built once the file loads, since the count/names depend
        # on the file; this is just the scrollable container they go in
        self._columns_box = QVBoxLayout()
        self._columns_box.setSpacing(2)
        columns_widget = QWidget()
        columns_widget.setLayout(self._columns_box)
        columns_scroll = QScrollArea()
        columns_scroll.setWidgetResizable(True)
        columns_scroll.setWidget(columns_widget)
        columns_scroll.setMinimumHeight(110)
        self._column_checkboxes: list[QCheckBox] = []

        self._log_y_checkbox = QCheckBox("Log Y axis (pressures)")
        self._log_y_checkbox.setChecked(self.vm.settings.log_y)
        self._log_y_checkbox.stateChanged.connect(self._on_log_y_changed)

        self._smoothing_spin = QSpinBox()
        self._smoothing_spin.setRange(1, 1000)
        self._smoothing_spin.setValue(self.vm.settings.smoothing_points)
        self._smoothing_spin.valueChanged.connect(self._on_smoothing_changed)

        self._time_unit_combo = QComboBox()
        self._time_unit_combo.addItems(list(TIME_UNIT_LABELS.keys()))
        self._time_unit_combo.currentTextChanged.connect(self._on_time_unit_changed)

        save_button = QPushButton("Save PNG")
        save_button.clicked.connect(self._on_save)

        # Shows one note at a time (via Prev/Next below), since a floating hover tooltip
        # proved unreliable (see _on_hover's comment) and a full list was more than needed
        self._notes_text = QTextEdit()
        self._notes_text.setReadOnly(True)
        self._notes_text.setMinimumHeight(80)

        self._note_label = QLabel("No notes")
        prev_button = QPushButton("< Prev")
        prev_button.clicked.connect(lambda: self._cycle_note(-1))
        next_button = QPushButton("Next >")
        next_button.clicked.connect(lambda: self._cycle_note(1))
        cycle_row = QHBoxLayout()
        cycle_row.addWidget(prev_button, 1)
        cycle_row.addWidget(next_button, 1)

        set_min_button = QPushButton("Set as min")
        set_min_button.clicked.connect(lambda: self._set_range_from_selected("min"))
        set_max_button = QPushButton("Set as max")
        set_max_button.clicked.connect(lambda: self._set_range_from_selected("max"))
        range_row = QHBoxLayout()
        range_row.addWidget(set_min_button, 1)
        range_row.addWidget(set_max_button, 1)

        reset_range_button = QPushButton("Reset range")
        reset_range_button.clicked.connect(self._on_reset_range)

        side_panel = QWidget()
        side_col = QVBoxLayout(side_panel)
        side_col.setContentsMargins(0, 0, 0, 0)
        side_col.addWidget(QLabel("Columns to plot"))
        side_col.addWidget(columns_scroll)
        smoothing_row = QHBoxLayout()
        smoothing_row.addWidget(QLabel("Smoothing (points/bin)"))
        smoothing_row.addWidget(self._smoothing_spin)
        side_col.addLayout(smoothing_row)
        time_unit_row = QHBoxLayout()
        time_unit_row.addWidget(QLabel("Time axis"))
        time_unit_row.addWidget(self._time_unit_combo)
        side_col.addLayout(time_unit_row)
        side_col.addWidget(self._log_y_checkbox)
        side_col.addWidget(save_button)
        side_col.addWidget(QLabel("Notes"))
        side_col.addWidget(self._notes_text)
        side_col.addWidget(self._note_label)
        side_col.addLayout(cycle_row)
        side_col.addLayout(range_row)
        side_col.addWidget(reset_range_button)
        side_col.addStretch(1)
        side_panel.setFixedWidth(320)

        plot_row = QHBoxLayout()
        plot_row.addLayout(plot_col, stretch=1)
        plot_row.addWidget(side_panel)

        self._path_edit = QLineEdit(str(self.vm.path))
        self._path_edit.textChanged.connect(self._on_path_changed)
        self._path_edit.returnPressed.connect(self._on_path_entered)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._on_browse)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._replot)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(QLabel("Pressures file"))
        bottom_row.addWidget(self._path_edit, stretch=1)
        bottom_row.addWidget(browse_button)
        bottom_row.addWidget(refresh_button)

        self.body_layout.addLayout(plot_row, stretch=1)
        self.body_layout.addLayout(bottom_row)
        self._rebuild_columns()
        self._replot()

    def _show_error(self, e: Exception) -> None:
        self._fig.clear()
        self._ax_temp = None
        self._ax = self._fig.add_subplot(111)
        self._ax.set_title(f"Error: {e}", color="r")
        self._canvas.draw_idle()
        self.vm.report_error(e)

    def _rebuild_columns(self) -> None:
        # Only rebuilds the checkbox list when the set of columns actually changes (i.e.
        # a different file was loaded), so toggling a checkbox doesn't get clobbered
        try:
            data, _ = self.vm.load()
        except Exception as e:
            self._show_error(e)
            return
        if self.vm.path == self._known_path and len(self._column_checkboxes) == len(data.column_names):
            return
        self._known_path = self.vm.path
        for checkbox in self._column_checkboxes:
            checkbox.setParent(None)
        self._column_checkboxes = []
        self.vm.selected_columns = set(range(len(data.column_names)))
        for i, name in enumerate(data.column_names):
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda _state, idx=i: self._on_column_toggled(idx))
            self._columns_box.addWidget(checkbox)
            self._column_checkboxes.append(checkbox)

    def _on_column_toggled(self, index: int) -> None:
        checkbox = self._column_checkboxes[index]
        if checkbox.isChecked():
            self.vm.selected_columns.add(index)
        else:
            self.vm.selected_columns.discard(index)
        self._replot()

    def _on_log_y_changed(self, _state: int) -> None:
        self.vm.settings.log_y = self._log_y_checkbox.isChecked()
        self._replot()

    def _on_smoothing_changed(self, value: int) -> None:
        self.vm.settings.smoothing_points = value
        self._replot()

    def _on_time_unit_changed(self, label: str) -> None:
        self.vm.settings.time_unit = TIME_UNIT_LABELS[label]
        self._replot()

    def _replot(self) -> None:
        self._note_display_x = None
        try:
            data, notes = self.vm.load()
        except Exception as e:
            self._show_error(e)
            return
        settings = self.vm.settings
        unit_label = settings.time_unit
        seconds_per_unit = SECONDS_PER_TIME_UNIT[unit_label]
        in_range = [n for n in notes if data.time.min() <= n.time <= data.time.max()] if len(data.time) else []
        self._notes = sorted(in_range, key=lambda n: n.time)
        if self._selected_note_index is not None and self._selected_note_index >= len(self._notes):
            self._selected_note_index = None
        if self._hovered_note_index is not None and self._hovered_note_index >= len(self._notes):
            self._hovered_note_index = None
        # Cached alongside the notes so hover lookups and the notes box use the same unit
        # as the axis, without recomputing the conversion on every mouse move
        self._note_display_times = [n.time / seconds_per_unit for n in self._notes]

        # The temperature subplot only exists while at least one temperature channel is
        # selected; rebuild the figure's axes each time, since whether it's there at all
        # (not just whether it's empty) depends on that.
        self._fig.clear()
        selected_temp = self.vm.selected_columns & TEMPERATURE_CHANNEL_INDICES
        selected_pressure = self.vm.selected_columns - TEMPERATURE_CHANNEL_INDICES
        if selected_temp:
            gs = self._fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.08)
            self._ax_temp = self._fig.add_subplot(gs[0])
            self._ax = self._fig.add_subplot(gs[1], sharex=self._ax_temp)
        else:
            self._ax_temp = None
            self._ax = self._fig.add_subplot(111)
        self._fig.subplots_adjust(left=0.08, right=0.8, bottom=0.1, top=0.95)

        for index in sorted(self.vm.selected_columns):
            if index >= len(data.column_names):
                continue
            time, values = smooth_resample(data.time, data.columns[:, index], settings.smoothing_points)
            target_ax = self._ax_temp if index in TEMPERATURE_CHANNEL_INDICES else self._ax
            target_ax.plot(time / seconds_per_unit, values, label=data.column_labels[index], linewidth=1.2)

        if self._ax_temp is not None:
            self._ax_temp.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
            self._ax_temp.set_ylabel("Temp (K)")
            self._ax_temp.grid(True, alpha=0.3)
            self._ax_temp.tick_params(labelbottom=False)

        if selected_pressure:
            self._ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
        self._ax.set_xlabel(f"Time ({unit_label})")
        self._ax.set_ylabel("Pressure (Torr)")
        self._ax.grid(True, alpha=0.3)
        if settings.log_y:
            self._ax.set_yscale("log")

        # Note markers sit at a fixed height near the top of each axes (not the data), so
        # they stay visible regardless of the plotted values' scale, on both subplots since
        # they share the x-axis
        if self._notes:
            for ax in filter(None, (self._ax_temp, self._ax)):
                trans = blended_transform_factory(ax.transData, ax.transAxes)
                ax.scatter(
                    self._note_display_times,
                    [0.97] * len(self._notes),
                    transform=trans,
                    marker="v",
                    color="black",
                    s=30,
                    zorder=6,
                    clip_on=False,
                )
                for t in self._note_display_times:
                    ax.axvline(t, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

        # Manual x-range (from "Set as min/max"), converted from the stored seconds into
        # the current display unit; setting it on self._ax also moves self._ax_temp, since
        # the two axes share their x-axis
        if settings.x_min_seconds is not None or settings.x_max_seconds is not None:
            x_min = settings.x_min_seconds / seconds_per_unit if settings.x_min_seconds is not None else None
            x_max = settings.x_max_seconds / seconds_per_unit if settings.x_max_seconds is not None else None
            self._ax.set_xlim(x_min, x_max)

        self._refresh_notes_text()
        self._canvas.draw_idle()

    def _on_draw(self, _event) -> None:
        # Cache each note marker's current pixel x-position, recomputed after every draw
        # (zoom/pan/resize all move it), so hover lookups are a cheap distance check
        if not self._notes:
            self._note_display_x = None
            return
        trans = blended_transform_factory(self._ax.transData, self._ax.transAxes)
        points = [(t, 0.97) for t in self._note_display_times]
        self._note_display_x = trans.transform(points)[:, 0]

    def _on_hover(self, event) -> None:
        if self._note_display_x is None or event.x is None:
            if self._hovered_note_index is not None:
                self._hovered_note_index = None
                self._refresh_notes_text()
            return
        # Only the horizontal distance matters: each note has a guide line running the
        # full height of the plot, so hovering anywhere along it (not just right on the
        # small top marker) previews it in the notes box.
        distances = np.abs(self._note_display_x - event.x)
        nearest = int(np.argmin(distances))
        new_hover = nearest if distances[nearest] <= HOVER_RADIUS_PX else None
        if new_hover != self._hovered_note_index:
            self._hovered_note_index = new_hover
            self._refresh_notes_text()

    # ============================================================
    # Notes box: shows one note at a time (hover previews it, Prev/Next picks it), plus
    # the range buttons that use whichever note Prev/Next last selected
    # ============================================================
    def _displayed_note_index(self) -> int | None:
        # A hovered note previews in the box without disturbing the Prev/Next selection
        return self._hovered_note_index if self._hovered_note_index is not None else self._selected_note_index

    def _refresh_notes_text(self) -> None:
        index = self._displayed_note_index()
        if not self._notes:
            self._notes_text.setPlainText("(no notes in range)")
        elif index is None:
            self._notes_text.setPlainText("(use Prev/Next, or hover a marker)")
        else:
            self._notes_text.setPlainText(self._notes[index].text)
        self._update_note_label()

    def _update_note_label(self) -> None:
        index = self._displayed_note_index()
        if not self._notes:
            self._note_label.setText("No notes")
        elif index is None:
            self._note_label.setText(f"{len(self._notes)} notes")
        else:
            display_time = self._note_display_times[index]
            self._note_label.setText(
                f"Note {index + 1}/{len(self._notes)}: t={display_time:g} {self.vm.settings.time_unit}"
            )

    def _cycle_note(self, delta: int) -> None:
        if not self._notes:
            return
        if self._selected_note_index is None:
            self._selected_note_index = 0
        else:
            self._selected_note_index = (self._selected_note_index + delta) % len(self._notes)
        self._refresh_notes_text()

    def _set_range_from_selected(self, which: str) -> None:
        if not self._notes:
            return
        if self._selected_note_index is None:
            self._selected_note_index = 0
        seconds = self._notes[self._selected_note_index].time
        if which == "min":
            self.vm.settings.x_min_seconds = seconds
        else:
            self.vm.settings.x_max_seconds = seconds
        self._replot()

    def _on_reset_range(self) -> None:
        self.vm.settings.x_min_seconds = None
        self.vm.settings.x_max_seconds = None
        self._replot()

    def _on_path_changed(self, text: str) -> None:
        self.vm.path = Path(text.strip())

    def _on_path_entered(self) -> None:
        self._rebuild_columns()
        self._replot()

    def _on_browse(self) -> None:
        start_dir = str(self.vm.path.parent) if self.vm.path.parent.exists() else ""
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select pressures file",
            start_dir,
            "Text files (*.txt);;All files (*.*)",
        )
        if not selected:
            return
        self._path_edit.setText(selected)
        self._rebuild_columns()
        self._replot()

    def _on_save(self) -> None:
        try:
            self.vm.save(self._fig)
        except Exception as e:
            self._show_error(e)


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    install_ui(app)
    bus = EventBus()
    dispatcher = QtDispatcher()

    window = LabMainWindow("Pressure viewer", bus, dispatcher)
    vm = PressureViewModel(bus, dispatcher, PressureSettings(), Path(DEFAULT_PATH))
    window.setCentralWidget(PressurePanel(vm))
    window.resize(1400, 750)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
