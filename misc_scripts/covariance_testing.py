from dataclasses import dataclass, replace
from pathlib import Path
import sys

import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data

from apps.c2t_calculation.domain.plotting import plot_ions_square
from base_core.framework.events import EventBus
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.math.special_models import AngularCovariance
from base_core.plotting.covariance_plotting import plot_covariance
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length
from base_qt.app.dispatcher import QtDispatcher
from base_qt.ui import LabMainWindow, MessageLevel, Panel, PanelViewModel
from base_qt.ui.apply import install_ui
from base_qt.ui.form import AngleSpec, BoolSpec, DirtyFormWidget, FloatSpec, IntSpec, LengthSpec, RangeSpec


# ============================================================
# Defaults (all editable in the window)
# ============================================================
# Either a single .dat file or a folder of data files
DEFAULT_PATH = r"Z:\Droplets\20261104\VerticalPump\20261104145650DLY__15p7950mm.dat"


@dataclass
class CovarianceSettings:
    # IonDataAnalysisConfig
    delay_center: Length = Length(93.3, Prefix.MILLI)
    center_x: float = 98.0
    center_y: float = 102.0
    angle: Angle = Angle(12, AngleUnit.DEG)
    analysis_zone: Range[int] = Range[int](10, 120)
    transform_parameter: float = 0.76
    # Which IonData to show
    scan_index: int = 0
    ion_data_index: int = 0
    # AngularCovariance; the radial range is only used while radial_cut is on
    angle_bins: int = 180
    radial_cut: bool = False
    radial_range: Range[float] = Range[float](0.0, 120.0)
    binary_per_frame: bool = False
    normalize: bool = True


# Optional alternative:
# set this to a concrete stage position if you want to select by value
TARGET_STAGE_POSITION = None
# example:
# TARGET_STAGE_POSITION = Length(0.0, Prefix.MILLI)


def build_config(settings: CovarianceSettings) -> IonDataAnalysisConfig:
    return IonDataAnalysisConfig(
        delay_center=settings.delay_center,
        center=Point(settings.center_x, settings.center_y),
        angle=settings.angle,
        analysis_zone=settings.analysis_zone,
        transform_parameter=settings.transform_parameter,
    )


def find_file_paths(path: Path) -> list[list[Path]]:
    if path.is_file():
        return [[path]]
    return DatFinder(path, is_full_path=True).find_datafiles()


def select_ion_data_by_index(raw_scan, index: int):
    if index < 0 or index >= len(raw_scan.ion_datas):
        raise IndexError(
            f"ION_DATA_INDEX={index} out of range for {len(raw_scan.ion_datas)} ion_datas."
        )
    return raw_scan.ion_datas[index]


def select_ion_data_by_stage_position(raw_scan, target_stage_position):
    for ion_data in raw_scan.ion_datas:
        if ion_data.stage_position == target_stage_position:
            return ion_data
    raise ValueError(f"No IonData found for stage_position={target_stage_position}")


def load_selected_ion_data(path: Path, scan_index: int, ion_data_index: int):
    # ============================================================
    # Load data
    # ============================================================
    raw_scans = load_ion_data(find_file_paths(path))

    if scan_index < 0 or scan_index >= len(raw_scans):
        raise IndexError(
            f"SCAN_INDEX={scan_index} out of range for {len(raw_scans)} raw_scans."
        )

    raw_scan = raw_scans[scan_index]

    print(f"Loaded {len(raw_scans)} scans.")
    print(f"Selected scan {scan_index} with {len(raw_scan.ion_datas)} ion_datas.\n")

    print("Available ion_datas:")
    for i, ion_data in enumerate(raw_scan.ion_datas):
        print(
            f"[{i:03d}] stage_position={ion_data.stage_position}, "
            f"ions_per_frame={ion_data.ions_per_frame:.3f}"
        )

    # ============================================================
    # Select one IonData
    # ============================================================
    if TARGET_STAGE_POSITION is not None:
        ion_data = select_ion_data_by_stage_position(raw_scan, TARGET_STAGE_POSITION)
    else:
        ion_data = select_ion_data_by_index(raw_scan, ion_data_index)

    print("\nSelected ion_data:")
    print(f"stage_position={ion_data.stage_position}")
    return ion_data


def calculate_centroid(ion_data) -> Point:
    # Mean position of all raw hits, in detector coordinates
    return Point(float(np.mean(ion_data.points.x)), float(np.mean(ion_data.points.y)))


def normalize_covariance(cov: AngularCovariance) -> AngularCovariance:
    # Scale to [0, 1]; negatives are clipped, same as plot_covariance does by default
    matrix = np.maximum(cov.matrix, 0.0)
    peak = float(np.max(matrix)) if matrix.size else 0.0
    return replace(cov, matrix=matrix / peak if peak > 0 else matrix)


def plot_ions_and_covariance(fig, ax_l, ax_r, ion_data, config: IonDataAnalysisConfig, angle_bins: int, radial_range: Range[float] | None, binary_per_frame: bool, normalize: bool):
    # ============================================================
    # Prepare hits
    # ============================================================
    hits = ion_data.get_points_after_config(config)

    print(f"hits={len(hits)}")

    if len(hits) == 0:
        raise ValueError("Selected IonData has no hits after config.")
    if hits.marker_max < 5:
        raise ValueError("Selected IonData has too few markers for covariance.")

    # ============================================================
    # Compute covariance
    # ============================================================
    cov = AngularCovariance.compute_covariance(
        hits,
        angle_bins=angle_bins,
        radial_range=radial_range,
        binary_per_frame=binary_per_frame,
    )

    print(f"Computed covariance with n_frames={cov.n_frames}")

    # ============================================================
    # Plot ions + covariance
    # ============================================================
    plot_ions_square(
        ax_l,
        hits,
        color="red",
        label=None,
    )
    ax_l.set_title(
        "Ions\n"
        f"markers={hits.marker_max}, hits={len(hits)}"
    )
    ax_l.set_aspect("equal")

    if normalize:
        cov = normalize_covariance(cov)

    mesh = plot_covariance(ax_r, cov)
    ax_r.set_title(
        "Angular covariance\n"
        f"frames={cov.n_frames}"
    )
    return fig.colorbar(mesh, ax=ax_r, label="normalized covariance" if normalize else "covariance")


# ============================================================
# GUI
# ============================================================
class CovarianceForm(DirtyFormWidget):
    _specs = {
        "delay_center": LengthSpec("Delay center", Prefix.MILLI),
        "center_x": FloatSpec("Center x", min=-10_000.0, max=10_000.0, decimals=2, step=1.0),
        "center_y": FloatSpec("Center y", min=-10_000.0, max=10_000.0, decimals=2, step=1.0),
        "angle": AngleSpec("Angle"),
        "analysis_zone": RangeSpec("Zone", IntSpec("", min=0, max=10_000)),
        "transform_parameter": FloatSpec("ScaleX", min=0.0, max=10.0, decimals=3, step=0.01),
        "scan_index": IntSpec("Scan index", min=0, max=10_000),
        "ion_data_index": IntSpec("IonData index", min=0, max=100_000),
        "angle_bins": IntSpec("Angle bins", min=1, max=3600),
        "radial_cut": BoolSpec("Radial cut"),
        "radial_range": RangeSpec("Radial range", FloatSpec("", min=0.0, max=10_000.0, decimals=1, step=1.0)),
        "binary_per_frame": BoolSpec("Binary per frame"),
        "normalize": BoolSpec("Normalize"),
    }
    _groups = [
        ("Analysis config", ["delay_center", "center_x", "center_y", "angle", "analysis_zone", "transform_parameter"]),
        ("Data selection", ["scan_index", "ion_data_index"]),
        ("Covariance", ["angle_bins", "radial_cut", "radial_range", "binary_per_frame", "normalize"]),
    ]

    def __init__(self, settings: CovarianceSettings, on_apply, parent: QWidget | None = None) -> None:
        super().__init__(settings, parent)
        self._on_apply = on_apply

    def on_apply(self) -> None:
        self._on_apply()


class CovarianceViewModel(PanelViewModel):
    def __init__(self, bus: EventBus, dispatcher: QtDispatcher, settings: CovarianceSettings, path: Path) -> None:
        super().__init__(bus, dispatcher)
        self.settings = settings
        self.path = path

    def load_ion_data(self):
        return load_selected_ion_data(self.path, self.settings.scan_index, self.settings.ion_data_index)

    def radial_range(self) -> Range[float] | None:
        return self.settings.radial_range if self.settings.radial_cut else None

    def auto_center(self) -> None:
        centroid = calculate_centroid(self.load_ion_data())
        print(f"Centroid: ({centroid.x:.1f}, {centroid.y:.1f})")
        self.settings.center_x = round(centroid.x, 1)
        self.settings.center_y = round(centroid.y, 1)
        self._msg(f"Centroid: ({centroid.x:.1f}, {centroid.y:.1f})")

    # ============================================================
    # Save single plot
    # ============================================================
    def save(self, fig: Figure) -> None:
        out_dir = (self.path.parent if self.path.is_file() else self.path) / "single_covariance_with_ions"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "selected_iondata_covariance.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to: {out_path}")
        self._msg(f"Saved plot to: {out_path}")

    def report_error(self, e: Exception) -> None:
        self._msg(f"Error: {e}", MessageLevel.ERROR)


class CovariancePanel(Panel):
    def __init__(self, vm: CovarianceViewModel, parent: QWidget | None = None) -> None:
        super().__init__("Angular covariance", vm, parent)

    def setup(self) -> None:
        self._fig = Figure(figsize=(12, 6))
        self._ax_l, self._ax_r = self._fig.subplots(1, 2)
        self._fig.subplots_adjust(left=0.06, right=0.94, wspace=0.3)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._colorbars = []

        plot_col = QVBoxLayout()
        plot_col.addWidget(NavigationToolbar2QT(self._canvas, self))
        plot_col.addWidget(self._canvas, stretch=1)

        # Edits stay pending (dirty) until Apply, which writes them to the settings and replots
        self._form = CovarianceForm(self.vm.settings, self._replot)
        apply_row = QHBoxLayout()
        apply_row.addStretch(1)
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

        self._path_edit = QLineEdit(str(self.vm.path))
        self._path_edit.textChanged.connect(self._on_path_changed)
        self._path_edit.returnPressed.connect(self._replot)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._on_browse)
        auto_center_button = QPushButton("Auto center")
        auto_center_button.clicked.connect(self._on_auto_center)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._replot)
        save_button = QPushButton("Save")
        save_button.clicked.connect(lambda: self.vm.save(self._fig))

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(QLabel("Path"))
        bottom_row.addWidget(self._path_edit, stretch=1)
        for button in (browse_button, auto_center_button, refresh_button, save_button):
            bottom_row.addWidget(button)

        self.body_layout.addLayout(plot_row, stretch=1)
        self.body_layout.addLayout(bottom_row)
        self._replot()

    def _show_error(self, e: Exception) -> None:
        self._ax_l.set_title(f"Error: {e}", color="r")
        self._canvas.draw_idle()
        self.vm.report_error(e)

    def _replot(self) -> None:
        for colorbar in self._colorbars:
            colorbar.remove()
        self._colorbars.clear()
        self._ax_l.clear()
        self._ax_r.clear()
        settings = self.vm.settings
        try:
            self._colorbars.append(plot_ions_and_covariance(
                self._fig,
                self._ax_l,
                self._ax_r,
                self.vm.load_ion_data(),
                build_config(settings),
                settings.angle_bins,
                self.vm.radial_range(),
                settings.binary_per_frame,
                settings.normalize,
            ))
        except Exception as e:
            self._show_error(e)
            return
        self._canvas.draw_idle()

    def _on_path_changed(self, text: str) -> None:
        self.vm.path = Path(text.strip())

    def _on_auto_center(self) -> None:
        try:
            self.vm.auto_center()
        except Exception as e:
            self._show_error(e)
            return
        self._form.refresh_fields(frozenset({"center_x", "center_y"}))
        self._replot()

    def _on_browse(self) -> None:
        path = self.vm.path
        initial_dir = path.parent if path.is_file() else path
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select data file",
            str(initial_dir) if initial_dir.exists() else "",
            "Data files (*.dat);;All files (*.*)",
        )
        if not selected:
            return
        self._path_edit.setText(str(Path(selected)))
        self._replot()


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    install_ui(app)
    bus = EventBus()
    dispatcher = QtDispatcher()

    window = LabMainWindow("Covariance testing", bus, dispatcher)
    vm = CovarianceViewModel(bus, dispatcher, CovarianceSettings(), Path(DEFAULT_PATH))
    window.setCentralWidget(CovariancePanel(vm))
    window.resize(1500, 750)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
