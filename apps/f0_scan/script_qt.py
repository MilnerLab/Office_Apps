from dataclasses import dataclass
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

from _data_io.dat_loader import load_ion_data
from apps.c2t_calculation.domain.analysis import run_pipeline
from _data_io.dat_finder import MOST_RECENT_FOLDER, DatFinder

from base_core.framework.events import EventBus
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
import base_core.physics.molecules as mol
from base_core.physics.helpers import twophoton
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length
from base_qt.app.dispatcher import QtDispatcher
from base_qt.ui import LabMainWindow, Panel, PanelViewModel
from base_qt.ui.apply import install_ui
from base_qt.ui.form import AngleSpec, DirtyFormWidget, FieldSpec, FloatSpec, IntSpec, LengthSpec, RangeSpec


POSZEROSHIFT = 0 #millimetres :)
PS_TO_GHZ = 2*40.206/6.6 #kevin's fit from susCFG paper

DEFAULT_FOLDER = r"Z:\Droplets\20261110\FreqScan_2"
DEFAULT_MOLECULE = "OCS"

j_arr = np.arange(0,1,1)


@dataclass
class F0ScanSettings:
    delay_center: Length = Length(15.735, Prefix.MILLI)
    center_x: float = 139.0
    center_y: float = 210.0
    angle: Angle = Angle(12, AngleUnit.DEG)
    analysis_zone: Range[int] = Range[int](60, 120)
    transform_parameter: float = 0.8


def build_config(settings: F0ScanSettings) -> IonDataAnalysisConfig:
        return IonDataAnalysisConfig(
            delay_center=settings.delay_center,
            center=Point(settings.center_x, settings.center_y),
            angle=settings.angle,
            analysis_zone=settings.analysis_zone,
            transform_parameter=settings.transform_parameter)


def get_scan_number(folder_path: Path = MOST_RECENT_FOLDER) -> int:
        scan_files = DatFinder(folder_path).find_scanfiles()
        return len(scan_files)


def get_transitions(molecule_name: str) -> np.ndarray:
        molecule_cls = getattr(mol, molecule_name.strip(), None)
        if not (isinstance(molecule_cls, type) and issubclass(molecule_cls, mol.Molecule)):
                raise ValueError(f"Unknown molecule '{molecule_name}'")
        molecule = molecule_cls()
        return np.array([twophoton(molecule,j,True).value(Prefix.GIGA) for j in j_arr])


def plot_f0_scan(ax, folder_path: Path, molecule_name: str, config: IonDataAnalysisConfig):
        transitions = get_transitions(molecule_name)

        file_paths = DatFinder(folder_path,is_full_path=True).find_datafiles()
        n_files = sum(len(group) for group in file_paths)
        raw_scans = load_ion_data(file_paths)
        calculated_Scan = run_pipeline(raw_scans, [config])[0]

        x = [time*1e12 for time in calculated_Scan.delays]
        y = [c2t.value for c2t in calculated_Scan.measured_values]
        y_error = np.array([c.error for c in calculated_Scan.measured_values])

        freq = PS_TO_GHZ*np.array(x)

        ax.grid(visible=True,which='major',alpha=1.0)
        ax.errorbar(freq,y,y_error,elinewidth=1,ecolor="r")
        ax.plot(transitions,0.52*np.ones(len(transitions)),linestyle='',marker='|',markersize=150,color='k')
        ax.set_xlabel('Centrifuge Two-Photon Frequency (GHz)')
        ax.set_ylabel(r"$\langle \cos^2 \theta_\mathrm{2D} \rangle$")
        ax.set_title(f"{folder_path.name} ({n_files} files), {molecule_name}")

        # Top horizontal axis
        ax2 = ax.twiny()
        # Move the second x-axis to the bottom
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 40))
        ax2.spines['top'].set_visible(False)
        ax2.set_xlabel("Time Between Arms (ps)")
        # Same data as ax, so just match the limits rather than plotting it twice
        ax2.set_xlim(np.array(ax.get_xlim())/PS_TO_GHZ)
        return ax2


# ============================================================
# GUI
# ============================================================
class F0ScanForm(DirtyFormWidget):
    _specs: dict[str, FieldSpec] = {
        "delay_center": LengthSpec("Delay center", Prefix.MILLI, extra_decimals=1),
        "center_x": FloatSpec("Center x", min=-10_000.0, max=10_000.0, decimals=2, step=1.0),
        "center_y": FloatSpec("Center y", min=-10_000.0, max=10_000.0, decimals=2, step=1.0),
        "angle": AngleSpec("Angle"),
        "analysis_zone": RangeSpec("Zone", IntSpec("", min=0, max=10_000)),
        "transform_parameter": FloatSpec("ScaleX", min=0.0, max=10.0, decimals=3, step=0.01),
    }
    _groups = [
        ("Analysis config", ["delay_center", "center_x", "center_y", "angle", "analysis_zone", "transform_parameter"]),
    ]

    def __init__(self, settings: F0ScanSettings, on_apply, parent: QWidget | None = None) -> None:
        super().__init__(settings, parent)
        self._on_apply = on_apply

    def on_apply(self) -> None:
        self._on_apply()


class F0ScanViewModel(PanelViewModel):
    def __init__(self, bus: EventBus, dispatcher: QtDispatcher, settings: F0ScanSettings, folder: Path, molecule: str) -> None:
        super().__init__(bus, dispatcher)
        self.settings = settings
        self.folder = folder
        self.molecule = molecule

    def report_error(self, e: Exception) -> None:
        self._msg(f"Error: {e}")


class F0ScanPanel(Panel):
    def __init__(self, vm: F0ScanViewModel, parent: QWidget | None = None) -> None:
        super().__init__("F0 scan", vm, parent)

    def setup(self) -> None:
        self._fig = Figure(figsize=(11, 6))
        self._ax = self._fig.subplots()
        self._fig.subplots_adjust(bottom=0.3, right=0.95)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._twin_ax = None

        plot_col = QVBoxLayout()
        plot_col.addWidget(NavigationToolbar2QT(self._canvas, self))
        plot_col.addWidget(self._canvas, stretch=1)

        # Edits stay pending (dirty) until Apply, which writes them to the settings and replots
        self._form = F0ScanForm(self.vm.settings, self._replot)
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

        self._folder_edit = QLineEdit(str(self.vm.folder))
        self._folder_edit.textChanged.connect(self._on_folder_changed)
        self._folder_edit.returnPressed.connect(self._replot)
        self._molecule_edit = QLineEdit(self.vm.molecule)
        self._molecule_edit.setFixedWidth(100)
        self._molecule_edit.textChanged.connect(self._on_molecule_changed)
        self._molecule_edit.returnPressed.connect(self._replot)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._on_browse_folder)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._replot)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(QLabel("Folder"))
        bottom_row.addWidget(self._folder_edit, stretch=1)
        bottom_row.addWidget(browse_button)
        bottom_row.addWidget(QLabel("Molecule"))
        bottom_row.addWidget(self._molecule_edit)
        bottom_row.addWidget(refresh_button)

        self.body_layout.addLayout(plot_row, stretch=1)
        self.body_layout.addLayout(bottom_row)
        self._replot()

    def _show_error(self, e: Exception) -> None:
        self._ax.set_title(f"Error: {e}", color="r")
        self._canvas.draw_idle()
        self.vm.report_error(e)

    def _replot(self) -> None:
        if self._twin_ax is not None:
            self._twin_ax.remove()
            self._twin_ax = None
        self._ax.clear()
        try:
            self._twin_ax = plot_f0_scan(
                self._ax,
                self.vm.folder,
                self.vm.molecule,
                build_config(self.vm.settings),
            )
        except Exception as e:
            self._show_error(e)
            return
        self._canvas.draw_idle()

    def _on_folder_changed(self, text: str) -> None:
        self.vm.folder = Path(text.strip())

    def _on_molecule_changed(self, text: str) -> None:
        self.vm.molecule = text.strip()

    def _on_browse_folder(self) -> None:
        folder = self.vm.folder
        initial_dir = str(folder) if folder.exists() else ""
        selected = QFileDialog.getExistingDirectory(self, "Select scan folder", initial_dir)
        if not selected:
            return
        self._folder_edit.setText(str(Path(selected)))
        self._replot()


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    install_ui(app)
    bus = EventBus()
    dispatcher = QtDispatcher()

    window = LabMainWindow("F0 scan", bus, dispatcher)
    vm = F0ScanViewModel(bus, dispatcher, F0ScanSettings(), Path(DEFAULT_FOLDER), DEFAULT_MOLECULE)
    window.setCentralWidget(F0ScanPanel(vm))
    window.resize(1400, 700)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
