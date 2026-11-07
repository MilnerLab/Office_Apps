from dataclasses import dataclass, replace
from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colorbar import make_axes
from matplotlib.figure import Figure
from matplotlib.widgets import RangeSlider
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data

from apps.c2t_calculation.domain.plotting import plot_ions_square
from base_core.framework.events import EventBus
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, MarkedPoints, Point, Range
from base_core.math.special_models import AngularCovariance
from base_core.plotting.covariance_plotting import plot_covariance
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length
from base_qt.app.dispatcher import QtDispatcher
from base_qt.ui import LabMainWindow, MessageLevel, Panel, PanelViewModel
from base_qt.ui.apply import install_ui
from base_qt.ui.form import AngleSpec, BoolSpec, DirtyFormWidget, FieldSpec, FloatSpec, IntSpec, LengthSpec, RangeSpec


# ============================================================
# Defaults (all editable in the window)
# ============================================================
# Either a single .dat file or a folder of data files
DEFAULT_PATH = r"Z:\Droplets\20261106\Droplets_Cov2"
# Hits within this distance (detector pixels) of a masked brightest pixel's center are removed
MASK_RADIUS = 1.0


@dataclass
class CovarianceSettings:
    # IonDataAnalysisConfig
    delay_center: Length = Length(93.3, Prefix.MILLI)
    center_x: float = 108
    center_y: float = 100.0
    angle: Angle = Angle(12, AngleUnit.DEG)
    analysis_zone: Range[int] = Range[int](15, 50)
    transform_parameter: float = 0.8
    # Which IonData to show
    scan_index: int = 0
    ion_data_index: int = 0
    # Drop all hits within MASK_RADIUS of the N most-hit integer detector pixels
    # (e.g. hot pixels); 0 = off
    mask_brightest_n: int = 1
    # Keep only frames with exactly n_body ions (or at least n_body, while n_body_at_least
    # is on) inside the analysis zone; only used while n_body_filter is on
    n_body_filter: bool = False
    n_body: int = 2
    n_body_at_least: bool = False
    # AngularCovariance; the radial range is only used while radial_cut is on
    angle_bins: int = 180
    radial_cut: bool = False
    radial_range: Range[float] = Range[float](0.0, 120.0)
    binary_per_frame: bool = False
    # Gaussian blur of the covariance map, FWHM in angle bins; only used while blur is on
    blur: bool = False
    blur_fwhm_bins: float = 1.5
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


def mask_brightest_coordinates(ion_data, n: int, radius: float = MASK_RADIUS):
    # Bin hits into integer detector pixels, find the n pixels with the most hits, then
    # remove every hit within `radius` of any of those pixels' centers. Works in detector
    # coordinates, so the mask does not depend on the center/angle/scale config.
    # Returns the masked IonData and the number of hits removed.
    if n <= 0 or len(ion_data.points) == 0:
        return ion_data, 0
    points = ion_data.points
    xy = np.column_stack((points.x, points.y))
    # floor, not round: many hits sit on exact .5 values, which np.round sends to even
    unique_pixels, counts = np.unique(np.floor(xy), axis=0, return_counts=True)
    # Stable sort, so ties are broken the same way on every replot
    brightest = np.argsort(-counts, kind="stable")[:n]
    for i in brightest:
        print(f"Masking around pixel ({unique_pixels[i, 0]:g}, {unique_pixels[i, 1]:g}): {counts[i]} hits")
    # Pixel k covers [k, k+1), so measure distance from its center
    centers = unique_pixels[brightest] + 0.5
    # Distance from each hit to its nearest masked center; inf when none is in range.
    # The query's bound is strict and the 0.1-step floats are not exact, so allow a little
    # slack to keep hits at exactly `radius` inside the mask.
    bound = radius + 1e-6
    distance, _ = cKDTree(centers).query(xy, distance_upper_bound=bound)
    keep = distance > bound
    return replace(ion_data, points=points.filter_by_mask(keep)), int(np.sum(~keep))


def filter_n_body(hits: MarkedPoints, n: int, at_least: bool = False) -> MarkedPoints:
    # Keep only the hits whose frame (marker) has exactly n hits in `hits`, or n or more
    # when at_least is set
    if len(hits) == 0:
        return hits
    _, inverse, counts = np.unique(hits.marker, return_inverse=True, return_counts=True)
    per_hit_counts = counts[inverse.reshape(-1)]
    return hits.filter_by_mask(per_hit_counts >= n if at_least else per_hit_counts == n)


def calculate_centroid(ion_data) -> Point:
    # Mean position of all raw hits, in detector coordinates
    return Point(float(np.mean(ion_data.points.x)), float(np.mean(ion_data.points.y)))


# FWHM = 2 * sqrt(2 ln 2) * sigma
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def blur_covariance(cov: AngularCovariance, fwhm_bins: float) -> AngularCovariance:
    # Both axes span the full [0, 2*pi), so the blur wraps around at the edges
    if fwhm_bins <= 0:
        return cov
    sigma = fwhm_bins / FWHM_PER_SIGMA
    return replace(cov, matrix=gaussian_filter(cov.matrix, sigma=sigma, mode="wrap"))


def normalize_covariance(cov: AngularCovariance) -> AngularCovariance:
    # Scale to [0, 1]; negatives are clipped, same as plot_covariance does by default
    matrix = np.maximum(cov.matrix, 0.0)
    peak = float(np.max(matrix)) if matrix.size else 0.0
    return replace(cov, matrix=matrix / peak if peak > 0 else matrix)


def plot_ions_and_covariance(ax_l, ax_r, ion_data, config: IonDataAnalysisConfig, angle_bins: int, radial_range: Range[float] | None, binary_per_frame: bool, blur_fwhm_bins: float | None, normalize: bool, mask_text: str = "", n_body: int | None = None, n_body_at_least: bool = False):
    # ============================================================
    # Prepare hits
    # ============================================================
    # The config already drops hits outside the analysis zone
    hits = ion_data.get_points_after_config(config)

    print(f"hits={len(hits)}")

    if len(hits) == 0:
        raise ValueError("Selected IonData has no hits after config.")

    # N-body filter: count ions per frame inside the zone, keep only frames with exactly N
    # (or at least N)
    n_body_text = ""
    if n_body is not None:
        hits = filter_n_body(hits, n_body, n_body_at_least)
        n_body_frames = hits.n_observed_markers
        n_label = f"≥{n_body}" if n_body_at_least else f"{n_body}"
        print(f"N-body filter (N{'' if n_body_at_least else '='}{n_label}): {n_body_frames} frames, {len(hits)} hits kept")
        if len(hits) == 0:
            raise ValueError(
                f"No frames with {'at least' if n_body_at_least else 'exactly'} {n_body} ions in the zone."
            )
        n_body_text = f"\n{n_label}-body frames={n_body_frames}"
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
        f"markers={hits.marker_max}, hits={len(hits)}{mask_text}{n_body_text}"
    )
    ax_l.set_aspect("equal")

    # Blur before normalizing, so the peak is taken from the smoothed map
    if blur_fwhm_bins is not None:
        cov = blur_covariance(cov, blur_fwhm_bins)
    if normalize:
        cov = normalize_covariance(cov)

    mesh = plot_covariance(ax_r, cov)
    blur_text = f", blur FWHM={blur_fwhm_bins:g} bins" if blur_fwhm_bins is not None else ""
    ax_r.set_title(
        "Angular covariance\n"
        f"frames={cov.n_frames}{blur_text}"
    )
    return mesh


# ============================================================
# GUI
# ============================================================
class _FloatSliderWidget(QWidget):
    # A slider with a spin box beside it, kept in sync; the spin box holds the value
    def __init__(self, lo: float, hi: float, decimals: int, step: float) -> None:
        super().__init__()
        scale = 10 ** decimals
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(round(lo * scale), round(hi * scale))
        self.slider.setSingleStep(max(1, round(step * scale)))
        self.slider.setPageStep(max(1, round(10 * step * scale)))
        self.spin_box = QDoubleSpinBox()
        self.spin_box.setRange(lo, hi)
        self.spin_box.setDecimals(decimals)
        self.spin_box.setSingleStep(step)
        self.spin_box.setFixedWidth(70)
        # Setting an unchanged value emits nothing, so the two signals cannot loop
        self.slider.valueChanged.connect(lambda v: self.spin_box.setValue(v / scale))
        self.spin_box.valueChanged.connect(lambda v: self.slider.setValue(round(v * scale)))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.spin_box)


class FloatSliderSpec(FieldSpec):
    def __init__(self, label: str, lo: float, hi: float, decimals: int = 1, step: float = 0.1) -> None:
        super().__init__(label)
        self._lo = lo
        self._hi = hi
        self._decimals = decimals
        self._step = step

    def create_widget(self) -> _FloatSliderWidget:
        return _FloatSliderWidget(self._lo, self._hi, self._decimals, self._step)

    def set_value(self, widget: _FloatSliderWidget, value: float) -> None:  # type: ignore[override]
        widget.spin_box.setValue(float(value))

    def get_value(self, widget: _FloatSliderWidget) -> float:  # type: ignore[override]
        return widget.spin_box.value()

    def connect_change(self, widget: _FloatSliderWidget, slot) -> None:  # type: ignore[override]
        widget.spin_box.valueChanged.connect(lambda _: slot())


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
        "mask_brightest_n": IntSpec("Mask brightest N", min=0, max=100_000),
        "n_body_filter": BoolSpec("N-body filter"),
        "n_body": IntSpec("N ions per frame", min=1, max=10_000),
        "n_body_at_least": BoolSpec("At least N (else exactly)"),
        "angle_bins": IntSpec("Angle bins", min=1, max=3600),
        "radial_cut": BoolSpec("Radial cut"),
        "radial_range": RangeSpec("Radial range", FloatSpec("", min=0.0, max=10_000.0, decimals=1, step=1.0)),
        "binary_per_frame": BoolSpec("Binary per frame"),
        "blur": BoolSpec("Gaussian blur"),
        "blur_fwhm_bins": FloatSliderSpec("Blur FWHM (bins)", 0.0, 10.0, decimals=1, step=0.1),
        "normalize": BoolSpec("Normalize"),
    }
    _groups = [
        ("Analysis config", ["delay_center", "center_x", "center_y", "angle", "analysis_zone", "transform_parameter"]),
        ("Data selection", ["scan_index", "ion_data_index", "mask_brightest_n"]),
        ("Covariance", ["n_body_filter", "n_body", "n_body_at_least", "angle_bins", "radial_cut", "radial_range", "binary_per_frame", "blur", "blur_fwhm_bins", "normalize"]),
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
        # Returns the masked IonData and the number of hits the mask removed
        ion_data = load_selected_ion_data(self.path, self.settings.scan_index, self.settings.ion_data_index)
        return mask_brightest_coordinates(ion_data, self.settings.mask_brightest_n)

    def radial_range(self) -> Range[float] | None:
        return self.settings.radial_range if self.settings.radial_cut else None

    def n_body(self) -> int | None:
        return self.settings.n_body if self.settings.n_body_filter else None

    def blur_fwhm_bins(self) -> float | None:
        return self.settings.blur_fwhm_bins if self.settings.blur else None

    def auto_center(self) -> None:
        ion_data, _ = self.load_ion_data()
        centroid = calculate_centroid(ion_data)
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
        # A colour-limit slider stands where the colorbar would. Its axes are made once,
        # after subplots_adjust, so replots do not keep shrinking ax_r to make room.
        self._cax, _ = make_axes(self._ax_r)
        self._cax.set_axis_off()
        self._clim_slider: RangeSlider | None = None
        # Handle positions as fractions of the colour range, kept across replots
        self._clim_frac = (0.0, 1.0)

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
        # Browse for either a single data file or a folder of data files
        browse_menu = QMenu(self)
        browse_menu.addAction("File...", self._on_browse_file)
        browse_menu.addAction("Folder...", self._on_browse_folder)
        browse_button = QPushButton("Browse")
        browse_button.setMenu(browse_menu)
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

    def _clear_clim_slider(self) -> None:
        if self._clim_slider is not None:
            self._clim_slider.disconnect_events()
            self._clim_slider = None
        self._cax.clear()
        self._cax.set_axis_off()

    def _build_clim_slider(self, mesh, label: str) -> None:
        # A colorbar whose two handles set the mesh's clim. The gradient shares the mesh's
        # norm, so it shows the same clipping as the map while the handles move.
        lo, hi = float(mesh.norm.vmin), float(mesh.norm.vmax)
        span = hi - lo
        cax = self._cax
        cax.imshow(
            np.linspace(lo, hi, 256).reshape(-1, 1),
            extent=(0.0, 1.0, lo, hi),
            origin="lower",
            aspect="auto",
            cmap=mesh.cmap,
            norm=mesh.norm,
        )
        init = (lo + self._clim_frac[0] * span, lo + self._clim_frac[1] * span)
        slider = RangeSlider(
            cax, "", lo, hi,
            valinit=init,
            orientation="vertical",
            valfmt="%.3g",
            track_color="none",
            facecolor="none",
            edgecolor="white",
            linewidth=1.0,
        )
        # RangeSlider hides its axes; bring back a colorbar-style value axis on the right
        cax.set_axis_on()
        cax.set_xlim(0.0, 1.0)
        cax.set_xticks([])
        cax.yaxis.set_ticks_position("right")
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(label)

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
        self._ax_l.clear()
        self._ax_r.clear()
        settings = self.vm.settings
        try:
            ion_data, masked_hits = self.vm.load_ion_data()
            n_masked = settings.mask_brightest_n
            mask_text = f"\nmasked {n_masked} brightest pixels ({masked_hits} hits)" if n_masked > 0 else ""
            mesh = plot_ions_and_covariance(
                self._ax_l,
                self._ax_r,
                ion_data,
                build_config(settings),
                settings.angle_bins,
                self.vm.radial_range(),
                settings.binary_per_frame,
                self.vm.blur_fwhm_bins(),
                settings.normalize,
                mask_text,
                self.vm.n_body(),
                settings.n_body_at_least,
            )
            self._build_clim_slider(mesh, "normalized covariance" if settings.normalize else "covariance")
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

    def _browse_start_dir(self) -> str:
        path = self.vm.path
        initial_dir = path.parent if path.is_file() else path
        return str(initial_dir) if initial_dir.exists() else ""

    def _on_browse_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select data file",
            self._browse_start_dir(),
            "Data files (*.dat);;All files (*.*)",
        )
        self._set_browsed_path(selected)

    def _on_browse_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select data folder", self._browse_start_dir())
        self._set_browsed_path(selected)

    def _set_browsed_path(self, selected: str) -> None:
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
