from dataclasses import replace
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, TextBox

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data

from apps.c2t_calculation.domain.plotting import plot_ions_square
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.math.special_models import AngularCovariance
from base_core.plotting.covariance_plotting import plot_covariance
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length


# ============================================================
# Defaults (all editable on the figure)
# ============================================================
# Either a single .dat file or a folder of data files
DEFAULT_PATH = r"Z:\Droplets\20261103\ProbeOnly\20261103132926DLY__15p7950mm.dat"

# (label, default) for each widget in the right-hand column
CONFIG_DEFAULTS = {
    "delay_center": ("Delay center (mm) ", 93.3),
    "center_x": ("Center x ", 98),
    "center_y": ("Center y ", 102),
    "angle": ("Angle (deg) ", 12),
    "zone_min": ("Zone min ", 10),
    "zone_max": ("Zone max ", 120),
    "transform_parameter": ("ScaleX ", 0.76),
    "scan_index": ("Scan index ", 0),
    "ion_data_index": ("IonData index ", 0),
}

# (label, default) for the AngularCovariance widgets; leave a radial box blank for no radial cut
COVARIANCE_DEFAULTS = {
    "angle_bins": ("Angle bins ", "180"),
    "radial_min": ("Radial min ", ""),
    "radial_max": ("Radial max ", ""),
}
DEFAULT_BINARY_PER_FRAME = False
DEFAULT_NORMALIZE = True

# Optional alternative:
# set this to a concrete stage position if you want to select by value
TARGET_STAGE_POSITION = None
# example:
# TARGET_STAGE_POSITION = Length(0.0, Prefix.MILLI)


def build_config(values: dict[str, float]) -> IonDataAnalysisConfig:
    return IonDataAnalysisConfig(
        delay_center=Length(values["delay_center"], Prefix.MILLI),
        center=Point(values["center_x"], values["center_y"]),
        angle=Angle(values["angle"], AngleUnit.DEG),
        analysis_zone=Range[int](int(values["zone_min"]), int(values["zone_max"])),
        transform_parameter=values["transform_parameter"],
    )


def ask_for_file(fig, current_path: Path) -> str:
    # Reuse matplotlib's Tk window when it has one, otherwise make a hidden one
    get_tk_widget = getattr(fig.canvas, "get_tk_widget", None)
    root = None if get_tk_widget else tk.Tk()
    if root is not None:
        root.withdraw()
        root.attributes("-topmost", True)
    initial_dir = current_path.parent if current_path.is_file() else current_path
    try:
        return filedialog.askopenfilename(
            parent=get_tk_widget() if get_tk_widget else root,
            initialdir=initial_dir if initial_dir.exists() else None,
            filetypes=[("Data files", "*.dat"), ("All files", "*.*")],
        )
    finally:
        if root is not None:
            root.destroy()


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


def build_radial_range(radial_min: str, radial_max: str) -> Range[float] | None:
    if not radial_min.strip() and not radial_max.strip():
        return None
    if not radial_min.strip() or not radial_max.strip():
        raise ValueError("Set both radial min and max, or leave both blank.")
    return Range[float](float(radial_min), float(radial_max))


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


def main() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
    # Reserve room for the widgets below and to the right
    fig.subplots_adjust(left=0.05, bottom=0.18, right=0.76, wspace=0.3)

    def column_axes(row: int):
        return fig.add_axes((0.88, 0.88 - 0.06 * row, 0.08, 0.045))

    config_boxes = {}
    for i, (key, (label, default)) in enumerate(CONFIG_DEFAULTS.items()):
        config_boxes[key] = TextBox(column_axes(i), label, initial=str(default))

    covariance_boxes = {}
    for i, (key, (label, default)) in enumerate(COVARIANCE_DEFAULTS.items(), start=len(config_boxes)):
        covariance_boxes[key] = TextBox(column_axes(i), label, initial=default)

    binary_axes = column_axes(len(config_boxes) + len(covariance_boxes))
    binary_axes.set_frame_on(False)
    binary_check = CheckButtons(binary_axes, ["Binary per frame"], [DEFAULT_BINARY_PER_FRAME])

    normalize_axes = column_axes(len(config_boxes) + len(covariance_boxes) + 1)
    normalize_axes.set_frame_on(False)
    normalize_check = CheckButtons(normalize_axes, ["Normalize"], [DEFAULT_NORMALIZE])

    path_box = TextBox(fig.add_axes((0.08, 0.03, 0.39, 0.05)), "Path ", initial=DEFAULT_PATH)
    browse_button = Button(fig.add_axes((0.48, 0.03, 0.08, 0.05)), "Browse...")
    auto_center_button = Button(fig.add_axes((0.58, 0.03, 0.11, 0.05)), "Auto center")
    refresh_button = Button(fig.add_axes((0.71, 0.03, 0.11, 0.05)), "Refresh")
    save_button = Button(fig.add_axes((0.84, 0.03, 0.11, 0.05)), "Save")

    def read_values() -> dict[str, float]:
        return {key: float(box.text) for key, box in config_boxes.items()}

    def read_ion_data(values: dict[str, float]):
        return load_selected_ion_data(
            Path(path_box.text.strip()),
            int(values["scan_index"]),
            int(values["ion_data_index"]),
        )

    colorbars = []

    def on_refresh(event):
        for colorbar in colorbars:
            colorbar.remove()
        colorbars.clear()
        ax_l.clear()
        ax_r.clear()
        try:
            values = read_values()
            colorbars.append(plot_ions_and_covariance(
                fig,
                ax_l,
                ax_r,
                read_ion_data(values),
                build_config(values),
                int(covariance_boxes["angle_bins"].text),
                build_radial_range(covariance_boxes["radial_min"].text, covariance_boxes["radial_max"].text),
                binary_check.get_status()[0],
                normalize_check.get_status()[0],
            ))
        except Exception as e:
            ax_l.set_title(f"Error: {e}", color="r")
        fig.canvas.draw_idle()

    def on_auto_center(event):
        try:
            centroid = calculate_centroid(read_ion_data(read_values()))
        except Exception as e:
            ax_l.set_title(f"Error: {e}", color="r")
            fig.canvas.draw_idle()
            return
        print(f"Centroid: ({centroid.x:.1f}, {centroid.y:.1f})")
        config_boxes["center_x"].set_val(f"{centroid.x:.1f}")
        config_boxes["center_y"].set_val(f"{centroid.y:.1f}")
        on_refresh(event)

    def on_browse(event):
        selected = ask_for_file(fig, Path(path_box.text.strip()))
        if not selected:
            return
        path_box.set_val(str(Path(selected)))
        on_refresh(event)

    # ============================================================
    # Save single plot
    # ============================================================
    def on_save(event):
        path = Path(path_box.text.strip())
        out_dir = (path.parent if path.is_file() else path) / "single_covariance_with_ions"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "selected_iondata_covariance.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to: {out_path}")

    refresh_button.on_clicked(on_refresh)
    save_button.on_clicked(on_save)
    auto_center_button.on_clicked(on_auto_center)
    browse_button.on_clicked(on_browse)
    on_refresh(None)

    plt.show()


if __name__ == "__main__":
    main()
