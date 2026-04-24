from pathlib import Path

import matplotlib.pyplot as plt

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data

from apps.c2t_calculation.domain.plotting import plot_ions_square
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, AngularCovariance, Point, Range
from base_core.plotting.covariance_plotting import plot_covariance
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length


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


def main() -> None:
    # ============================================================
    # Input
    # ============================================================
    folder_path = Path(
        r"/home/soeren/Downloads/20220124_PCS8_Covariance in Jet/PCS8/PCset_1/DLY_0p0000mm"
        #r"/mnt/valeryshare/Droplets/20260416/Scan6"
    )
    file_paths = DatFinder(folder_path, is_full_path=True).find_datafiles()
    
    

    config = IonDataAnalysisConfig(
        delay_center=Length(93.3, Prefix.MILLI),
        center=Point(99, 107),
        angle=Angle(12, AngleUnit.DEG),
        analysis_zone=Range[int](60, 100),
        #analysis_zone=Range[int](25, 65),
        transform_parameter=0.95,
    )
    
    config = IonDataAnalysisConfig(
        delay_center=Length(93.3, Prefix.MILLI),
        center=Point(150, 145),
        angle=Angle(12, AngleUnit.DEG),
        analysis_zone=Range[int](40, 90),
        #analysis_zone=Range[int](15, 35),
        transform_parameter=0.9,
    )

    # ============================================================
    # Selection settings
    # ============================================================
    SCAN_INDEX = 0
    ION_DATA_INDEX = 0

    # Optional alternative:
    # set this to a concrete stage position if you want to select by value
    TARGET_STAGE_POSITION = None
    # example:
    # TARGET_STAGE_POSITION = Length(0.0, Prefix.MILLI)

    # ============================================================
    # Load data
    # ============================================================
    raw_scans = load_ion_data(file_paths)

    if SCAN_INDEX < 0 or SCAN_INDEX >= len(raw_scans):
        raise IndexError(
            f"SCAN_INDEX={SCAN_INDEX} out of range for {len(raw_scans)} raw_scans."
        )

    raw_scan = raw_scans[SCAN_INDEX]

    print(f"Loaded {len(raw_scans)} scans.")
    print(f"Selected scan {SCAN_INDEX} with {len(raw_scan.ion_datas)} ion_datas.\n")

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
        ion_data = select_ion_data_by_index(raw_scan, ION_DATA_INDEX)

    print("\nSelected ion_data:")
    print(f"stage_position={ion_data.stage_position}")

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
        angle_bins=180,
        radial_range=None,
        binary_per_frame=False,
    )

    print(f"Computed covariance with n_frames={cov.n_frames}")

    # ============================================================
    # Plot ions + covariance
    # ============================================================
    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        constrained_layout=True,
    )

    plot_ions_square(
        ax_l,
        hits,
        color="red",
        label=None,
    )
    ax_l.set_title(
        "Ions\n"
        f"stage_position={ion_data.stage_position}, "
        f"markers={hits.marker_max}, hits={len(hits)}"
    )
    ax_l.set_aspect("equal")

    mesh = plot_covariance(ax_r, cov)
    ax_r.set_title(
        "Angular covariance\n"
        f"stage_position={ion_data.stage_position}, "
        f"frames={cov.n_frames}"
    )
    fig.colorbar(mesh, ax=ax_r, label="covariance")

    plt.show()

    # ============================================================
    # Save single plot
    # ============================================================
    out_dir = folder_path / "single_covariance_with_ions"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "selected_iondata_covariance.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()