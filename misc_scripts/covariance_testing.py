from pathlib import Path

import matplotlib.pyplot as plt

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data

from apps.c2t_calculation.domain.plotting import plot_ions_square
from base_core.lab_specifics.base_models import IonData, IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, AngularCovariance, Point, Range
from base_core.plotting.covariance_plotting import plot_covariance
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length


def main() -> None:
    # ============================================================
    # Input
    # ============================================================
    folder_path = Path(
        r"/home/soeren/Downloads/20220124_PCS8_Covariance in Jet/PCS8/PCset_1/DLY_0p0000mm"
    )
    file_paths = DatFinder(folder_path, is_full_path=True).find_datafiles()

    config = IonDataAnalysisConfig(
        delay_center=Length(93.3, Prefix.MILLI),
        center=Point(200, 201),
        angle=Angle(-30,  AngleUnit.DEG),
        analysis_zone=Range[int](50, 110),
        transform_parameter=0.9,
    )

    # ============================================================
    # Load data
    # ============================================================
    raw_scans = load_ion_data(file_paths)
    raw_scan = raw_scans[0]

    # ============================================================
    # Compute covariance for every stage position
    # ============================================================
    covariances: list[tuple[IonData, AngularCovariance]] = []

    for ion_data in raw_scan.ion_datas:
        # This must return your marked / framed points type
        hits = ion_data.get_points_after_config(config)

        if len(hits) == 0:
            continue
        if hits.n_markers < 5:
            continue

        cov = AngularCovariance.compute_covariance(
            hits,
            angle_bins=90,
            radial_range=None,      # already filtered by analysis_zone above
            binary_per_frame=False, # start with count version
        )

        covariances.append((ion_data, cov))

    print(f"Computed {len(covariances)} covariance maps.")

    # ============================================================
    # Plot one example: ions left, covariance right
    # ============================================================
    if covariances:
        idx = len(covariances) // 2
        ion_data, cov = covariances[idx]
        hits = ion_data.get_points_after_config(config)

        fig, (ax_l, ax_r) = plt.subplots(
            1,
            2,
            figsize=(11, 5),
            constrained_layout=True,
        )

        # left: ion positions
        plot_ions_square(
            ax_l,
            hits,
            color="red",
            label=None,
        )
        ax_l.set_title(
            "Ions\n"
            f"stage_position={ion_data.stage_position}, "
            f"markers={hits.n_markers}, hits={len(hits)}"
        )
        ax_l.set_aspect("equal")

        # right: angular covariance
        mesh = plot_covariance(ax_r, cov)
        ax_r.set_title(
            "Angular covariance\n"
            f"stage_position={ion_data.stage_position}, "
            f"frames={cov.n_frames}"
        )
        fig.colorbar(mesh, ax=ax_r, label="covariance")

        plt.show()

    # ============================================================
    # Save all plots: ions left, covariance right
    # ============================================================
    out_dir = folder_path / "covariance_with_ions"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (ion_data, cov) in enumerate(covariances):
        hits = ion_data.get_points_after_config(config)

        fig, (ax_l, ax_r) = plt.subplots(
            1,
            2,
            figsize=(11, 5),
            constrained_layout=True,
        )

        # left: ion positions
        plot_ions_square(
            ax_l,
            hits,
            color="red",
            label=None,
        )
        ax_l.set_title(
            "Ions\n"
            f"stage_position={ion_data.stage_position}, "
            f"markers={hits.n_markers}, hits={len(hits)}"
        )
        ax_l.set_aspect("equal")

        # right: angular covariance
        mesh = plot_covariance(ax_r, cov)
        ax_r.set_title(
            "Angular covariance\n"
            f"stage_position={ion_data.stage_position}, "
            f"frames={cov.n_frames}"
        )
        fig.colorbar(mesh, ax=ax_r, label="covariance")

        fig.savefig(out_dir / f"{i:03d}_ions_covariance.png", dpi=150)
        plt.close(fig)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()