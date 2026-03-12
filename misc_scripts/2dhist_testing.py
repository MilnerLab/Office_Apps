from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

from _data_io.dat_loader import load_ion_data
from base_core.lab_specifics.base_models import IonDataAnalysisConfig, RawScanData
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Histogram2D, Point, Range
from base_core.plotting.histogram_plotting import plot_contour, plot_histogram2d
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length


BINS = 100
MIN_COUNT = 1000
POSZEROSHIFT = 0.0


def add_labeled_textbox(
    fig: plt.Figure,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    initial: str,
) -> TextBox:
    fig.text(x, y + height + 0.008, label, fontsize=10, ha="left", va="bottom")
    ax_box = fig.add_axes((x, y, width, height))
    return TextBox(ax_box, "", initial=initial)


def main() -> None:
    file_path = Path(r"/mnt/valeryshare/Droplets/20260312/Scan1/20260312111952DLY___4p5000mm.dat")
    # file_path = DatFinder().find_most_recent_scanfile()

    raw_scans = load_ion_data([[file_path]])
    raw_scan: RawScanData = raw_scans[0]
    ion_data = raw_scan.ion_datas[0]

    points_before = ion_data.points

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 7), facecolor="white")

    ax_left = fig.add_axes((0.05, 0.18, 0.34, 0.72))
    ax_right = fig.add_axes((0.61, 0.18, 0.34, 0.72))

    # Middle control column
    x_mid = 0.43
    full_w = 0.14
    small_w = 0.065
    gap = 0.01
    box_h = 0.055

    y_delay = 0.72
    y_center = 0.61
    y_angle = 0.50
    y_range = 0.39
    y_transform = 0.28
    y_button = 0.17
    y_info = 0.08

    tb_delay = add_labeled_textbox(
        fig,
        x_mid,
        y_delay,
        full_w,
        box_h,
        "delay [mm]",
        str(94.5 - POSZEROSHIFT),
    )

    tb_center_x = add_labeled_textbox(
        fig,
        x_mid,
        y_center,
        small_w,
        box_h,
        "center x",
        "171",
    )
    tb_center_y = add_labeled_textbox(
        fig,
        x_mid + small_w + gap,
        y_center,
        small_w,
        box_h,
        "center y",
        "208",
    )

    tb_angle = add_labeled_textbox(
        fig,
        x_mid,
        y_angle,
        full_w,
        box_h,
        "angle [deg]",
        "12",
    )

    tb_range_min = add_labeled_textbox(
        fig,
        x_mid,
        y_range,
        small_w,
        box_h,
        "range min",
        "60",
    )
    tb_range_max = add_labeled_textbox(
        fig,
        x_mid + small_w + gap,
        y_range,
        small_w,
        box_h,
        "range max",
        "120",
    )

    tb_transform = add_labeled_textbox(
        fig,
        x_mid,
        y_transform,
        full_w,
        box_h,
        "transform",
        "0.75",
    )

    info_ax = fig.add_axes((x_mid - 0.005, y_info, full_w + 0.03, 0.07))
    info_ax.axis("off")
    c2t_text = info_ax.text(0.0, 0.60, "c2t: -", fontsize=12)
    error_text = info_ax.text(0.0, 0.05, "", fontsize=10, color="red")

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def build_config() -> IonDataAnalysisConfig:
        return IonDataAnalysisConfig(
            delay_center=Length(float(tb_delay.text), Prefix.MILLI),
            center=Point(
                int(float(tb_center_x.text)),
                int(float(tb_center_y.text)),
            ),
            angle=Angle(float(tb_angle.text), AngleUnit.DEG),
            analysis_zone=Range[float](
                float(tb_range_min.text),
                float(tb_range_max.text),
            ),
            transform_parameter=float(tb_transform.text),
        )

    def draw_hist(ax, points, title: str) -> None:
        ax.clear()

        hist = Histogram2D.compute_histogram(points, x_bins=BINS, y_bins=BINS)
        plot_histogram2d(ax, hist)
        # plot_contour(ax, hist, min_count=MIN_COUNT)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_box_aspect(1)

    def on_refresh(_event) -> None:
        try:
            config = build_config()
            points_after = ion_data.get_points_after_config(config)

            draw_hist(ax_left, points_before, "points_before")
            draw_hist(ax_right, points_after, "points_after")

            c2t = ion_data.avg_c2t(points_after)
            c2t_text.set_text(f"c2t: {c2t.value:.6f} ± {c2t.error:.6f}")
            error_text.set_text("")

        except Exception as exc:
            error_text.set_text(f"Error: {exc}")

        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Widget connections
    # ------------------------------------------------------------------

    for tb in (
        tb_delay,
        tb_center_x,
        tb_center_y,
        tb_angle,
        tb_range_min,
        tb_range_max,
        tb_transform,
    ):
        tb.on_submit(on_refresh)

    on_refresh(None)
    plt.show()


if __name__ == "__main__":
    main()