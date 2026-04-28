from pathlib import Path
import traceback
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, CheckButtons
from matplotlib import collections, contour, artist
from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from _domain.plotting import plot_ScanData
from apps.scan_averaging.domain import averaging
from base_core.lab_specifics.base_models import IonData, IonDataAnalysisConfig, RawScanData
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Histogram2D, Point, Range
from base_core.plotting.histogram_plotting import plot_contour, plot_histogram2d
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.c2t_calculation.domain.plotting import plot_calculated_scan
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import StftAnalysis



BINS = 50
MIN_COUNT = 10
POSZEROSHIFT = 0.0
contours_toggle = [True]
hist_toggle = [True]

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

def add_labeled_checkbox(
    fig: plt.Figure,
    x: float,
    y: float,
    width: float,
    height: float,
    label: list[str],
    toggle: bool,
) -> CheckButtons:
    ax_cb = fig.add_axes((x,y,width,height))
    ax_cb.set_frame_on(False)
    return CheckButtons(ax_cb,label,actives=toggle)


def main() -> None:
    folder_path = Path(r"/mnt/valeryshare/Droplets/20260417/Scan5")
    file_paths = DatFinder(folder_path,is_full_path=True).find_datafiles()

    raw_scans = load_ion_data(file_paths)
    
    #Choose what scan you want to analyze by changing the index here:
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
    y_info = 0.08

    #Upper check buttons
    y_up = 0.9
    
    
    cb_contours = add_labeled_checkbox(
        fig,
        x_mid,
        y_up,
        small_w,
        box_h,
        ["Contours"],
        contours_toggle,
    )
    
    cb_hist = add_labeled_checkbox(
        fig,
        x_mid + small_w + gap,
        y_up,
        small_w,
        box_h,
        ["2DHistogram"],
        hist_toggle,
    )

    tb_bins = add_labeled_textbox(
        fig,
        x_mid,
        y_delay+0.1,
        small_w,
        box_h,
        "Bins",
        f"{BINS}",
    )
    
    tb_delay = add_labeled_textbox(
        fig,
        x_mid,
        y_delay,
        full_w,
        box_h,
        "File path",
        folder_path,
    )

    tb_center_x = add_labeled_textbox(
        fig,
        x_mid,
        y_center,
        small_w,
        box_h,
        "center x",
        "99",
    )
    tb_center_y = add_labeled_textbox(
        fig,
        x_mid + small_w + gap,
        y_center,
        small_w,
        box_h,
        "center y",
        "106",
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
        "20",
    )
    tb_range_max = add_labeled_textbox(
        fig,
        x_mid + small_w + gap,
        y_range,
        small_w,
        box_h,
        "range max",
        "50",
    )

    tb_transform = add_labeled_textbox(
        fig,
        x_mid,
        y_transform,
        full_w,
        box_h,
        "transform",
        "0.95",
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

    def plot_c2t(ax, raw_scans: list[RawScanData], config: IonDataAnalysisConfig) -> None:
        c2t_data = run_pipeline(raw_scans,config)
        c2t_avg = averaging.average_scans(c2t_data)
        plot_ScanData(ax,c2t_avg)
        
    
    def draw_hist(ax, hist: Histogram2D, title: str) -> collections.QuadMesh:
        ax.clear()
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_box_aspect(1)
        return plot_histogram2d(ax, hist)

    def draw_contours(ax, hist: Histogram2D) -> contour.QuadContourSet:
        return plot_contour(ax, hist)
    
    
    def on_refresh(_event):
        global hist_artist, contour_artist
        try:
            config = build_config()
            points_after = ion_data.get_points_after_config(config)
            
            bins = int(tb_bins.text)
            hist_before = Histogram2D.compute_histogram(points_before,bins,bins)
            hist_after = Histogram2D.compute_histogram(points_after,bins,bins)
            draw_hist(ax_left, hist_before, "points_before")
            draw_contours(ax_left,hist_before)
            c2t = ion_data.avg_c2t(points_after)
            c2t_text.set_text(f"c2t: {c2t.value:.5f} ± {c2t.error:.4f}")
            error_text.set_text("")
            hist_artist = draw_hist(ax_right, hist_after, "After coordinate transformation")
            contour_artist = draw_contours(ax_right,hist_after)

        except Exception as exc:
            error_text.set_text(f"Error: {exc}")
            

        fig.canvas.draw_idle()

    def on_toggle_contours(_event):
        try:
            visible = contour_artist.get_visible()
            #for c in contour_artist:
                #c.set_visible(not visible)
            contour_artist.set_visible(not visible)
        except Exception as exc:
            error_text.set_text(f"Error: {exc}")      
        
        fig.canvas.draw_idle()  
    def on_toggle_hist(_event):
        try:
            visible = hist_artist.get_visible()
            hist_artist.set_visible(not visible)
        except Exception as exc:
            error_text.set_text(f"Error: {exc}")   
            
        fig.canvas.draw_idle()
    # ------------------------------------------------------------------
    # Widget connections
    # ------------------------------------------------------------------

    for tb in (
        tb_bins,
        tb_delay,
        tb_center_x,
        tb_center_y,
        tb_angle,
        tb_range_min,
        tb_range_max,
        tb_transform,
    ):
        tb.on_submit(on_refresh)
        
    cb_contours.on_clicked(on_toggle_contours)
    cb_hist.on_clicked(on_toggle_hist)
    on_refresh(None)
    plt.show()


if __name__ == "__main__":
    main()