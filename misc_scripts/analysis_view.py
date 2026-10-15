from pathlib import Path
import traceback
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, CheckButtons
import pyqtgraph as pg
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QWidget,
)
from matplotlib import collections, contour, artist
from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from base_core.lab_specifics.base_models import IonData,IonDataAnalysisConfig, RawScanData
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.math.special_models import Histogram2D
from base_core.math.models import Angle, Point, Range
from base_core.math.special_models import Histogram2D
from base_core.plotting.histogram_plotting import plot_contour, plot_histogram2d
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length


BINS = 50
MIN_COUNT = 10
POSZEROSHIFT = 0.0
DELAY = 93.3 
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


def main():
        
    
    folder_path = Path(r"Z:\Droplets\20260915\ProbeOnly")
    file_paths = DatFinder(folder_path,is_full_path=True).find_datafiles()

    raw_scans = load_ion_data(file_paths)
    raw_scan: RawScanData = raw_scans[0]
    ion_data = raw_scan.ion_datas[0]

    points_before = ion_data.points
    
    
    # ------------------------------------------------------------------
    # Panel layout
    # ------------------------------------------------------------------
    fig = plt.figure(facecolor="white")
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    
    
    #rect = (left,bottom,width,height)
    
    #Left c2t plot with spectrogram underneath
    col_l = 0.1
    c2t_rect = (col_l,0.5,0.5)
    
    ax_left = fig.add_axes((0.05, 0.18, 0.34, 0.72))
    
    
    #Right 2dhist column. 
    col_r = 0.6
    hist_rect = (col_r,0.5,0.5)

    ax_hist = fig.add_axes(hist_rect)
    
    #Bottom row
    row_bot = 0.1
    
    
    # Middle control column
    x_mid = 0.43
    full_w = 0.14
    small_w = 0.065
    gap = 0.01
    box_h = 0.055

    y_path = 0.72
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
        y_path+0.1,
        small_w,
        box_h,
        "Bins",
        f"{BINS}",
    )
    
   

    tb_center_x = add_labeled_textbox(
        fig,
        x_mid,
        y_center,
        small_w,
        box_h,
        "center x",
        "175",
    )
    tb_center_y = add_labeled_textbox(
        fig,
        x_mid + small_w + gap,
        y_center,
        small_w,
        box_h,
        "center y",
        "205",
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
        "0.76",
    )
    
    tb_path = add_labeled_textbox(
        fig,
        x_mid,
        y_path,
        full_w,
        box_h,
        "File path",
        folder_path,
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
            delay_center=Length(DELAY, Prefix.MILLI),
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