from pathlib import Path
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from yaml import scan
from matplotlib.widgets import Button

from _domain.plotting import plot_GaussianFit
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.c2t_calculation.domain.plotting import plot_calculated_scan
from apps.scan_averaging.domain import averaging
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from _data_io.dat_finder import MOST_RECENT_FOLDER, DatFinder
from _data_io.dat_loader import load_ion_data
from apps.scan_averaging.domain import averaging
#from apps.scan_averaging.domain.models import AveragedScansData
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from apps.single_scan.domain.plotting import plot_single_scan
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time
from base_core.math.models import Point, Angle, Range
from base_core.math.enums import AngleUnit

#folder_path = Path(r"/mnt/valeryshare/Droplets/20260417/Scan6")


mpl.rcParams.update({
    
    # --- Axes ---
    "axes.titlesize": "medium",
    "axes.labelsize": 8,
    "axes.formatter.use_mathtext": True,
    "axes.linewidth": 0.5,
    "axes.grid": True,
    "axes.grid.axis": "both",  # which axis the grid should apply to
    "axes.grid.which": "major",
    "axes.axisbelow" : True,
    "grid.alpha": 1.0,

    # --- Grid lines ---
    "grid.linewidth": 0.3,
    "grid.linestyle": "solid",
    "grid.color": "grey",

    # --- Lines ---
    "lines.linewidth": 0.5,
    "lines.marker": "o",
    "lines.markersize": 1.0,
    "hatch.linewidth": 0.25,
    "patch.antialiased": True,
    
    #---Errorbars---
    "errorbar.capsize": 1,

    # --- Ticks (X) ---
    #"xtick.top": True,
    "xtick.bottom": True,
    "xtick.major.size": 3.0,
    "xtick.minor.size": 1.5,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "xtick.direction": "in",
    "xtick.minor.visible": False,
    #"xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.bottom": True,
    "xtick.major.pad": 5.0,
    "xtick.minor.pad": 5.0,
    "xtick.labelsize": 8,

    # --- Ticks (Y) ---
    "ytick.left": True,
    #"ytick.right": True,
    "ytick.major.size": 3.0,
    #"ytick.minor.size": 1.5,
    "ytick.major.width": 0.5,
    #"ytick.minor.width": 0.5,
    "ytick.direction": "in",
    #"ytick.minor.visible": True,
    "ytick.major.left": True,
    #"ytick.major.right": True,
    #"ytick.minor.left": True,
    "ytick.major.pad": 2.0,
    #"ytick.minor.pad": 5.0,
    "ytick.labelsize": 8,
    
    
    # --- Legend ---
    "legend.frameon": True,
    "legend.fontsize": 8,
    "legend.handlelength": 1.375,
    "legend.labelspacing": 0.4,
    "legend.columnspacing": 1,
    "legend.facecolor": "white",
    "legend.edgecolor": "white",
    "legend.framealpha": 1,
    "legend.title_fontsize": 8,
 
    # --- Figure size ---
    #"figure.figsize": (3.375, 3.6), #1- column fig
    #"figure.figsize": (3.375, 3),
    #"figure.figsize": (6.75, 3.6), #approx. 2- column fig
    "figure.subplot.left": 0.125,
    "figure.subplot.bottom": 0.175,
    "figure.subplot.top": 0.95,
    "figure.subplot.right": 0.95,

    # --- Fonts (computer modern) ---
    "font.size": 8,
    "text.usetex": True,      
    #"mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"]

})

def get_scan_number(folder_path: Path = MOST_RECENT_FOLDER) -> int:
        scan_files = DatFinder(folder_path).find_scanfiles()
        return len(scan_files)
    

vmi_config: list[IonDataAnalysisConfig] = []

# vmi_config.append(IonDataAnalysisConfig(    #05/04 cs2 droplet scan
#     delay_center= Length(93.3, Prefix.MILLI),
#     center=Point(206, 194),
#     angle= Angle(12, AngleUnit.DEG),
#     analysis_zone= Range[int](60, 110),
#     transform_parameter=0.78))

# vmi_config.append(IonDataAnalysisConfig( #05/05 cs2 droplet scans
#     delay_center= Length(93.3, Prefix.MILLI),
#     center=Point(204, 196),
#     angle= Angle(12, AngleUnit.DEG),
#     analysis_zone= Range[int](60, 110),
#     transform_parameter=0.78))

# vmi_config.append(vmi_config[1])

# vmi_config.append(IonDataAnalysisConfig( #CS2 Jet 05/11 config
#     delay_center= Length(93.3, Prefix.MILLI),
#     center=Point(228, 186),
#     angle= Angle(12, AngleUnit.DEG),
#     analysis_zone= Range[int](40, 110),
#     transform_parameter=0.78))

folder_path: list[Path] = []
#folder_path.append(Path(r"Z:\Droplets\20260504\Scan3")) #earlier times included
#folder_path.append(Path(r"Z:\Droplets\20260505\Scan2")) #''         ''      ''
#folder_path.append(Path(r"Z:\Droplets\20260505\Scan3 stabilized only"))

def main() -> None:
    folder_path.append(Path(r"Z:\Droplets\20260511\Scan1"))
    

    vmi_config.append(IonDataAnalysisConfig(
    delay_center= Length(93.3, Prefix.MILLI),
    center=Point(228, 186),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](40, 110),
    transform_parameter=0.78))
    
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(8,5),gridspec_kw={"hspace":0})
    plt.subplots_adjust(bottom=0.2)  
    button_ax = fig.add_axes((0.8, 0.05, 0.15, 0.075))
    ax1.xaxis.label.set_visible(False)
    ax1.tick_params(axis='x',which='both',bottom=False, labelbottom=False)
    ax2.tick_params(axis='x',which='both',direction='out')
    yticks = ax2.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    
    refresh_button = Button(button_ax, "Refresh")

    def on_refresh(event):
        files = DatFinder(folder_path,is_full_path=True)
        datafile_paths = files.find_datafiles()
        num_scans = get_scan_number()
        
        raw_scans = load_ion_data(datafile_paths)
        calculated_Scans = run_pipeline(raw_scans, vmi_config)   
        
        start = Time(-1000,prefix=Prefix.PICO)
        end = Time(2000,prefix=Prefix.PICO)
        truncated_Scans = [c2tscan.cut(start,end) for c2tscan in calculated_Scans]
        

        stft_config = StftAnalysisConfig(truncated_Scans, Time(180, Prefix.PICO))
        resampled_scans = resample_scans(truncated_Scans,stft_config.axis)
        averaged_data = averaging.average_scans(truncated_Scans)
        resampled_scan = resample_scans([averaged_data],stft_config.axis)
        _, baseline = resampled_scan[0].detrend_moving_average()

        ax1.clear()
        ax2.clear()
        
        if len(averaged_data.run_ids) > 1:
            plot_averaged_scan(ax1,data=averaged_data,ecolor=PlotColor.RED)
        else:
            plot_calculated_scan(ax1,data=averaged_data,number_of_scans=num_scans)
        
        #x = [time.value(Prefix.PICO) for time in resampled_scan[0].delays]
        #ax1.plot(x, baseline)
        ax1.grid(visible=True,which='major',alpha=1.0)
        spectrogram = StftAnalysis(resampled_scans,stft_config).calculate_averaged_spectrogram()
        #xrange = [-300,300]
        #ax1.set_xlim(xrange)
        plot_Spectrogram(ax2,spectrogram,shading='auto')
        plot_nyquist_frequency(ax2,truncated_Scans[0])
        #ax2.set_xlim(xrange)
        fig.suptitle('CS2 Jet', fontsize=12)
        fig.canvas.draw_idle()
    
    refresh_button.on_clicked(on_refresh)
    plt.show()
    
if __name__ == "__main__":
    main()