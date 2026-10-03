print('Code start!')
from pathlib import Path
from pydoc import locate
from altair import FontWeight
import matplotlib as mpl
from matplotlib import pyplot as plt

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from _data_io.dat_saver import create_save_path_for_calc_ScanFile
from _domain.plotting import plot_GaussianFit
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from apps.single_scan.domain.plotting import plot_single_scan
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.models import AggregateSpectrogram
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from base_core.lab_specifics.averaging.models import AveragedScansData
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time


STFTWINDOWSIZE = Time(180,Prefix.PICO)  
EARLIEST_DELAY_PS = -620
LATEST_DELAY_PS = 1370
POSZEROSHIFT = 0 #millimetres :)

MAJORTITLEFONTSIZE = 16
YLABELX = -0.135

#FUNCTION TO GENERATE THE PLOTTABLE DATA
def calculating(folders: list[Path], configs: list[IonDataAnalysisConfig]) -> tuple[AveragedScansData, AggregateSpectrogram]:
    
    
    scans_paths = DatFinder(folders).find_datafiles() #Change this if you want a specific path rather than the Droplets folder
    raw_datas = load_ion_data(scans_paths)  
    calculated_scans = run_pipeline(raw_datas, configs)
    averagedScanData = average_scans(calculated_scans)
    config = StftAnalysisConfig(calculated_scans, STFTWINDOWSIZE)
    resampled_scans = resample_scans(calculated_scans, config.axis)

    spectrogram = StftAnalysis(resampled_scans, config).calculate_averaged_spectrogram()
    
    return (averagedScanData, spectrogram)
#--------------------------------------------------------------------------------------------------

#Path to save figure in
fig_filedir = r"Z:\Droplets\plots" 
fig_filename = fig_filedir + r"\DIB_TEMP.png" #Name the file to save here

#Path to save processed data in
savedata_filedir = r"Z:\Droplets\exportdata" 
savedata_filename_1 = savedata_filedir + r"\DIB_TEMP.csv" #Name the file to save here


#Plot on top


PlotTitle = r"DIB in 40 bar / 13 K droplets" + "\n" + r"with centrifuge"

MINRADIUS = 40

#--------------------------------------------------------------------------------------------------------------
#Comparing different bandwidths of the centrifuge
#--------------------------------------------------------------------------------------------------------------
'''
configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []
folders.append(Path(r"Z:\Droplets\20260929\Scan4_AVG"))  #full centrifuge
configs.append(IonDataAnalysisConfig(
    delay_center= Length(70-POSZEROSHIFT, Prefix.MILLI),
    center=Point(200,200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS,90),
    transform_parameter=0.85))
folders_1 = folders
configs_1 = configs
label_1 = "11GHz - Full"
#--------------------------------------------------------------------------------------------------------------
configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []

folders.append(Path(r"Z:\Droplets\20261001\Scan1_CFG")) 
configs.append(IonDataAnalysisConfig(
    delay_center= Length(81.5-POSZEROSHIFT, Prefix.MILLI),
    center=Point(200,200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS,90),
    transform_parameter=0.85))
folders_2 = folders
configs_2 = configs
label_2 = "3GHz - Full"
#--------------------------------------------------------------------------------------------------------------
configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []
folders.append(Path(r"Z:\Droplets\20260929\Scan2_AVG")) 
configs.append(IonDataAnalysisConfig(
    delay_center= Length(80-POSZEROSHIFT, Prefix.MILLI),
    center=Point(200,200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS,90),
    transform_parameter=0.85))
folders_3 = folders
configs_3 = configs
label_3 = "5Ghz - Full"
'''
#--------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------
#Looking at field-free rotation with the 3GHz centrifuge
#--------------------------------------------------------------------------------------------------------------
'''
configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []
folders.append(Path(r"Z:\Droplets\20261001\Scan6_CFG"))  #full centrifuge
configs.append(IonDataAnalysisConfig(
    delay_center= Length(88-POSZEROSHIFT, Prefix.MILLI),
    center=Point(200,200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS,90),
    transform_parameter=0.85))
folders.append(Path(r"Z:\Droplets\20261001\Scan7_CFG")) 
configs.append(configs[0]) #Use the same config for both folders, but different data
folders_1 = folders
configs_1 = configs
label_1 = "3mJ 3GHz Centrifuge"
#--------------------------------------------------------------------------------------------------------------
configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []
folders.append(Path(r"Z:\Droplets\20261001\Scan8_CFG"))  #full centrifuge
configs.append(IonDataAnalysisConfig(
    delay_center= Length(88-POSZEROSHIFT, Prefix.MILLI),
    center=Point(200,200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS,90),
    transform_parameter=0.85))
folders.append(Path(r"Z:\Droplets\20261001\Scan9_CFG")) 
configs.append(configs[0]) #Use the same config for both folders, but different data
folders_2 = folders
configs_2 = configs
label_2 = "3.8mJ 3GHz Centrifuge"
#--------------------------------------------------------------------------------------------------------------
configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []
folders.append(Path(r"Z:\Droplets\20261001\Scan10_CFG"))  #full centrifuge
configs.append(IonDataAnalysisConfig(
    delay_center= Length(88-POSZEROSHIFT, Prefix.MILLI),
    center=Point(200,200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS,90),
    transform_parameter=0.85))
#folders.append(Path(r"Z:\Droplets\20261001\Scan9_CFG")) 
#configs.append(configs[0]) #Use the same config for both folders, but different data
folders_3 = folders
configs_3 = configs
label_3 = "1.7mJ 3GHz Centrifuge"
'''
#--------------------------------------------------------------------------------------------------------------
#Looking at field-free rotation with the 5 Ghz centrifuge
#--------------------------------------------------------------------------------------------------------------
'''
configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []
folders.append(Path(r"Z:\Droplets\20260929\Scan2_AVG")) 
configs.append(IonDataAnalysisConfig(
    delay_center= Length(80-POSZEROSHIFT, Prefix.MILLI),
    center=Point(200,200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS,90),
    transform_parameter=0.85))
folders_1 = folders
configs_1 = configs
label_1 = "3mJ 5Ghz Centrifuge"
#--------------------------------------------------------------------------------------------------------------
configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []
folders.append(Path(r"Z:\Droplets\20261001\Scan11_CFG"))  #full centrifuge
configs.append(IonDataAnalysisConfig(
    delay_center= Length(80-POSZEROSHIFT, Prefix.MILLI),
    center=Point(200,200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS,90),
    transform_parameter=0.85))
#folders.append(Path(r"Z:\Droplets\20261001\Scan11_CFG")) 
#configs.append(configs[0]) #Use the same config for both folders, but different data
folders_2 = folders
configs_2 = configs
label_2 = "1.7mJ 5Ghz Centrifuge"
#--------------------------------------------------------------------------------------------------------------
configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []
folders.append(Path(r"Z:\Droplets\20261001\Scan12_CFG"))  #full centrifuge
configs.append(IonDataAnalysisConfig(
    delay_center= Length(80-POSZEROSHIFT, Prefix.MILLI),
    center=Point(200,200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS,90),
    transform_parameter=0.85))
#folders.append(Path(r"Z:\Droplets\20261001\Scan11_CFG")) 
#configs.append(configs[0]) #Use the same config for both folders, but different data
folders_3 = folders
configs_3 = configs
label_3 = "0.3mJ 5Ghz Centrifuge"
'''
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#Comparing 5GHz to faster CFG
#--------------------------------------------------------------------------------------------------------------
'''configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []
folders.append(Path(r"Z:\Droplets\20261001\Scan12_CFG"))  #full centrifuge
configs.append(IonDataAnalysisConfig(
    delay_center= Length(80-POSZEROSHIFT, Prefix.MILLI),
    center=Point(200,200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS,90),
    transform_parameter=0.85))
#folders.append(Path(r"Z:\Droplets\20261001\Scan11_CFG")) 
#configs.append(configs[0]) #Use the same config for both folders, but different data
folders_1 = folders
configs_1 = configs
label_1 = "0.3mJ 5Ghz Centrifuge"
#--------------------------------------------------------------------------------------------------------------
configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []
folders.append(Path(r"Z:\Droplets\20261001\Scan13_CFG"))  #full centrifuge
configs.append(IonDataAnalysisConfig(
    delay_center= Length(80-POSZEROSHIFT, Prefix.MILLI),
    center=Point(200,200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS,90),
    transform_parameter=0.85))
#folders.append(Path(r"Z:\Droplets\20261001\Scan11_CFG")) 
#configs.append(configs[0]) #Use the same config for both folders, but different data
folders_2 = folders
configs_2 = configs
label_2 = "0.3mJ 2-7GHz ?"
#--------------------------------------------------------------------------------------------------------------
configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []
folders.append(Path(r"Z:\Droplets\20261001\Scan14_CFG"))  #full centrifuge
configs.append(IonDataAnalysisConfig(
    delay_center= Length(80-POSZEROSHIFT, Prefix.MILLI),
    center=Point(200,200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS,90),
    transform_parameter=0.85))
#folders.append(Path(r"Z:\Droplets\20261001\Scan11_CFG")) 
#configs.append(configs[0]) #Use the same config for both folders, but different data
folders_3 = folders
configs_3 = configs
label_3 = "0.3mJ 5-10GHz ?"
#--------------------------------------------------------------------------------------------------------------
configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []
folders.append(Path(r"Z:\Droplets\20261001\Scan15_CFG"))  #full centrifuge
configs.append(IonDataAnalysisConfig(
    delay_center= Length(75-POSZEROSHIFT, Prefix.MILLI),
    center=Point(200,200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS,90),
    transform_parameter=0.85))
#folders.append(Path(r"Z:\Droplets\20261001\Scan11_CFG")) 
#configs.append(configs[0]) #Use the same config for both folders, but different data
folders_4 = folders
configs_4 = configs
label_4 = "0.3mJ 1-8GHz ?"
#--------------------------------------------------------------------------------------------------------------
configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []
folders.append(Path(r"Z:\Droplets\20261001\Scan16_CFG"))  #full centrifuge
configs.append(IonDataAnalysisConfig(
    delay_center= Length(73-POSZEROSHIFT, Prefix.MILLI),
    center=Point(200,200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS,90),
    transform_parameter=0.85))
#folders.append(Path(r"Z:\Droplets\20261001\Scan11_CFG")) 
#configs.append(configs[0]) #Use the same config for both folders, but different data
folders_5 = folders
configs_5 = configs
label_5 = "0.3mJ 1.5-13GHz ?"
#--------------------------------------------------------------------------------------------------------------
configs: list[IonDataAnalysisConfig] = []
folders: list[Path] = []
folders.append(Path(r"Z:\Droplets\20261001\Scan17_CFG"))  #full centrifuge
configs.append(IonDataAnalysisConfig(
    delay_center= Length(66-POSZEROSHIFT, Prefix.MILLI),
    center=Point(200,200),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](MINRADIUS,90),
    transform_parameter=0.85))
#folders.append(Path(r"Z:\Droplets\20261001\Scan11_CFG")) 
#configs.append(configs[0]) #Use the same config for both folders, but different data
folders_6 = folders
configs_6 = configs
label_6 = "0.3mJ 1-17GHz ?"
'''
#--------------------------------------------------------------------------------------------------------------







#Update the matplotlib settings
plt.style.use(r"stylefiles\compare_c2t_spectrogram.mplstyle")

#Pipeline 
plottable_scan_1, plottable_spectrogram_1 = calculating(folders_1, configs_1)
if 'folders_2' in locals():
    plottable_scan_2, plottable_spectrogram_2 = calculating(folders_2, configs_2)
if 'folders_3' in locals():
    plottable_scan_3, plottable_spectrogram_3 = calculating(folders_3, configs_3)
if 'folders_4' in locals():
    plottable_scan_4, plottable_spectrogram_4 = calculating(folders_4, configs_4)
if 'folders_5' in locals():
    plottable_scan_5, plottable_spectrogram_5 = calculating(folders_5, configs_5)
if 'folders_6' in locals():
    plottable_scan_6, plottable_spectrogram_6 = calculating(folders_6, configs_6)

#PlotTitle = PlotTitle + "\n" + str(folders_1)  + "\n" + str(folders_2) + "\n" + str(folders_3) 
#Main figure
mainfig, (axs) = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(6.75, 4.2),
            sharex=True,             
            gridspec_kw={'hspace': 0.1,'wspace': 0.3}
        )

a = axs
plot_averaged_scan(a, plottable_scan_1, PlotColor.BLUE,ecolor=PlotColor.BLUE,marker='d', label = label_1,elinewidth=2)
if 'folders_2' in locals():
    plot_averaged_scan(a, plottable_scan_2, PlotColor.RED,ecolor=PlotColor.RED,marker='x', label = label_2,elinewidth=2)
if 'folders_3' in locals():
    plot_averaged_scan(a, plottable_scan_3, PlotColor.BLACK,ecolor=PlotColor.BLACK,marker='x', label = label_3,elinewidth=2)
if 'folders_4' in locals():
    plot_averaged_scan(a, plottable_scan_4, PlotColor.GREEN,ecolor=PlotColor.GREEN,marker='x', label = label_4,elinewidth=2)
if 'folders_5' in locals():
    plot_averaged_scan(a, plottable_scan_5, PlotColor.PURPLE,ecolor=PlotColor.PURPLE,marker='x', label = label_5,elinewidth=2)
if 'folders_6' in locals():
    plot_averaged_scan(a, plottable_scan_6, PlotColor.GRAY,ecolor=PlotColor.GRAY,marker='x', label = label_6,elinewidth=2)

a.grid()
a.set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
a.legend(loc="upper right")

a.set_title(PlotTitle,fontsize=MAJORTITLEFONTSIZE,color='black')

#Save scans
plottable_scan_1.to_csv(savedata_filename_1)

mainfig.savefig(fig_filename,format='png',dpi=300)
plt.show()
print('Done!')