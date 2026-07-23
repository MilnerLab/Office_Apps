print('Code start!')
from pathlib import Path
from altair import FontWeight
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from _data_io.dat_saver import create_save_path_for_calc_ScanFile
from _domain.plotting import plot_GaussianFit
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from apps.single_scan.domain.plotting import plot_single_scan
from base_core.lab_specifics.averaging.models import AveragedScansData
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length

DROPLETRADIUSMIN = 65

EARLIEST_DELAY_PS = -200
LATEST_DELAY_PS = -EARLIEST_DELAY_PS
POSZEROSHIFT = 0 #millimetres :)

MAJORTITLEFONTSIZE = 16

#FUNCTION TO GENERATE THE PLOTTABLE DATA
def calculating(folders: list[Path], configs: list[IonDataAnalysisConfig]) -> AveragedScansData:


    scans_paths = DatFinder(folders).find_datafiles() #Change this if you want a specific path rather than the Droplets folder
    raw_datas = load_ion_data(scans_paths)
    calculated_scans = run_pipeline(raw_datas, configs)
    averagedScanData = average_scans(calculated_scans)

    return averagedScanData
#--------------------------------------------------------------------------------------------------

#Path to save figure in
fig_filedir = r"Z:\Droplets\plots" 
fig_filename = fig_filedir + r"\CS2_directionality_droplets_TEMP.pdf" #Name the file to save here

#Path to save processed data in
savedata_filedir = r"Z:\Droplets\exportdata" 
savedata_filename_1 = savedata_filedir + r"\CS2_accelerating_droplets.csv" #Name the file to save here
savedata_filename_2 = savedata_filedir + r"\CS2_decelerating_droplets.csv" #Name the file to save here


#Plot on top


PlotTitle = r"CS$_2$ in 30 bar / 16 K droplets"


#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
# Accelerating
configs_1: list[IonDataAnalysisConfig] = []
folders_1: list[Path] = []

folders_1.append(Path(r"20260430\Scan3")) #GA=0, DA=15.5, ACCELLERATING
configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
    center=Point(205, 194),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.78))

folders_1.append(Path(r"20260501\Scan1")) #GA=0, DA=15.5, ACCELLERATING
configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
    center=Point(205, 194),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.78))
'''
#Better without 0507
#20260507\Scan1 looks good on its own, but busies up the graph if added to the others
# folders_1.append(Path(r"20260507\Scan1"))  #GA=0, DA=15.5, ACCELLERATING
# configs_1.append(IonDataAnalysisConfig(
#     delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
#     center=Point(204, 196),
#     angle= Angle(12, AngleUnit.DEG),
#     analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
#     transform_parameter=0.78))
'''
# folders_1.append(Path(r"20260507\Scan1")) #GA=0, DA=15.5, ACCELLERATING, TO COMPARE WITH 2026/05/13 AND JET FROM 05/11 and 05/13
# configs_1.append(IonDataAnalysisConfig(
#     delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
#     center=Point(204, 196),
#     angle= Angle(12, AngleUnit.DEG),
#     analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
#     transform_parameter=0.78))
    

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
# DECELERATING

configs_2: list[IonDataAnalysisConfig] = []
folders_2: list[Path] = []

'''
#Better without 0427 and 0428
# 20260427\Scan3 seems worse than the others 
# folders_2.append(Path(r"20260427\Scan3"))  #GA=0, DA=16.42, DECELLERATING
# configs_2.append(IonDataAnalysisConfig(
#     delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
#     center=Point(205, 194),
#     angle= Angle(12, AngleUnit.DEG),
#     analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
#     transform_parameter=0.78))


# folders_2.append(Path(r"20260428\Scan1")) #GA=0, DA=16.42, DECELLERATING
# configs_2.append(IonDataAnalysisConfig(
#     delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
#     center=Point(205, 194),
#     angle= Angle(12, AngleUnit.DEG),
#     analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
#     transform_parameter=0.78))
'''

folders_2.append(Path(r"20260429\Scan1")) #GA=0, DA=16.42, DECELLERATING
configs_2.append(IonDataAnalysisConfig(
    delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
    center=Point(205, 194),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.78))


# folders_2.append(Path(r"20260513\Scan1_without_overdoped_scan1")) #GA=0, DA=16.45, DECELERATING, TO COMPARE WITH 2026/05/07 AND JET FROM 05/11 and 05/13
# configs_2.append(IonDataAnalysisConfig(
#     delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
#     center=Point(197, 195),
#     angle= Angle(12, AngleUnit.DEG),
#     analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
#     transform_parameter=0.77))

# folders_2.append(Path(r"20260513\Scan2")) #GA=0, DA=16.45, DECELERATING, TO COMPARE WITH 2026/05/07 AND JET FROM 05/11 and 05/13
# configs_2.append(IonDataAnalysisConfig(
#     delay_center= Length(93.3-POSZEROSHIFT, Prefix.MILLI),
#     center=Point(197, 195),
#     angle= Angle(12, AngleUnit.DEG),
#     analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
#     transform_parameter=0.77))


#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
# SIMULATION
simulation_filename1 = r"/mnt/data/git/Milner_Lab/Latex/droplet_theory_paper/theory_calc/CS2_accelerating_droplets_CS2_cos2theta2D_vs_t_model_only.csv"
simulation_filename2 = r"/mnt/data/git/Milner_Lab/Latex/droplet_theory_paper/theory_calc/CS2_decelerating_droplets_CS2_cos2theta2D_vs_t_model_only.csv"

forward = pd.read_csv(simulation_filename1,names = ['time','signal_raw','signal_scaled'])
reverse = pd.read_csv(simulation_filename2,names = ['time','signal_raw','signal_scaled'])


#--------------------------------------------------------------------------------------------------
#Update the matplotlib settings
plt.style.use(r"stylefiles/compare_c2t_spectrogram.mplstyle")

#Pipeline
plottable_scan_1 = calculating(folders_1, configs_1)
plottable_scan_2 = calculating(folders_2, configs_2)

#(a) (b) placement etc
#Labels
textx = 0.1
texty = 0.9



#Main figure
mainfig, (axs) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(3.375, 3.5),
            sharex=True,
            gridspec_kw={'hspace': 0.1}
        )

#Plot first experiment
a = axs[0]
plot_averaged_scan(a, plottable_scan_1, PlotColor.BLUE,ecolor=PlotColor.RED,marker='d', label = None,elinewidth=0)
a.plot(forward.time,forward.signal_scaled,color=PlotColor.RED) #plot theory simulation
a.text(textx, texty, '($\\textbf{a}$)',color='k', horizontalalignment='center', verticalalignment='center', transform=a.transAxes)

a.grid()
a.set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
a.set_xlabel(None)

#Plot second experiment
a = axs[1]
plot_averaged_scan(a, plottable_scan_2, PlotColor.BLUE,ecolor=PlotColor.RED,marker='d',label=None,elinewidth=0)
a.plot(forward.time,reverse.signal_scaled,color=PlotColor.RED) #plot theory simulation
a.text(textx, texty, '($\\textbf{b}$)',color='k', horizontalalignment='center', verticalalignment='center', transform=a.transAxes)

a.grid()

#mainfig.suptitle(PlotTitle,fontsize=MAJORTITLEFONTSIZE,color='black')

#Save scans
plottable_scan_1.to_csv(savedata_filename_1)
plottable_scan_2.to_csv(savedata_filename_2)

mainfig.savefig(fig_filename,format='pdf',dpi=300)
plt.show()
print('Done!')