from pathlib import Path
import numpy as np
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
from base_core.lab_specifics.base_models import C2TScanData, IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time

#Update the matplotlib settings
plt.style.use(r"stylefiles\compare_c2t_spectrogram.mplstyle")

DROPLETRADIUSMIN = 60

STFTWINDOWSIZE = Time(180,Prefix.PICO)  
POSZEROSHIFT = 5 #millimetres :)

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
#Data and calculation parameter

#Loading droplet data. GA=26mm, DA = 15.9mm
configs_droplets: list[IonDataAnalysisConfig] = []
folders_droplets: list[Path] = []

folders_droplets.append(Path(r"20251212\Scan4")) #Combination of 20251212 and 20251213. 
configs_droplets.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.79))

folders_droplets.append(Path(r"20251213\Scan1")) #Combination of 20251212 and 20251213. 
configs_droplets.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.79))

folders_droplets.append(Path(r"20251213\Scan2")) #Combination of 20251212 and 20251213. 
configs_droplets.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.79))


folders_droplets.append(Path(r"20251213\Scan3")) #Combination of 20251212 and 20251213. 
configs_droplets.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(194, 204),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter= 0.79)) 

#Loading jet data
folders_jet: list[Path]=[]
configs_jet: list[IonDataAnalysisConfig]=[]

folders_jet.append(Path(r"20251210\JSS3"))  #20251210 JSS3 is dense throughout the centrifuge
configs_jet.append(IonDataAnalysisConfig(
    delay_center= Length(90.55-POSZEROSHIFT, Prefix.MILLI),
    center=Point(195, 197),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](20, 120),
    transform_parameter= 0.79))

folders_jet.append(Path(r"20251210\JSS4"))  #20251210 JSS4 is dense before the centrifuge
configs_jet.append(configs_jet[0]) #same config  


plottable_scan_drop,plottable_spec_drop = calculating(folders=folders_droplets,configs=configs_droplets)
plottable_scan_jet,plottable_spec_jet = calculating(folders_jet,configs_jet)

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

#Folder to save figures
savefig_folder = r"Z:\Droplets\plots\\"

#-----------------------------------------------------------------------------------
#Figure 1 - CS2 jet and droplets 2025 data BREAKING THROUGH THE WALLLLL

savefig1_filename = savefig_folder + r"cs2-breakingwall_TEMP.png"
#fig1,(ax1,ax2) = plt.subplots(2,1,sharex=True,gridspec_kw={'hspace':0})
#Main figure
fig1, (ax1,ax2) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(6.75/2, 3),
            sharex=True,             
            gridspec_kw={'hspace': 0,'wspace': 0.3}
        )


#Truncating so that the datasets have same delay range
times = np.array([t.value(Prefix.PICO) for t in plottable_scan_drop.delays])
mask = (times > -300)
inds = np.where(mask)
start = inds[0][0]
truncated_scan_drop = plottable_scan_drop.cut(start=start)

times = np.array([t.value(Prefix.PICO) for t in plottable_scan_jet.delays])
mask = (times < 300)
inds = np.where(mask)
end = inds[-1][-1]
plottable_scan_jet.cut(start=0,end=end)

#add a markersize option or a separate rcparams for plot bot and other scripts
plot_averaged_scan(ax1,plottable_scan_jet,PlotColor.BLUE,ecolor=PlotColor.RED,marker='d',label = None) 
plot_averaged_scan(ax2,plottable_scan_drop,PlotColor.BLUE,ecolor=PlotColor.RED,marker='d',label = None)

ax1.set_xlim([-250,220])

#Enable the grids
ax1.grid(True) 
ax2.grid(True)


#Vertical line on second plot
x = [-85,-85] #approximate position of dashed line
y = list(ax2.get_ylim()) 
ax2.plot(x,y,'k--',linewidth=1)


#(a) (b) placement etc
textx = 0.055
texty = 0.9
ax1.text(textx, texty, '(a)',color='k', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
ax2.text(textx, texty, '(b)',color='k', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

ax1.set_ylabel(r'$\langle \cos^2\theta_{2D}\rangle$')
ax2.set_ylabel(r'$\langle \cos^2\theta_{2D}\rangle$')

fig1.savefig(savefig1_filename,format='png',dpi=300) 
plt.show()
print('Done!')