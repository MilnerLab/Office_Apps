print('Code start!')
from pathlib import Path
from altair import FontWeight
import matplotlib as mpl

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import lmfit

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from _data_io.dat_saver import create_save_path_for_calc_ScanFile
from _domain.plotting import plot_GaussianFit
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan

from apps.stft_analysis.domain.models import AggregateSpectrogram

from base_core.lab_specifics.averaging.models import AveragedScansData
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time
print('Dependencies loaded.')

DROPLETRADIUSMIN = 60
STFTWINDOWSIZE = Time(180,Prefix.PICO)  
EARLIEST_DELAY_PS = -800
LATEST_DELAY_PS = 3500
POSZEROSHIFT = 0 #millimetres, applies to all scans calculated from raw ion histograms so is probably not useful here.


#FUNCTION TO GENERATE THE PLOTTABLE DATA
def calculating(folders: list[Path], configs: list[IonDataAnalysisConfig]) -> tuple[AveragedScansData, AggregateSpectrogram]:
    print('Starting: ',folders)
    
    scans_paths = DatFinder(folders).find_datafiles() #Change this if you want a specific path rather than the Droplets folder
    print('Data found!')
    raw_datas = load_ion_data(scans_paths)
    print('Data loaded!')
    calculated_scans = run_pipeline(raw_datas, configs)
    averagedScanData = average_scans(calculated_scans)
    print('Scans resampled!')
    
    return (averagedScanData)
#--------------------------------------------------------------------------------------------------

#Path to save figure in
fig_filedir = r"Z:\Droplets\plots" 
fig_filename = fig_filedir + r"\\fig5_longlasting_TEMP.png" #Name the file to save here

warnings: list[str] = []



#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#OCS DATA FIRST
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#OCS plots can have tracenames array (old from previous script...)
tracenames: list[str] = []
#Trace 2
configs_1: list[IonDataAnalysisConfig] = []
folders_1: list[Path] = [Path(r"20260211\Scan1")]
tracenames.append("3mJ usCFG (polarization averaged)")
#folders_1.append(Path(r"20260211\Scan1"))  

configs_1.append(IonDataAnalysisConfig(
    delay_center= Length(92.654-POSZEROSHIFT, Prefix.MILLI),
    center=Point(174, 206),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](50, 120),
    transform_parameter= 0.78))

#Trace 2
#folders_2: list[Path] = [Path(r"20260211\Scan2")]
#tracenames.append("1.5mJ Circular Polarization")

#Trace 3
#folders_3: list[Path] = [Path(r"20260211\Scan4")]#scan 3 also but not needed.
#tracenames.append("1.5mJ // Linear Polarization")
#Trace 4
#folders_4: list[Path] = [Path(r"20260211\Scan5")]
#tracenames.append("1.5mJ // Linear Polarization \n with droplet beam blocked.")

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#CS2 DATA
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#CS2 IN JET
cs2_jet_folders: list[IonDataAnalysisConfig] = []
cs2_jet_configs: list[Path] = []

# folders_1.append(Path(r"20251128\Scan4"))  #20251128\Scan4 set has a different maximum from 20260112\4+5 that looks weird mid-scan
# configs_1.append(IonDataAnalysisConfig(
#     delay_center= Length(91-POSZEROSHIFT, Prefix.MILLI),
#     center=Point(199, 203),
#     angle= Angle(12, AngleUnit.DEG),
#     analysis_zone= Range[int](25, 120),
#     transform_parameter=0.9))


warnings.append("CS2 jet data has been shifted by 15mm to better overlap droplet data for in-field peak")
cs2_jet_folders.append(Path(r"20260112\JetScan4")) 
cs2_jet_configs.append(IonDataAnalysisConfig(
    delay_center= Length(98.054-POSZEROSHIFT-15, Prefix.MILLI),
    center=Point(214, 191),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](25, 120),
    transform_parameter=0.75))

cs2_jet_folders.append(Path(r"20260112\JetScan5")) 
cs2_jet_configs.append(IonDataAnalysisConfig(
    delay_center= Length(98.054-POSZEROSHIFT-15, Prefix.MILLI),
    center=Point(214, 191),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](25, 120), 
    transform_parameter=0.75))


#--------------------------------------------------------------------------------------------------------------
#CS2 IN DROPLETS WITH CFG
#Can look at 20260121\Scan2_ScanFiles and 20260120 scans 1 and 2, also 20260119\Scan2_ScanFiles for direct 1-arm comparison

cs2_droplets_folders: list[IonDataAnalysisConfig] = []
cs2_droplets_configs: list[Path] = []

cs2_droplets_folders.append(Path(r"20260120\Scan1")) 
cs2_droplets_configs.append(IonDataAnalysisConfig(
    delay_center= Length(98.054-POSZEROSHIFT, Prefix.MILLI),
    center=Point(228, 193),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.75))


cs2_droplets_folders.append(Path(r"20260120\Scan2")) 
cs2_droplets_configs.append(IonDataAnalysisConfig(
    delay_center= Length(98.054-POSZEROSHIFT, Prefix.MILLI),
    center=Point(228, 193),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.75))

cs2_droplets_folders.append(Path(r"20260119\Scan2_CFG")) 
cs2_droplets_configs.append(IonDataAnalysisConfig(
    delay_center= Length(98.054-POSZEROSHIFT, Prefix.MILLI),
    center=Point(228, 193),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.75))

#--------------------------------------------------------------------------------------------------------------
#CS2 IN DROPLETS WITH SINGLE ARM

cs2_droplets_1arm_folders: list[IonDataAnalysisConfig] = []
cs2_droplets_1arm_configs: list[Path] = []

# cs2_droplets_1arm_folders.append(Path(r"20260114\Scan1_HorizontalDA")) 
# cs2_droplets_1arm_configs.append(IonDataAnalysisConfig(
#     delay_center= Length( 98.054-POSZEROSHIFT, Prefix.MILLI),
#     center=Point(188, 200),
#     angle= Angle(12, AngleUnit.DEG),
#     analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
#     transform_parameter=0.73))

cs2_droplets_1arm_folders.append(Path(r"20260119\Scan1_Horizontal")) 
cs2_droplets_1arm_configs.append(IonDataAnalysisConfig(
    delay_center= Length(98.054-POSZEROSHIFT, Prefix.MILLI),
    center=Point(228, 193),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
    transform_parameter=0.75))

# cs2_droplets_1arm_folders.append(Path(r"20260203\Scan1_horizontal")) 
# cs2_droplets_1arm_configs.append(IonDataAnalysisConfig(
#     delay_center= Length(89.65-POSZEROSHIFT, Prefix.MILLI),
#     center=Point(181, 205),
#     angle= Angle(12, AngleUnit.DEG),
#     analysis_zone= Range[int](DROPLETRADIUSMIN, 120),
#     transform_parameter=0.73))
#--------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------
# Import 20250615 OCS cfCFG data from CSVs
OCS_cfCFG_resonant = r"Z:\Droplets\exportdata\cfCFG_2025_OCS_processed\FromC2T\OCS_Resonant.csv"
OCS_cfCFG_nonresonant = r"Z:\Droplets\exportdata\cfCFG_2025_OCS_processed\FromC2T\OCS_Out.csv"

OCS_cfCFG_resonant = pd.read_csv(OCS_cfCFG_resonant,header = None,names = ['DELAY','C2T','ERR'])
OCS_cfCFG_nonresonant = pd.read_csv(OCS_cfCFG_nonresonant,header = None,names = ['DELAY','C2T','ERR'])


##--------------------------------------------------------------------------------------------------

#Update the matplotlib settings
plt.style.use(r"stylefiles\compare_c2t_spectrogram.mplstyle")
##--------------------------------------------------------------------------------------------------

#Pipeline for the usCFG data
OCS_usCFG = calculating(folders_1, configs_1) #Same config used for each 

CS2_jet = calculating(cs2_jet_folders, cs2_jet_configs)

cs2_droplets_CFG = calculating(cs2_droplets_folders, cs2_droplets_configs)
cs2_droplets_1arm = calculating(cs2_droplets_1arm_folders, cs2_droplets_1arm_configs)
print('Done importing data.')
##--------------------------------------------------------------------------------------------------

DelayArray = OCS_cfCFG_resonant.DELAY
#Try to calculate the decay
start_ind = 16 #hardcoded index corresponding to approximately +500ps (field-free)
end_ind = len(np.asarray(OCS_cfCFG_resonant.DELAY))
mod_t = np.asarray(DelayArray[start_ind:end_ind])
mod_amp = np.asarray(OCS_cfCFG_resonant.C2T[start_ind:end_ind])

#Single decay
decmodel = lmfit.Model(lambda tvals,ft_tdecay,ft_A: 0.5 + ft_A*np.exp(-tvals/ft_tdecay))
decayfit = decmodel.fit(mod_amp, tvals=mod_t, ft_tdecay=4000, ft_A=0.04)

#%%
fitparams = decayfit.params.valuesdict()

Lifetime = fitparams['ft_tdecay']
Lifetime_err = decayfit.params['ft_tdecay'].stderr

print('Tau = ' + str(int(Lifetime)) + ' +/- '+ str(int(Lifetime_err)) + '\n')
t_interp = np.linspace(mod_t[0],mod_t[-1],100*len(mod_t))
c2t_interp = decayfit.model.func(t_interp,**decayfit.best_values)

print('Ready to plot...')
##--------------------------------------------------------------------------------------------------

#%%
#Main figure
mainfig, (axs) = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(6.75/2, 4.5),
            sharex=True,             
            gridspec_kw={'hspace': 0,'wspace': 0.3}
        )

#First subplot is OCS in droplets, usCFG vs resonant cfCFG vs nonresonant cfCFG
a = axs[0]
plot_averaged_scan(a, OCS_usCFG, marker='x',color=PlotColor.BLUE,ecolor=PlotColor.RED, label = None)
a.errorbar(OCS_cfCFG_resonant.DELAY,OCS_cfCFG_resonant.C2T,yerr=OCS_cfCFG_resonant.ERR,color='k',ecolor='r',marker='d',elinewidth=1)
a.errorbar(OCS_cfCFG_nonresonant.DELAY,OCS_cfCFG_nonresonant.C2T,yerr=OCS_cfCFG_nonresonant.ERR,color='grey',ecolor='grey',marker='*',elinewidth=1)
a.plot(t_interp,c2t_interp,'c-',linewidth=3) #fit to field-free decay
a.set_ylabel(None)
a.grid()

#Second subplot is CS2 in the jet with the averaged usCFG
a = axs[1]
plot_averaged_scan(a,CS2_jet,marker='d',color=PlotColor.BLACK,ecolor=PlotColor.RED,label=None)
a.set_ylabel(None)
a.grid()


#Third subplot is CS2 in droplets with averaged usCFG and then also 1 arm
a = axs[2]
plot_averaged_scan(a,cs2_droplets_CFG,marker='d',color=PlotColor.BLUE,ecolor=PlotColor.RED,label=None)
plot_averaged_scan(a,cs2_droplets_1arm,marker='d',color=PlotColor.GRAY,ecolor=PlotColor.GRAY,label=None)

a.set_ylabel(None)
a.grid()



a.set_xlim([EARLIEST_DELAY_PS,LATEST_DELAY_PS])
plt.xlabel('Probe Delay (ps)')
mainfig.supylabel('$\langle \cos^2 \\theta_{\mathrm{2D}} \\rangle$\n',horizontalalignment = 'right')


mainfig.savefig(fig_filename,format='png',dpi=300)
print("\n\nWARNINGS:\n",warnings)
plt.show()