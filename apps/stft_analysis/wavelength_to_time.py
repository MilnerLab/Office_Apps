from matplotlib import pyplot as plt
import numpy as np 

from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.plotting import plot_Spectrogram
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from base_core.lab_specifics.base_models import C2TScanData, Measurement, ScanDataBase
from base_core.math.models import Range
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Time

#At GA_14_5mm
data = np.loadtxt("Z:\\Droplets\\20260904\\Spectral_Info\\Scan4_CFG_GA_14p5mm.csv",delimiter=',',skiprows=1)

c = 3e8 #in m/s
tau = 320e-12 #in s
tau_0 = 0.44*((802e-9+4.5e-9)*(802e-9-4.5e-9))/c/9e-9 #in s FWHM
beta_0 = (1/tau**2)/3.1415*((tau/tau_0)**2-1)**0.5 #in ps^-2
lambda_0 = 802e-9 #in m
y = data[:,1]
t = c/beta_0*(1/(1e-9*data[:,0]) - 1/lambda_0)*1e12 #in ps

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(8,5),gridspec_kw={"hspace":0})

sort_arg = np.argsort(np.array(t))


ax1.plot(t,y,'b-')
WINDOWSIZE = Time(180,Prefix.PICO)
n = len(y)
measured_values = [Measurement(y[i],0) for i in sort_arg]
delays = [Time(t[i],Prefix.PICO) for i in sort_arg]
scan_data = ScanDataBase(delays=delays,measured_values=measured_values,run_id=0)
stft_config = StftAnalysisConfig(scan_data, WINDOWSIZE)
resampled_scans = resample_scans(scan_data,stft_config.axis)
spectrogram = StftAnalysis(resampled_scans,stft_config).calculate_averaged_spectrogram()
        
plot_Spectrogram(ax2,spectrogram,v_range=Range(0,1),shading='auto')

plt.show()
