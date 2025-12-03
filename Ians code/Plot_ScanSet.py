    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last modified: 2025 November 18th 7pm


@author: ian
"""


#If True, program works with Volumes/ValeryShare/Droplets/.
#It plots all ScanFIles, but if there's more than 1 it doesn't include the last (incomplete) one.
UseMostRecent = True



print('Starting....')

#Directory with curveanalysis function (two paths for Ian's windows vs mac computers)
import pandas as pd
import pathlib
import numpy as np
import curveanalysis as curan
import matplotlib.pyplot as plt

#Folder if not using most recent data
scanfolder = r'Z:\Droplets\20251119\JetScanSet5'

#Note is above the plot in the title, not included if UseMostRecent=True
note = '\nCS$_2$ in Helium Droplets' +'\n' + '(GA=26mm, DA=15.9mm)'



#Plot parameters, some defunct :)))
class Params:
    xleft = -220
    xright = 220
    t_offset = 0
    windfract = 3 
    cmin = 1E-5
    cmax = 5E-5
    fmax = 120 #max frequency in spectrogram

#input parameters above, actual operations are below.
#%%
if UseMostRecent == True:
    scanfolder = r'/Volumes/ValeryShare/Droplets/' 
    note = ''
    
FileList = list(pathlib.Path(scanfolder).glob('*ScanFile.dat')) 
FileList = sorted(FileList) #sort them by date

if UseMostRecent == True:
    if len(FileList) > 1:
        FileList = FileList[0:-1] #all but the last one, since we assume it's incomplete.
        
#Some older data is missing the timestamp column.
namearr = ['timestamp','delay','c2t','c2t_err','ipf','c2t-BG']
#namearr = ['delay','c2t','c2t_err','ipf','c2t-BG']
SpecSigSum = None

for filename in FileList:
    print('Processing ' + str(filename))
    Data = pd.read_csv(filename, header=None,names=namearr ,sep='\t', lineterminator='\n',dtype=float)
  
    Data.delay = Data.delay + Params.t_offset #If you want to fix time zero
    
    #Interpolation for constant step size
    t_int, c2t_arr = curan.constant_spacing(Data.delay,Data.c2t,type='cubic',densityfactor=1)

    
    #FFT stuff
    f, t_s, SpecSig, Zxx = curan.make_spectrogram(t_int, c2t_arr,Params.windfract,remove_DC=True)
    if filename == FileList[0]:
        SpecSigSum = SpecSig
    else:
        SpecSigSum = SpecSigSum + SpecSig
        
#SpecSigSum = SpecSigSum/len(FileList) #Normalize but keep real PSD
if SpecSigSum is not None:
    SpecSigSum = SpecSigSum/np.amax(SpecSigSum) #Normalize to 1

Data = pd.read_csv(FileList[-1], header=None,names=namearr ,sep='\t', lineterminator='\n',dtype=float)
Data.delay = Data.delay + Params.t_offset #If you want to fix time zero
#Interpolation for constant step size
t_int, c2t_arr = curan.constant_spacing(Data.delay,Data.c2t,type='cubic',densityfactor=1)
f, t_s, SpecSig, Zxx = curan.make_spectrogram(t_int, c2t_arr,Params.windfract,remove_DC=True)

#%%
#PLOTTING

fftfig, (axs) = plt.subplots(2)
#First plot is C2T data
a = axs[0]
#First the real data with errorbars
a.errorbar(Data.delay,Data.c2t,yerr=Data.c2t_err,color='k',ecolor='grey',elinewidth  = 1, marker = 'd',markersize = 4, linestyle = None,linewidth=0)
#Then the interpolated data
a.plot(t_int,c2t_arr,color='b',marker='x',markersize=2,linewidth=0.5,linestyle='--')

a.set_xlim(Params.xleft,Params.xright)
a.set_ylabel('$\langle \cos^2\\theta_{\mathrm{2D}} \\rangle$')
a.set_title((scanfolder.split('ValeryShare/Droplets/')[-1] + note))
a.grid('both',color='grey',linewidth=0.3)

#Second plot is spectrogram
a = axs[1]
#pcm = a.pcolormesh(t_s, f, SpecSigSum, shading="gouraud",cmap="viridis", vmin = Params.cmin, vmax = Params.cmax)
pcm = a.pcolormesh(t_s, f, SpecSigSum, shading="gouraud",cmap="viridis")
#fftfig.colorbar(pcm, ax=a)

#Plot the nyquist frequency of the raw data (before interpolation)
a.plot(Data.delay[0:-1],500/np.diff(Data.delay),'r:')

#cbartop = fftfig.colorbar(pcm, ax=axs[0]) #add colourbar to top figure just to align the graphs


a.set_xlim(Params.xleft,Params.xright)
a.set_ylim(0, Params.fmax)
a.set_xlabel("Time (ps)")
a.set_ylabel("Oscillation \n Frequency (GHz)")
a.grid('both',color='grey',linewidth=0.3)

plt.show()

#%%

#Figure that gets overwritten each time you run it
#fftfig.figure.savefig(r'/Users/ian/Library/CloudStorage/Dropbox/Postdoc_DB/PD_Code/Figure_Dumping/FFT_Stack.png', format='png', dpi=150,bbox_inches="tight")

print('Final Delay = ' + str(Data.delay.iloc[-1]) + ' ps. Ions/frame = ' + str(np.mean(Data.ipf)))
print('Done plotting ' + str(len(FileList)) + ' scans')