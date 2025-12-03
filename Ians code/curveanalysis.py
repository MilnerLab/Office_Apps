#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last modified: 2025 November 18th 7pm

@author: ian
"""


import numpy as np
from scipy.signal import stft, detrend
from scipy.interpolate import CubicSpline


def find_min_distance(lst):
    sorted_lst = sorted(set(lst))
    return min(n2 - n1 for n1, n2 in zip(sorted_lst, sorted_lst[1:]))

def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def find_nearest(array, value):     
    array = np.asarray(array);     
    idx = (np.abs(array - value)).argmin();     
    return idx


def make_spectrogram(t_arr,sig_arr,windfract=2, remove_DC = True):
    
    fs = 1000/ np.mean(np.diff(t_arr)) #1000 converts from 1/ps to 1/ns

    if remove_DC == True:
        sig_arr = detrend(sig_arr, type='constant') #Remove constant component
        sig_arr = detrend(sig_arr, type='linear') #remove "tilt" (linear change in C2T over time)

    samplesperseg = int(len(t_arr))/windfract
    numoverlap = samplesperseg - 1
    f, t_s, Zxx = stft(sig_arr, fs=fs, nperseg=samplesperseg, noverlap=numoverlap,window='blackman')

    # -----
    SpecSig = (np.abs(Zxx))**2
    t_s = t_s*1000 + t_arr[0] #convert back to ps and get time zero back
    return f, t_s, SpecSig, Zxx

def constant_spacing(t_arr,sig_arr,type='linear',densityfactor=1):
    
    stepsize = find_min_distance(t_arr)
 
    t_new = np.linspace(min(t_arr), max(t_arr), int(densityfactor*(max(t_arr)-min(t_arr))/stepsize))
        
    if type == 'cubic':
        cub_func = CubicSpline(t_arr, sig_arr)
        sig_new = cub_func(t_new)
        
    else: 
        sig_new = np.interp(t_new, t_arr, sig_arr)
        
    return t_new, sig_new