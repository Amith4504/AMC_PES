#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 22:23:24 2019

@author: amith
"""


import os,random
import numpy as np
from scipy.signal import hilbert,chirp
import matplotlib.pyplot as plt
import pickle
from scipy import signal

Xd = pickle.load(open("RML2016.10a_dict.pkl",'rb'),encoding='latin')
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)


# implementation of procedure for feature extraction
# 128 sample points exist of '8PSK'
# K vector represents 128 samples of 8PSK modulation type , FFT .
K=X[0]
K.shape
Kreal = K[0,:]
Kimag = K[1,:]
# IQ data information from national instruments support
# plotting IQ data for all 11 modulation techniques at their highest SNRs ie +17-20SNR
t=np.arange(128)
plt.plot(t,Kreal,t,Kimag)

for i in range(11):
    K=X[16000+(i*(20000))]
    Kreal=K[0,:]
    Kimag=K[1,:]
    plt.figure(i)
    plt.title(mods[i])
    plt.scatter(Kreal,Kimag,s=10)
    plt.savefig(mods[i],dpi=400)
    
for i in range(11):
    K=X[12000+(i*(20000))]
    Kreal=K[0,:]
    Kimag=K[1,:]
    plt.figure(i)
    plt.title(mods[i]+' IQ data')
    plt.plot(t,Kreal,t,Kimag)
    plt.savefig(mods[i]+'_IQdata',dpi=400)


# gamma max feauture extraction
#EDIT FROM HERE
# to initialize the lists to 1
Zabs=[1 for _ in range(11)]
avgZabs=[1 for _ in range(11)]
normZabs=[1 for _ in range(11)]
cen_normZabs=[1 for _ in range(11)]
Zfft=[1 for _ in range(11)]
abs_Zfft=[1 for _ in range(11)]
gamma_max=[1 for _ in range(11)]

for i in range(11):
    K=X[16000+(i*(20000))]
    Kreal=K[0,:]
    Kimag=K[1,:]
    Zabs[i] = np.sqrt(np.square(Kreal)+np.square(Kimag))
    #plt.plot(t,Zabs[i])
    #plt.savefig('Absdata 8PSK.png',dpi=400)

# feature one gamma max
# maximum value of the power spectral density of the normalised and centered 
# instantaneous amplitude
    avgZabs[i] = np.average(Zabs[i])
    normZabs[i] = Zabs[i]/avgZabs[i]
    #plt.plot(t,normZabs)

    cen_normZabs[i] = normZabs[i] -1
    #plt.plot(t,cen_normZabs)

# taking discrete fourier transform 
    Zfft[i] = np.fft.fft(cen_normZabs[i],n=128,norm=None)

    abs_Zfft[i] = np.absolute(Zfft[i])

    gamma_max[i] = np.max(np.square(abs_Zfft[i]))/128


l=np.arange(11)
plt.plot(l,gamma_max)
# end of feature one


#Feature 2
# centered and non-linear component of instantaneous phase
# calculate instantaneous phase over all 128 samples 
phase=[1 for _ in range(11)]
term1=[1 for _ in range(11)]
term2=[1 for _ in range(11)]
sigma_ap=[1 for _ in range(11)]
for i in range(11):
    K=X[16000+(i*(20000))]
    Kreal=K[0,:]
    Kimag=K[1,:]
    phase[i]=np.arctan(Kimag/Kreal)
    term1[i]=(np.sum(np.square(phase[i])))/128
    term2[i]=np.square(((np.sum(np.abs(phase[i])))/128))
    sigma_ap[i]=np.sqrt(term2[i]+term1[i])




    