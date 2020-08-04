# -*- coding: utf-8 -*-
"""
Created on Thu Jul  10 12:00:01 2020

@author: saeidi
"""
#####################################################################
####                      *    FUNCTION CODES    *
#####################################################################   

#################################### Read and oversampling Function
import numpy as np
import wfdb
from scipy import signal
def read_resample_ECG(signalID, ECG_resamplerate):
    N_db = None
    ECG_samplerate = 360    
    record_name = '../mitdb/'+str(signalID)
    ecg_db, fields = wfdb.rdsamp(record_name, sampto=N_db)
    ecg = ecg_db[:,0]
    ecg2 = ecg_db[:,1]    
    n_resample = int( ECG_resamplerate / ECG_samplerate * len(ecg))
    ecg_resample = signal.resample(ecg, n_resample)
    ecg_resample2 = signal.resample(ecg2, n_resample)    
    return (ecg_resample, ecg_resample2)

#################################### Noise Functions
import colorednoise as cn
def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

def input_noise(samplerate, oversampling_factor):
    n_sample = 650000 * oversampling_factor * 4
    max_freq =150 
    min_freq = 0    
    V_noise_BL = 0.095e-6     
    noise_BL = band_limited_noise(min_freq, max_freq, n_sample, samplerate)
    noise_BL = noise_BL * 0.7 * np.sqrt(n_sample*samplerate) * V_noise_BL
    beta = 1 # the exponent
    f_flicker = 1    
    pink_noise = cn.powerlaw_psd_gaussian(beta, n_sample, 0) 
    pink_noise = pink_noise * 3.85 * V_noise_BL * np.sqrt(f_flicker) 
    return (noise_BL, pink_noise)

#################################### IA Distortion function
from scipy.interpolate import interp1d
def IA_dist():          
    X, X0 , Y = [], [], []
    for line in open('../AFE_data/IA_dist.dat', 'r'):
      values = [float(s) for s in line.split()]
      if round(values[2]*1e6)/1e6 != 0:
        X.append(values[0])
        X0.append(values[1])
        Y.append(values[2])
#   # to extend interpolation rage  
    x_min_extend = -0.1
    x_max_extend = 0.1
    n_extend = 20
    Xnew=np.append(np.linspace(x_min_extend,min(X)-1e-4,n_extend), X)
    Ynew=np.append(np.ones(n_extend)* min(Y), Y)    
    Xnew=np.append(Xnew, np.linspace(max(X)+1e-4,x_max_extend,n_extend))
    Ynew=np.append(Ynew, np.ones(n_extend) * max(Y))    
    IA_Gain = interp1d(Xnew, Ynew, kind = 'cubic')        
    return IA_Gain

#################################### VCO Transfer Function  
def VCO_dist():            
    X, Y , Y0 = [], [], []
    for line in open('../AFE_data/VCO_TF.dat', 'r'):
      values = [float(s) for s in line.split()]
      X.append(values[0])
      Y.append(values[1])
      Y0.append(values[2])
    # to extend interpolation rage     
    x_min_extend = -4
    x_max_extend = 4
    n_extend = 20
    Xnew=np.append(np.linspace(x_min_extend,min(X)-1e-4,n_extend), X)
    Ynew=np.append(np.ones(n_extend)* min(Y), Y)    
    Xnew=np.append(Xnew, np.linspace(max(X)+1e-4,x_max_extend,n_extend))
    Ynew=np.append(Ynew, np.ones(n_extend) * max(Y))      
    VCO_TF = interp1d(Xnew, Ynew, kind = 'cubic')
    return VCO_TF

#################################### Counter
def Counter_ADC(freq_VCO, dt, CLK_freq):  
    CLK_period = 1/CLK_freq
    N_dt = int(CLK_period / dt)      
    i = 0
    SUM_phi = 0
    Counter = np.array([])
    Counting_per_clock = 0
    res_counter = 0   
    for freq in freq_VCO:
        SUM_phi = 2 * np.pi * freq * dt + SUM_phi
        i += 1
        if (i % N_dt == 0):
            SUM_phi += res_counter
            Counting_per_clock = int((SUM_phi + res_counter) // (2 * np.pi))
            Counter = np.append(Counter, Counting_per_clock) 
            res_counter = ((SUM_phi + res_counter) % (2 * np.pi))
            SUM_phi = 0
    return Counter

#################################### AFE MODEL
def AFE_model(signalID, ECG_resamplerate, noise_BL, pink_noise, IA_TF, VCO_TF, CLK_freq, t_stop):   
    (ecg_resample, ecg_resample2) = read_resample_ECG(signalID, ECG_resamplerate)     
    ###### To reduce the simulation time        
    try:
        if t_stop is None: # The variable
            print('Stop time is not defined; therefore, the whole input data will be used!')
        else:
            print ("Stop time is defined for the input signal and has a value of", t_stop ,"Sec")  
            num_point = t_stop * ECG_resamplerate
            ecg_resample = ecg_resample [0:num_point]
            ecg_resample2 = ecg_resample2 [0:num_point]
    except NameError:
        print ("This variable is not defined")
    n_sample = len(ecg_resample)
    noise_BL = noise_BL[0:n_sample]
    pink_noise = pink_noise[0:n_sample]
    IA_Vin = ecg_resample * 1e-3 # (mV)    
    IA_DC = 0.6     
    IA_Vout = IA_TF(IA_Vin)  + IA_DC
    dt_resample = 1./ECG_resamplerate
    freq_VCO = VCO_TF(IA_Vout)
    Counter = Counter_ADC(freq_VCO, dt_resample, CLK_freq)
    return (Counter)
