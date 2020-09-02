# -*- coding: utf-8 -*-
"""
Created on Thu Jul  10 12:00:01 2020

@author: saeidi
"""

import numpy as np

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

def band_limited_white_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    out = fftnoise(f)
    return out

def white_noise(f_sample,n_sample,v_therm):
    noise_power = v_therm**2 * f_sample / 2
    noise = np.random.normal(0,np.sqrt(noise_power),n_sample)
    return noise
    
def input_noise(f_sample,n_sample, params = {}):
    max_freq = 100 
    min_freq = 0
    thermal_noise = white_noise(f_sample,n_sample,params['IA_thermal_noise'])    
    beta = 1 # the exponent
    
    flicker_noise = cn.powerlaw_psd_gaussian(beta, n_sample, 0) 
    f = np.fft.rfftfreq(n_sample,1/f_sample)
    f[0] = f[1]
    f = f**(-1/2.)
    s = np.sqrt(np.mean(f**2))
    flicker_noise = flicker_noise*s*params['IA_thermal_noise']/(params['IA_flicker_noise_corner']**(-beta/2))*np.sqrt(500)
    
    total_noise = thermal_noise+flicker_noise
    return thermal_noise, flicker_noise, total_noise
