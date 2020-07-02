# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:02:53 2020
@author: saeidi
"""

import numpy as np
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


def input_noise(samplerate, n_sample, min_freq, max_freq, V_noise_BL, beta, f_flicker):
#    samplerate = 360
#    n_sample = 650000
#    max_freq =150 
#    min_freq = 0
#    V_noise_BL = 0.095e-6
    noise_BL = band_limited_noise(min_freq, max_freq, n_sample, samplerate)
    noise_BL = noise_BL * 13.3 * np.sqrt(n_sample) * V_noise_BL 
    
#    beta = 1 # the exponent
#    f_flicker = 1
    pink_noise = cn.powerlaw_psd_gaussian(beta, n_sample, 0) 
    pink_noise = pink_noise * 3.85 * V_noise_BL * np.sqrt(f_flicker)

    return (noise_BL, pink_noise)

