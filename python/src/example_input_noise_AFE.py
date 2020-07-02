# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:02:53 2020
@author: saeidi
"""

n_sample = 650000
samplerate = 360
max_freq =150 
min_freq = 0

V_noise_BL = 0.095e-6

beta = 1 # the exponent
f_flicker = 1

(noise_BL, pink_noise) = input_noise(samplerate, n_sample, min_freq, max_freq, V_noise_BL, beta, f_flicker)

from matplotlib import mlab
PSD_noise_BL, f_noise_BL = mlab.psd(noise_BL, NFFT=2**8, Fs=samplerate, scale_by_freq=True)
PSD_pink_noise, f_pink_noise = mlab.psd(pink_noise, NFFT=2**12, Fs=samplerate, scale_by_freq=True)
total_noise = noise_BL + pink_noise
PSD_total_noise, f_total = mlab.psd(total_noise, NFFT=2**12, Fs=samplerate, scale_by_freq=True)

plt.loglog(f_noise_BL, np.sqrt(PSD_noise_BL), color='blue')
plt.loglog(f_pink_noise, np.sqrt(PSD_pink_noise), color='red')
plt.loglog(f_total, np.sqrt(PSD_total_noise), 'black')
plt.legend(['BL noise', 'Pink linear', 'Total noise'], loc='best')
plt.xlabel('frequency [Hz]')
plt.ylabel('Density [V RMS / sqrt(Hz)]')
plt.title('generated noise')
plt.ylim([1e-9, 2e-7])
plt.xlim([0.5, 20])
plt.show()

