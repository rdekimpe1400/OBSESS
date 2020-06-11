# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:23:18 2020

@author: saeidi

https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy

"""

#Here you find matlab code from Aslak Grinsted, creating noise with a specified power spectrum. It can easily be ported to python:

import numpy as np

def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

#You can use it for your case like this:

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)
#Seems to work as far as I see. For listening to your freshly created noise:

#from scipy.io import wavfile
samplerate = 5000
max_freq = 2000
min_freq = 500

noise = band_limited_noise(min_freq, max_freq, 2**12, samplerate)
#noise = noise * 6.57e-5

#for scaling
#noise = np.int16(noise * (2**15 - 1))
#wavfile.write("test.wav", 44100, noise)

import matplotlib.pyplot as plt
plt.plot(noise, color='blue')
#plt.plot(noise, color='blue')
plt.title('band_limited_noise')
plt.xlabel('Sample')
plt.show()


#from matplotlib import mlab

# s, f = mlab.psd(noise, NFFT=2**10, Fs=samplerate, scale_by_freq=False)
from scipy import signal
#scipy.signal.periodogram(x, fs=1.0, window=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)
f, PSD = signal.periodogram(noise, samplerate, 'flattop', scaling='spectrum')
fnew, PSDnew = signal.welch(noise, samplerate, 'flattop', scaling='spectrum')

#plt.loglog(f,PSDnew, color='blue')
#plt.grid(True)
#plt.plot(y, color='blue')
#plt.title('band_limited_noise')
#plt.xlabel('Frequency')
#plt.show()
plt.subplot(2,1,1)
plt.semilogy(f, np.sqrt(PSD))
plt.xlabel('frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')


plt.subplot(2,1,2)
plt.semilogy(fnew, np.sqrt(PSDnew))
plt.xlabel('frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.show()

#samplerate = 100000
#n= 2**15
print(np.sqrt(PSD.max()))
print(np.sqrt(PSDnew.max()))