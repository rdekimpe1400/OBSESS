# Analog front-end model
#
# R. Dekimpe
# Last update: 06.2020

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

# Analog front-end signal transfer function model with non-idealities
# Inputs:
# - ECG: vector with single-lead analog signal [mV] from database
# - time: vector with corresponding time (same size as ECG)
# - time_dig: optional time vector for sampling (None if unused)
# - parameters
# Outputs:
# - ECG_dig: vector of digitized ECG samples
# - time_dig: sampling time vector (same as input time_dig if provided)
def analogFrontEndModel(ECG,time, time_dig = None, gain=60, vDC=0.6, Fs=200, N_bits=16, vref=1.2, showFigures = False):
  
  # Amplification
  gain = 60
  vDC = 0.6
  ecg_amp = (ECG*gain/1000) + vDC
  
  # A2D conversion
  time_max = time[-1]
  N_in = len(time)
  T_in = time[1]-time[0]
  N = int(N_in*Fs*T_in)
  f_interp = interp1d(time, ECG,kind = 'linear')
  if time_dig is None:
    time_dig = np.linspace(0,(N-1)/Fs,N)
  ECG_dig = ((2**N_bits)*f_interp(time_dig)/vref).astype(np.int32)
  
  # Plot signals
  if showFigures:
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle("ECG signal digitization")
    axs[0].plot(time,ECG)
    axs[0].set(ylabel="Analog ECG [mV]")
    axs[1].plot(time_dig,ECG_dig,'.')
    axs[1].set(ylabel="Digital ECG [LSB]",xlabel="Time[s]")
  
  return ECG_dig, time_dig