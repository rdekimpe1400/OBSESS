# Digital back-end model
#
# R. Dekimpe
# Last update: 06.2020

import numpy as np
import matplotlib.pyplot as plt

import ECGlib

#
def digitalBackEndModel(ECG,time, Fs=200, showFigures = False):
  
  N = len(time)
  
  # Center signal
  ecg_zero = ECG[0][0]
  ECG = ECG - ecg_zero
  
  # Apply detection algorithm
  ecg_filt = np.zeros(N).astype(np.int32)
  ecg_detDelay = np.zeros(N).astype(np.int32)
  ECGlib.init()
  for i in range(0,N) :
    ecg_filt[i] = ECGlib.filter(ECG[0][i])
    ecg_detDelay[i] = ECGlib.detect(ECG[0][i],ecg_filt[i])
  
  # Transform detection delay output to logical signal values
  det_sample = np.array([])
  for i in range(0,N) :
    if ecg_detDelay[i]>0 :
      det_sample=np.append(det_sample,i-ecg_detDelay[i])
  
  plot_detectAlgo = showFigures
  if plot_detectAlgo :
    fig, axs = plt.subplots(3,sharex=True)
    fig.suptitle("ECG beat detection algorithm")
    axs[0].plot(time,ECG[0])
    axs[0].set(ylabel="ECG")
    axs[1].plot(time,ecg_filt)
    axs[1].set(ylabel="ECG transform")
    axs[2].plot(time,ecg_detDelay)
    axs[2].set(ylabel="ECG detection",xlabel="Time[s]")
    
  time_det = det_sample/Fs
  labels = 0
  return labels, time_det