# Digital back-end model
#
# R. Dekimpe
# Last update: 06.2020

import numpy as np
import matplotlib.pyplot as plt

import ECGlib
import QRSlib

#
def digitalBackEndModel(ECG,time, params = {}, showFigures = False):
  
  N = len(time)
  
  # Center signal
  ecg_zero = ECG[0][0]
  ECG = ECG - ecg_zero
  
  # Initialize
  ECGlib.init()
  
  det_time = []
  det_labels = []
  features = []
  # Run sample processing
  for i in range(0,N):
    output=ECGlib.compute_features(np.right_shift(ECG[0][i],3))
    if output is not None:
      det_time = det_time+[time[i-output['delay']]]
      det_labels = det_labels + [output['class']]
      features = features + [output['features']]
      
  # Close 
  ECGlib.finish()
  
  # Power
  power = DBEPower(params = params)
  
  return det_labels, det_time, features, power
  
def DBEPower(params = {}, showFigures = False):
  
  bpm = 100
  
  #Detect
  detectTimePerSample = 7.5e-6
  detectPower = 484e-6
  samplesPerBeat = params["ADC_Fs"]*60/bpm
  detectEnergy = detectTimePerSample*samplesPerBeat*detectPower
  
  #Feature extraction
  FEtime = 273e-6
  FEpower = 669e-6
  FEenergy = FEtime*FEpower
  
  #SVM
  n_sv = 253
  n_feat = 12
  SVMpower = 692e-6
  SVMtime = (3.91e-2*n_feat+9.74e-3*n_sv-0.4)*1e-3
  SVMenergy = SVMpower*SVMtime
  
  #Total
  totalEnergy = detectEnergy + FEenergy + SVMenergy
  averagePower = totalEnergy*bpm/60
  print("POWER")
  print(averagePower)
  
  return averagePower