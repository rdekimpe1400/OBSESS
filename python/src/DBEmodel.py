# Digital back-end model
#
# R. Dekimpe
# Last update: 06.2020

import numpy as np
import matplotlib.pyplot as plt

import ECGlib

#
def digitalBackEndModel(ECG,time, annotations, params = {}, showFigures = False):
  
  N = len(time)
  
  # Center signal
  ecg_zero = ECG[0][0]
  ECG = ECG - ecg_zero
  
  # Initialize
  config = ECGlib.init()
  
  det_time = []
  det_labels = []
  ann_time = annotations['time']
  ann_labels = annotations['label']
  features = []
  # Run sample processing
  for i in range(0,N):
    output=ECGlib.compute_features(np.right_shift(ECG[0][i],3),ann_labels[(np.abs(ann_time - time[i])).argmin()])
    if output is not None:
      det_time = det_time+[time[i-output['delay']]]
      det_labels = det_labels + [output['class']]
      features = features + [output['features']]
      #if len(det_time)>10:
      #  return det_labels, det_time, features, 0
   
  # Close 
  ECGlib.finish()
  
  # Power
  power = DBEPower(config['n_sv_V']+config['n_sv_S'], config['n_feat_V'],params = params)
  
  return det_labels, det_time, features, power
  
def DBEPower(n_sv, n_feat, params = {}, showFigures = False):
  
  bpm = params["average_bpm"]
  
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
  SVMpower = 692e-6
  SVMtime = (3.91e-2*n_feat+9.74e-3*n_sv-0.4)*1e-3
  SVMenergy = SVMpower*SVMtime
  
  #Total
  totalEnergy = detectEnergy + FEenergy + SVMenergy
  averagePower = totalEnergy*bpm/60
  
  return averagePower