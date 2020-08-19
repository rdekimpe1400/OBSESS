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
  
  return det_labels, det_time, features