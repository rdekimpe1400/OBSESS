# Digital back-end model
#
# R. Dekimpe
# Last update: 06.2020

import numpy as np
import matplotlib.pyplot as plt
import time as timeLib
from sklearn import metrics

import ECGlib_fast

#
def digitalBackEndModel(ECG,time, annotations, params = {}, showFigures = False):
  
  N_samples = len(time)
  
  # Center signal
  ecg_zero = ECG[0][0]
  ECG = ECG - ecg_zero
  
  print('DBE model init..',flush=True)
  # Initialize
  if params["save_feature"]:
    config = ECGlib_fast.init(N_samples,1,params["subset"],params["feature_file"])
  else:
    config = ECGlib_fast.init(N_samples,0,0,"")
  
  det_time = []
  det_labels = []
  ann_time = annotations['time']
  ann_labels = annotations['label']
  features = []
  label_gold = np.zeros((N_samples))
  # Run sample processing
  t_start = timeLib.time()
  ann_idx = 0;
  ann_idx_max = len(ann_time)
  for i in range(0,N_samples):
    if ann_idx+1 <ann_idx_max:
      if((time[i] - ann_time[ann_idx])>(ann_time[ann_idx+1] - time[i])):
        ann_idx = ann_idx+1
    label_gold[i] = ann_labels[ann_idx]
  print('DBE model start..',flush=True)
  ECG_scale = np.left_shift(ECG[0][:],np.log2(np.floor(6600000/(params["input_scale"]*params["ADC_VCO_f0"]))).astype(int))
  output=ECGlib_fast.compute_features(list(ECG_scale),list(label_gold))
  t_stop = timeLib.time()
  
  det_time = time[output['det_time']]
  det_labels = output['det_label']
  det_labels_gold = output['det_label_gold']
  
  # Print annotations
  #for i in range(0,len(ann_time)):
  #  print(ann_time[i],ann_labels[i])
  
  #for i in range(0,len(det_time)):
  #  print(det_time[i])
  
  if showFigures:
    fig, axs = plt.subplots(2, sharex=True)
    axs[0].plot(time,label_gold,'.')
    axs[0].plot(ann_time,ann_labels,'.')
    axs[0].set_xlim((0,60))
    plt.savefig('plots/DBE_labels.png')
   
  # Close 
  ECGlib_fast.finish()
  print('DBE model done!',flush=True)
  print('DBE execution time : {:4.3f}s'.format(t_stop-t_start),flush=True)
  
  # Pass SVM charcteristics
  params['SVM_SV_N_V'] = config['n_sv_V']
  params['SVM_SV_N_S'] = config['n_sv_S']
  params['SVM_feature_N_V'] = config['n_feat_V']
  params['SVM_feature_N_S'] = config['n_feat_S']
  params['beat_freq'] = len(det_time)/(N_samples/params['ADC_Fs'])
  
  return det_labels, det_time, features
