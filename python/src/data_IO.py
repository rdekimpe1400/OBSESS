# File data management
#
# R. Dekimpe
# Last update: 07.2020

import numpy as np

# Write output features to file
def save_features(detections, features,matched_labels,subset=0,file_name="output/features.dat",append=False):
  det_time = detections['time']
  N_beat,N_feat = np.shape(features)
  if append:
    file = open(file_name,'a')
  else:
    file = open(file_name,'w')
  #if write_header:
  #  file.write('RECORD,LABEL')
  #  for i in range(0,N_feat) :
  #    file.write(','+names[i])
  #  file.write(',\n')
  for i in range(20,N_beat) : # drop 20 first beats
    file.write('{:d},{:3.2f},{:d}'.format(subset,det_time[i],matched_labels[i]))
    #file.write('{:d},{:d}'.format(subset,matched_labels[i]))
    for j in range(0,N_feat) :
      file.write(','+str(features[i][j]))
    file.write('\n')
  file.close()
  
  return

# Read features from file
def read_features(file_name="output/features.dat"):
  file = open(file_name,'r')
  for i, l in enumerate(file):
    if i==1 :
      data = l.split(',')
      N_feat = len(data)-3
  N_beat = i+1
  subset = np.zeros(N_beat)
  labels = np.zeros(N_beat)
  time = np.zeros(N_beat)
  features = np.zeros((N_beat,N_feat))
  file.seek(0)
  for i, l in enumerate(file):
    data = l.split(',')
    subset[i] = int(data[0])
    time[i] = float(data[1])
    labels[i] = int(data[2])
    for j in range(0,N_feat):
      features[i][j] = int(data[j+3])
  file.close()
  features = np.array(features)
  labels = np.array(labels)
  return subset,features, labels,time

# Write signal data to file
def save_signal(signal,file_name="output/data.dat"):
  N_sig,N_data = np.shape(signal)
  print(N_sig)
  file = open(file_name,'w')
  for i in range(0,N_sig) : 
    for j in range(0,N_data) : 
      file.write('{:d},'.format(signal[i][j]))
    file.write('\n')
  file.close()
  
  return