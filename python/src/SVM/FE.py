
from src.defines import *
import numpy as np
from scipy.signal import lfilter, firwin
import pywt

import math

import matplotlib.pyplot as plt

def readFE(file_name = "FE.dat",read_header=False):
  file = open(file_name,'r')
  for i, l in enumerate(file):
    if i==1 :
      data = l.split(',')
      N_feat = len(data)-3
  if read_header:
    N_beat = i
  else:
    N_beat = i+1
  subset = np.zeros(N_beat)
  time = np.zeros(N_beat)
  labels = np.zeros(N_beat)
  features = np.zeros((N_beat,N_feat))
  file.seek(0)
  names = ['']*N_feat
  for i, l in enumerate(file):
    if i==0 and read_header:
      data = l.split(',')
      for j in range(0,N_feat):
        names[j] = data[j+3]
    else:
      data = l.split(',')
      if read_header:
        subset[i-1] = int(data[0])
        time[i-1] = float(data[1])
        labels[i-1] = int(data[2])
        for j in range(0,N_feat):
          features[i-1][j] = int(data[j+3])
      else:
        subset[i] = int(data[0])
        time[i] = float(data[1])
        labels[i] = int(data[2])
        for j in range(0,N_feat):
          features[i][j] = int(data[j+3])
  file.close()
  features = np.array(features)
  labels = np.array(labels)
  return subset,features, labels,names
  