
from defines import *
import numpy as np
from scipy.signal import lfilter, firwin
import pywt

import math

import matplotlib.pyplot as plt

def extractRR(det_time, det_time_raw, window_length = 8, defaultRR = 0.8, maxRR = 500, Fs=200, verbose=False):
  N = len(det_time)
  defaultRR_t = int(defaultRR*Fs)
  preRR = np.zeros(N).astype(int)
  postRR = np.zeros(N).astype(int)
  localRR = np.zeros(N).astype(int)
  varRR = np.zeros(N).astype(int)
  
  if verbose :
    print("Extract RR interval features :")
    print('- Previous intervals: 1 coefficient')
    print('- Following interval: 1 coefficient')
    print('- Average RR on {:d} previous beats: 1 coefficient'.format(window_length))
    print('- Average RR on whole waveform: 1 coefficient')
  
  delta = np.minimum(maxRR,np.diff(det_time_raw))
  delta = np.concatenate((np.ones(window_length)*delta[0],delta,[delta[-1]]))
  for i in range(0,N):
    idx = np.where(det_time_raw==det_time[i])[0][0]
    preRR[i] = delta[idx+window_length-1]
    postRR[i] = delta[idx+window_length]
    localRR[i] = np.average(delta[idx+1:(idx+window_length)])
    varRR[i] = 1000*np.std(delta[idx+1:(idx+window_length)])
    
  meanRR = np.ones(N)*np.average(preRR)
  meanRR = meanRR.astype(int)
  
  preRR = 1000*preRR//localRR
  postRR = 1000*postRR//localRR
  
  features = np.stack((preRR,postRR,localRR,varRR),axis=0)
  
  names = ['RR-PRE','RR-POST','RR-AVG','RR-STD']
  
  return features,names

#def extractTimeDom(det_time, ecg, before = 100, after = 100):
#  N = len(det_time)
#  features = np.zeros(((before+after)//5,N)).astype(int)
#  ecg = np.concatenate((np.ones(before)*ecg[0],ecg,np.ones(after)*ecg[-1]))
#  for i in range(0,N):
#    t = det_time[i]
#    features[:,i] = ecg[t:(t+after+before):5]
#    
#  return features

def extractTimeDom(det_time, ecg, channel = 0):
  N = len(det_time)
  win = 100
  #Filter 12 to 50Hz
  b = firwin(20, [12.0/100, 50.0/100], window = "hamming", pass_zero=False)
  ecg_filt = lfilter(b, [1.0], ecg)[5:]
  features = np.zeros((26,N)).astype(int)
  #Add padding at extremities
  ecg_filt = np.concatenate((np.ones(win)*ecg_filt[0],ecg_filt,np.ones(win)*ecg_filt[-1]))
  qrs_idx = np.arange(-36,19,3)
  t_idx = np.arange(30,91,10)
  idx = win+np.concatenate((qrs_idx,t_idx))
  for i in range(0,N):
    t = det_time[i]
    #Select window
    ecg_win = ecg_filt[t:t+(2*win)]
    #Normalize variance
    ecg_win = 10000*ecg_win/np.std(ecg_win)
    features[:,i] = ecg_win[idx]
  
  names = []
  for i in range(0,len(qrs_idx)):
    names = names+['TIME-QRS[{};{}]'.format(channel,qrs_idx[i])]
  for i in range(0,len(t_idx)):
    names = names+['TIME-T[{};{}]'.format(channel,t_idx[i]-t_idx[0])]
  
  return features,names

def extractWavelet(det_time, ecg, before = 60, after = 60, wavelet_name = 'db4', channel = 0,verbose=False):
  N = len(det_time)
  
  w = pywt.Wavelet(wavelet_name)
  w_l = len(w.filter_bank[0])
  l_D1 = math.floor((before+after+w_l-1)/2)-1
  l_D2 = math.floor((l_D1+w_l-1)/2)
  l_D3 = math.floor((l_D2+w_l-1)/2)
  l_D4 = math.floor((l_D3+w_l-1)/2)
  l_D5 = math.floor((l_D4+w_l-1)/2)
  
  if verbose :
    print("Extract DWT features (wavelet:'{:s}', {:d} input samples)".format(wavelet_name, before+after))
    print('- Decomposition level 2: {:d} coefficients'.format(l_D2))
    print('- Decomposition level 3: {:d} coefficients'.format(l_D3))
    print('- Decomposition level 4: {:d} coefficients'.format(l_D4))
  
  D2 = np.zeros((l_D2,N)).astype(int)
  D3 = np.zeros((l_D3,N)).astype(int)
  D4 = np.zeros((l_D4,N)).astype(int)
  e = np.zeros((3,N)).astype(int)
  
  ecg = np.concatenate((np.ones(before)*ecg[0],ecg,np.ones(after)*ecg[-1]))
  for i in range(0,N):
    t = det_time[i]
    #norm = (np.max(ecg[t:(t+after+before)]) - np.min(ecg[t:(t+after+before)]))/50000;
    coeffs = pywt.wavedec(ecg[t:(t+after+before)], wavelet_name, level=4, mode ='constant')
    cA4, cD4, cD3, cD2, cD1 = coeffs
    eD2 = np.sqrt((np.sum(np.square(cD2))/l_D2))
    D2[:,i] = 10000*cD2[:-1]/eD2;
    eD3 = np.sqrt((np.sum(np.square(cD3))/l_D3))
    D3[:,i] = 10000*cD3[:-1]/eD3;
    eD4 = np.sqrt((np.sum(np.square(cD4))/l_D4))
    D4[:,i] = 10000*cD4[:-1]/eD4;
    e[:,i]=10000*np.array([eD2,eD3,eD4])/(eD2+eD3+eD4)
  #features = np.concatenate((D2,D3,D4,e), axis=0)
  #length = [l_D2,l_D3,l_D4]
  features = np.concatenate((D3,D4,e), axis=0)
  length = [l_D3,l_D4]
  
  names = []
  for i in range(0,len(D3)):
    names = names+['DWT-D3[{};{}]'.format(channel,i)]
  for i in range(0,len(D4)):
    names = names+['DWT-D4[{};{}]'.format(channel,i)]
  for i in range(0,len(e)):
    names = names+['DWT-E[{};{}]'.format(channel,i)]
  
  return features, names, length
  
def fileInit(file_name = "FE.dat"):
  file = open(file_name,'w')
  file.close()

def writeFE(subset,features,labels,names,write_header=False,file_name = "FE.dat"):
  N_feat,N_beat = np.shape(features)
  file = open(file_name,'a')
  if write_header:
    file.write('RECORD,LABEL,')
    for i in range(0,N_feat) :
      file.write(','+names[i])
    file.write('\n')
  for i in range(0,N_beat) :
    file.write('{:d},{:d}'.format(subset,labels[i]))
    for j in range(0,N_feat) :
      file.write(','+str(features[j][i]))
    file.write('\n')
  file.close()

def readFE(file_name = "FE.dat",read_header=False):
  file = open(file_name,'r')
  for i, l in enumerate(file):
    if i==1 :
      data = l.split(',')
      N_feat = len(data)-2
  if read_header:
    N_beat = i
  else:
    N_beat = i+1
  subset = np.zeros(N_beat)
  labels = np.zeros(N_beat)
  features = np.zeros((N_beat,N_feat))
  file.seek(0)
  names = ['']*N_feat
  for i, l in enumerate(file):
    if i==0 and read_header:
      data = l.split(',')
      for j in range(0,N_feat):
        names[j] = data[j+2]
    else:
      data = l.split(',')
      if read_header:
        subset[i-1] = int(data[0])
        labels[i-1] = int(data[1])
        for j in range(0,N_feat):
          features[i-1][j] = int(data[j+2])
      else:
        subset[i] = int(data[0])
        labels[i] = int(data[1])
        for j in range(0,N_feat):
          features[i][j] = int(data[j+2])
  file.close()
  features = np.array(features)
  labels = np.array(labels)
  return subset,features, labels,names
  