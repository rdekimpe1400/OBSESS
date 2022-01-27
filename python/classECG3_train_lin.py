#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

import wfdb

#import ECGlib

from src.defines import *

from sklearn import svm
from sklearn import metrics

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import src.SVM.ECGclassification as ECGclassification
import src.SVM.ECGClassEval as ECGClassEval
import src.SVM.FE as FE

from sklearn import metrics

from joblib import Parallel, delayed
from joblib import dump, load

from random import random

verbose_info = 1


############################
###   Read features      ###
############################

# Read from file
subset_train, features_train, labels_train,names = FE.readFE(file_name = "output/features_run_all_train.dat",read_header=False)
subset_test, features_test, labels_test,_ = FE.readFE(file_name = "output/features_run_all_test.dat",read_header=False)

FE_preselect_V = np.array([0,1,2,3,4,5,6,7,21,22,23,24,25,26,27,28,29,30,31,32,33,34,118,119,120,121,122,123,124,144,145,146,147,148,149,150,151,152,156,157,158,159,160])
FE_preselect_S = np.array([0,1,2,3,4,5,6,7,12,13,14,15,16,17,18,137,138,139,140,141,158,159,160,161,172,173,174])

SV_params_V = {'kernel':'linear','C':0.1, 'gamma':0.1, 'degree':3, 'FS':None, 'pruning_D': 0, 'class_weight':'balanced','type':'VvX'}
SV_params_S = {'kernel':'linear','C':0.1, 'gamma':0.1, 'degree':3, 'FS':None, 'pruning_D': 0, 'class_weight':'balanced','type':'NvS'}

classes = np.array([NOT_BEAT,BEAT_N,BEAT_S,BEAT_V,BEAT_F,BEAT_Q])

# Reject Q
reject_NOT_BEAT = True
if reject_NOT_BEAT :
  idx_reject = np.where(labels_train == NOT_BEAT)
  features_train = np.delete(features_train,idx_reject,axis=0)
  labels_train = np.delete(labels_train,idx_reject,axis=0)
  subset_train = np.delete(subset_train,idx_reject,axis=0)
  idx_reject = np.where(labels_test == NOT_BEAT)
  features_test = np.delete(features_test,idx_reject,axis=0)
  labels_test = np.delete(labels_test,idx_reject,axis=0)
  subset_test = np.delete(subset_test,idx_reject,axis=0)
  classes = np.delete(classes,np.where(classes==NOT_BEAT))

# Reject Q
reject_Q = True
if reject_Q :
  idx_reject = np.where(labels_train == BEAT_Q)
  features_train = np.delete(features_train,idx_reject,axis=0)
  labels_train = np.delete(labels_train,idx_reject,axis=0)
  subset_train = np.delete(subset_train,idx_reject,axis=0)
  idx_reject = np.where(labels_test == BEAT_Q)
  features_test = np.delete(features_test,idx_reject,axis=0)
  labels_test = np.delete(labels_test,idx_reject,axis=0)
  subset_test = np.delete(subset_test,idx_reject,axis=0)
  classes = np.delete(classes,np.where(classes==BEAT_Q))
  
# Merge F
merge_F = True
if merge_F :
  labels_train[labels_train == BEAT_F] = BEAT_V
  labels_test[labels_test == BEAT_F] = BEAT_V
  classes = np.delete(classes,np.where(classes==BEAT_F))

# Remove V
remove_V = False
if remove_V :
  idx_reject = np.where(labels_train == BEAT_V)
  features_train = np.delete(features_train,idx_reject,axis=0)
  labels_train = np.delete(labels_train,idx_reject,axis=0)
  subset_train = np.delete(subset_train,idx_reject,axis=0)
  idx_reject = np.where(labels_test == BEAT_V)
  features_test = np.delete(features_test,idx_reject,axis=0)
  labels_test = np.delete(labels_test,idx_reject,axis=0)
  subset_test = np.delete(subset_test,idx_reject,axis=0)
  classes = np.delete(classes,np.where(classes==BEAT_V))

# Merge S
merge_S = False
if merge_S :
  labels_train[labels_train == BEAT_S] = BEAT_N
  labels_test[labels_test == BEAT_S] = BEAT_N
  classes = np.delete(classes,np.where(classes==BEAT_S))
  
N_beat_train,N_feat = np.shape(features_train)
N_beat_test,_ = np.shape(features_test)
N_rec_train = int(np.max(subset_train)+1)
N_rec_test = int(np.max(subset_test)+1)

if verbose_info :
  print("####################################################")
  print("Data set information")
  print("")
  print("Features : {:d}".format(N_feat))
  print("")
  print("Training set : {:d}".format(N_beat_train))
  print(" - N\t\t\t : %d" % (np.sum(labels_train==BEAT_N)))
  print(" - S\t\t\t : %d" % (np.sum(labels_train==BEAT_S)))
  print(" - V\t\t\t : %d" % (np.sum(labels_train==BEAT_V)))
  print(" - F\t\t\t : %d" % (np.sum(labels_train==BEAT_F)))
  print(" - Q\t\t\t : %d" % (np.sum(labels_train==BEAT_Q)))
  print(" - X\t\t\t : %d" % (np.sum(labels_train==NOT_BEAT)))
  print("")
  print("Number of training records : {:d}".format(N_rec_train))
  print("")
  print("Training set : {:d}".format(N_beat_test))
  print(" - N\t\t\t : %d" % (np.sum(labels_test==BEAT_N)))
  print(" - S\t\t\t : %d" % (np.sum(labels_test==BEAT_S)))
  print(" - V\t\t\t : %d" % (np.sum(labels_test==BEAT_V)))
  print(" - F\t\t\t : %d" % (np.sum(labels_test==BEAT_F)))
  print(" - Q\t\t\t : %d" % (np.sum(labels_test==BEAT_Q)))
  print(" - X\t\t\t : %d" % (np.sum(labels_test==NOT_BEAT)))
  print("")
  print("Number of test records : {:d}".format(N_rec_test))
  print("####################################################")

############################
###   Train classifier   ###
############################

print('-- S classifier --')
features_train_S = features_train[:,FE_preselect_S]
features_test_S = features_test[:,FE_preselect_S]
SV_params = SV_params_S

scaler = StandardScaler()
features_train_S=scaler.fit_transform(features_train_S)
features_test_S=scaler.transform(features_test_S)

print('Feature select')

FS, j = ECGclassification.select_features_SFFS(features_train_S, subset_train, labels_train, SV_params, N_select_feat = 20)

print(FS)
print(j)

print('Testing phase')

SV_params['FS'] = FS
j_val = ECGclassification.cross_val(features_train_S, subset_train, labels_train, SV_params)
cm_test = ECGclassification.train_and_test(features_train_S, labels_train, features_test_S, labels_test, SV_params)
j_test = ECGClassEval.evalStats_3(cm_test)
print(FS)
print(j_val)
print(j_test)

print('-- V classifier --')
features_train_V = features_train[:,FE_preselect_V]
features_test_V = features_test[:,FE_preselect_V]
SV_params = SV_params_V

scaler = StandardScaler()
features_train_V=scaler.fit_transform(features_train_V)
features_test_V=scaler.transform(features_test_V)

print('Feature select')

FS, j = ECGclassification.select_features_SFFS(features_train_V, subset_train, labels_train, SV_params, N_select_feat = 20)

print(FS)
print(j)

print('Testing phase')

SV_params['FS'] = FS
j_val = ECGclassification.cross_val(features_train_V, subset_train, labels_train, SV_params)
cm_test = ECGclassification.train_and_test(features_train_V, labels_train, features_test_V, labels_test, SV_params)
j_test = ECGClassEval.evalStats_3(cm_test)
print(FS)
print(j_val)
print(j_test)
