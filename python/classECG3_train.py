#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

import wfdb

#import ECGlib

from defines import *

from sklearn import svm
from sklearn import metrics

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import ECGClassEval
import batchFE as FE

from sklearn import metrics

from joblib import Parallel, delayed
from joblib import dump, load

from random import random

verbose_info = 1

def train_and_test(features_train, labels_train, features_test, labels_test, C=1, G='scale', use_pca = False, scale = False, verbose = False, weight='balanced') :
  
  # Standardize features
  if scale :
    scaler = StandardScaler()
    features_train=scaler.fit_transform(features_train)
    
    features_test=scaler.transform(features_test)
  
  # PCA
  if use_pca :
    pca = PCA(whiten=True)
    features_train = pca.fit_transform(features_train)
    features_test = pca.transform(features_test)
  
  # Train and test classifier
  clf = svm.SVC(kernel='rbf', class_weight = weight, verbose=False, C = C, gamma=G)
  clf.fit(features_train,labels_train)
  
  labels_predict = clf.predict(features_train)
  confmat_train = metrics.confusion_matrix(labels_train, labels_predict, labels=classes)
  
  labels_predict = clf.predict(features_test)
  confmat_test = metrics.confusion_matrix(labels_test, labels_predict, labels=classes)
  
  return confmat_test, confmat_train

def train_and_test_fold(features, folds, fold_idx, subset, labels, N_folds = 0, C=1, G='scale', use_pca = False, scale = False,verbose = False, weight='balanced') :
  test_rec = folds[fold_idx]
  train_rec = np.concatenate(np.delete(folds, fold_idx,axis=0))
  
  test_idx = np.isin(subset,test_rec)
  train_idx = np.isin(subset,train_rec)
  
  if(verbose):
    print("----  Fold {:d}/{:d} started  ----".format(fold_idx+1,N_folds))
    print("Test record [{:s}] ({:d} beats);\t train records[{:s}] ({:d} beats)".format(', '.join(map(str,test_rec)),np.sum(test_idx),', '.join(map(str,train_rec)),np.sum(train_idx)))
  
  features_train = features[train_idx]
  labels_train = labels[train_idx]
  features_test = features[test_idx]
  labels_test = labels[test_idx]
  
  # Standardize features
  if scale :
    scaler = StandardScaler()
    features_train=scaler.fit_transform(features_train)
    
    features_test=scaler.transform(features_test)
  
  # PCA
  if use_pca :
    pca = PCA(whiten=True)
    features_train = pca.fit_transform(features_train)
    features_test = pca.transform(features_test)
    
  # Train and test classifier
  clf = svm.SVC(kernel='rbf', class_weight = weight, verbose=False, C = C, gamma=G)

  clf.fit(features_train,labels_train)
  
  labels_predict = clf.predict(features_train)
  confmat_train = metrics.confusion_matrix(labels_train, labels_predict, labels=classes)
  
  labels_predict = clf.predict(features_test)
  confmat_test = metrics.confusion_matrix(labels_test, labels_predict, labels=classes)
  
  if(verbose):
    print("----  Fold {:d}/{:d} done  ----".format(fold_idx+1,N_folds))
    print('Classification result on training set')
    print(confmat_train)
    print('Classification result on test set')
    print(confmat_test)
  
  return confmat_test

def cross_val(features, subset, labels, features_select = None, C = 1, G='scale', verbose = False, weight = 'balanced') :
  N_rec = int(np.max(subset)+1)
  rec_idx = list(np.random.permutation(N_rec))

  folds = np.reshape(rec_idx, (-1,2))
  #folds = [[]]*8
  #folds[0] = rec_idx[0:2]
  #folds[1] = rec_idx[2:4]
  #folds[2] = rec_idx[4:7]
  #folds[3] = rec_idx[7:10]
  #folds[4] = rec_idx[10:13]
  #folds[5] = rec_idx[13:16]
  #folds[6] = rec_idx[16:19]
  #folds[7] = rec_idx[19:22]
  #folds = np.array(folds)
  N_folds = len(folds)
  print(" ******* Cross-validation of classifier : {:d} folds ******* ".format(N_folds))
  
  if not (np.array(features_select).any() == None) :
    print("Features used : [{:s}] (C = {:4.2f}, G = {:4.2f})".format(', '.join(map(str,features_select)),C,G))
    features = features[:,features_select]
  else :
    print("No features filter")
 
  result = Parallel(n_jobs=12)(delayed(train_and_test_fold)(features, folds, fold_idx, subset, labels, N_folds = N_folds, C=C, G=G, verbose = verbose, weight=weight) for fold_idx in range(0,N_folds)) 

  jk,_,_ = ECGClassEval.evalStats_3(np.sum(result,axis=0))
  print(jk)
  return jk
  
def fit_C(features, subset, labels, features_select = None, G = 'scale', verbose = False) :
  #Cs = np.logspace(-1.5,1,6)
  #Cs = [0.05]
  Cs=[0.01, 0.1, 1]
  N_C = len(Cs)
  jk = np.zeros_like(Cs)
  print("Fit C parameter")
  print("Search grid : {:s}".format(', '.join(map(str,Cs))))
  for i in range(0,N_C) :
    C = Cs[i]
    print("Testing C parameter : {:f}".format(C))
    jk[i] = cross_val(features, subset, labels, features_select = features_select, C=C, G=G, verbose = verbose)
  
  max_sen_idx = np.argmax(jk)
  
  print("Maximum jk index {:4.2f} - C = {:4.2f}".format(jk[max_sen_idx],Cs[max_sen_idx]))
  
  return jk[max_sen_idx],Cs[max_sen_idx]

def fit_g(features, subset, labels, features_select = None, verbose = False) :
  Gs = np.logspace(-2,1,4)
  N_G = len(Gs)
  jk = np.zeros_like(Gs)
  Cs = np.zeros_like(Gs)
  print("Fit Gamma parameter")
  print("Search grid : {:s}".format(', '.join(map(str,Gs))))
  for i in range(0,N_G) :
    G = Gs[i]
    print("Testing Gamma parameter : {:f}".format(G))
    jk[i],Cs[i] = fit_C(features, subset, labels, features_select = features_select, G = G/len(features_select), verbose = verbose)
  
  max_sen_idx = np.argmax(jk)
  
  print("Maximum jk index {:4.2f} - C = {:4.2f}, G = {:4.2f}".format(jk[max_sen_idx],Cs[max_sen_idx],Gs[max_sen_idx]))
  
  return jk[max_sen_idx],Cs[max_sen_idx], Gs[max_sen_idx]
  
def select_features(features, subset, labels, N_select_feat = 10, initial_set = []) :
  N_beat,N_feat = np.shape(features)
  selected_features = np.array(initial_set).astype(int)
  jk_FS = np.zeros(N_select_feat)

  for i in range(0,N_select_feat) :
    jk = np.zeros(N_feat)
    C = np.zeros(N_feat)
    for j in range(0,N_feat) :
      if np.isin(j,selected_features) :
        jk[j] = 0
      else :
        current_features = np.append(selected_features,j).astype(int)
        jk[j],C[j] = fit_C(features, subset, labels, features_select = current_features)
        
    new_feature = np.argmax(jk)
    print("New selected feature is {:d} with jk {:4.2f} - C = {:4.2f}".format(new_feature,jk[new_feature],C[new_feature]))
    selected_features = np.append(selected_features,new_feature)
    print("New feature set is {:s}".format(', '.join(map(str,selected_features))))
    jk_FS[i] = jk[new_feature]
    
  print("Final feature set is {:s}".format(', '.join(map(str,selected_features))))
  print(jk_FS)
  return selected_features, jk_FS
   
def select_features_restricted(features, subset, labels, N_select_feat = 10) :
  N_beat,N_feat = np.shape(features)
  selected_features = np.array([]).astype(int)
  jk_FS = np.zeros(N_select_feat)
  
  available_features = np.array(range(0,N_feat))
  
  for i in range(0,N_select_feat) :
    N_available = len(available_features)
    jk = np.zeros(N_available)
    C = np.zeros(N_available)
    for j in range(0,N_available) :
      tested_feat = available_features[j]
      current_features = np.append(selected_features,tested_feat).astype(int)
      jk[j],C[j] = fit_C(features, subset, labels, features_select = current_features)
      
    ord_idx = np.argsort(-jk)
    features_ordered = available_features[ord_idx]
    new_feature = features_ordered[0]
    available_features = features_ordered[1:np.maximum(N_select_feat,np.ceil(N_feat/(i+2)).astype(int))+1]
    print("New selected feature is {:d} with jk {:4.2f} - C = {:4.2f}".format(new_feature,jk[ord_idx[0]],C[ord_idx[0]]))
    selected_features = np.append(selected_features,new_feature)
    print("New feature set is {:s}".format(', '.join(map(str,selected_features))))
    print("Remaining features are {:s}".format(', '.join(map(str,available_features))))
    jk_FS[i] = jk[ord_idx[0]]
    
  print("Final feature set is {:s}".format(', '.join(map(str,selected_features))))
  print(jk_FS)
  return selected_features, jk_FS
  
def select_features_SFFS(features, subset, labels, N_select_feat = 10,restart=False) :
  N_beat,N_feat = np.shape(features)
  if restart:
    backup = load('train_bkp.sav')
    selected_features = backup['selected_features']
    jk_FS = backup['jk_FS']
    available_features = backup['available_features']
    C = backup['C']
    G = backup['G']
    start = backup['start']
  else:
    selected_features = np.array([]).astype(int)
    jk_FS = np.zeros(N_select_feat)
    available_features = np.array(range(0,N_feat))
    C = 0.1
    G = 0.1
    start=0
  
  i=start
  while i<N_select_feat :
    # Forward
    print("Forward step #{}".format(i))
    N_available = len(available_features)
    jk = np.zeros(N_available)
    for j in range(0,N_available) :
      tested_feat = available_features[j]
      current_features = np.append(selected_features,tested_feat).astype(int)
      jk[j] = cross_val(features, subset, labels, features_select = current_features, C=C, G=G/len(current_features))
    
    print("Result performance {:s}".format(', '.join(map(str,np.round(jk,2)))))
    ord_idx = np.argsort(-jk)
    features_ordered = available_features[ord_idx]
    new_feature = features_ordered[0]
    available_features = features_ordered[1:np.maximum(N_select_feat,np.ceil(N_feat/(i+2)).astype(int))+1]
    print("New selected feature is {:d} with jk {:4.2f} - C = {:4.2f}, G = {:4.2f}".format(new_feature,jk[ord_idx[0]],C,G))
    selected_features = np.append(selected_features,new_feature)
    print("New feature set is {:s}".format(', '.join(map(str,selected_features))))
    print("Remaining features are {:s}".format(', '.join(map(str,available_features))))
    
    #jk_FS[i],C,G = fit_g(features, subset, labels, features_select = selected_features)
    #print("Adjusted performance is jk index {:4.2f} with C = {:4.2f}, G = {:4.2f}".format(jk_FS[i],C,G))
    jk_FS[i] = jk[ord_idx[0]]
    
    # Backward
    if i==0:
      print("No backward step for #{}".format(i))
      i=i+1
    else:
      print("Backward step #{}".format(i))
      jk = np.zeros(i)
      for j in range(0,i) :
        current_features = np.delete(selected_features,j).astype(int)
        jk[j] = cross_val(features, subset, labels, features_select = current_features, C=C, G=G/len(current_features))
      print("Result performance {:s}".format(', '.join(map(str,np.round(jk,2)))))
      idx_max = np.argmax(jk)
      if jk[idx_max]>jk_FS[i-1]:
        print("Feature {:d} deleted with jk {:4.2f} - C = {:4.2f}".format(selected_features[idx_max],jk[idx_max],C))
        available_features = np.insert(available_features,0,selected_features[idx_max])
        selected_features = np.delete(selected_features,idx_max).astype(int)
        print("New feature set is {:s}".format(', '.join(map(str,selected_features))))
        print("Remaining features are {:s}".format(', '.join(map(str,available_features))))
        
        #jk_FS[i-1],C,G = fit_g(features, subset, labels, features_select = selected_features)
        #print("Adjusted performance is jk index {:4.2f} with C = {:4.2f}, G = {:4.2f}".format(jk_FS[i-1],C,G))
        jk_FS[i-1] = jk[idx_max]
      else:
        print("No feature deleted")
        i = i+1
    dump({'selected_features':selected_features,'available_features':available_features,'start':start,'jk_FS':jk_FS,'C':C,'G':G},'train_bkp.sav')
    
  print("Final feature set is {:s}".format(', '.join(map(str,selected_features))))
  print(jk_FS)
  return selected_features, jk_FS

############################
###   Read features      ###
############################

# Read from file
subset_train, features_train, labels_train,names = FE.readFE(file_name = "output/features_run_all_train_norm.dat",read_header=False)
subset_test, features_test, labels_test,_ = FE.readFE(file_name = "output/features_run_all_test_norm.dat",read_header=False)

FS_init = FS = np.concatenate((np.arange(0,4),np.arange(29,64)))
features_train = features_train[:,FS_init]
features_test = features_test[:,FS_init]

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

#print("Scale all")
##pca = PCA()
##features_train[:,4:] = pca.fit_transform(features_train[:,4:])
##print("Explained variance (%) [{:s}]".format(', '.join(map(str,np.round(100*pca.explained_variance_ratio_,2)))))
##features_test[:,4:] = pca.transform(features_test[:,4:])
#
scaler = StandardScaler()
features_train=scaler.fit_transform(features_train)
features_test=scaler.transform(features_test)
#
#FS, j = select_features_SFFS(features_train, subset_train, labels_train, N_select_feat = 20,restart=False)
#
#print('Testing phase')

FS = [ 0,  1, 18,  8, 30, 14, 32, 13, 29, 33, 31, 12, 15, 35, 28, 36, 11, 23, 37, 25]

j_val = cross_val(features_train, subset_train, labels_train, features_select = FS, C = 0.1, G=0.1/len(FS))
cm_test,cm_train = train_and_test(features_train[:,FS], labels_train, features_test[:,FS], labels_test, C=0.1, G=0.1/len(FS))
j_test,_,_ = ECGClassEval.evalStats_3(cm_test)
print(FS)
print(j_val)
print(j_test)