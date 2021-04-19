#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

import wfdb

#import ECGlib

import src.SVM.FE as FE
from src.defines import *

from sklearn import svm
from sklearn import metrics

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import src.SVM.ECGClassEval as ECGClassEval

from sklearn import metrics

from joblib import Parallel, delayed
from joblib import dump, load

from random import random

verbose_info = 1

def train(features_train, labels_train, params, prune = False, verbose = False, hier = False):
  if not hier:
    if params['gamma']=='scale':
      gamma = 'scale'
    else:
      gamma = params['gamma']/len(params['FS'])
  else:
    if params['gamma_2']=='scale':
      gamma = 'scale'
    else:
      gamma = params['gamma_2']/len(params['FS_2'])
  clf = svm.SVC(kernel=params['kernel'], class_weight = params['class_weight'], verbose=False, C = params['C'], gamma=gamma)
  clf.fit(features_train,labels_train)
  if prune:
    if verbose:
      print('Number of SV before pruning [{:d},{:d}] ({:d})'.format(clf.n_support_[0],clf.n_support_[1],np.sum(clf.n_support_)))
    labels_train_predict = clf.predict(features_train)
    labels_prune_selection = labels_train_predict==labels_train
    if verbose:
      confmat_preprune = metrics.confusion_matrix(labels_train, labels_train_predict)
      print(confmat_preprune)
      print('Misclassified training examples : {:d}'.format(np.sum(labels_prune_selection==False)))
    clf = svm.SVC(kernel=params['kernel'], class_weight = params['class_weight'], verbose=False, C = 1000, gamma=gamma)
    clf.fit(features_train[labels_prune_selection,:],labels_train[labels_prune_selection])
    if verbose: 
      labels_train_prune_predict = clf.predict(features_train[labels_prune_selection,:])
      confmat_postprune = metrics.confusion_matrix(labels_train[labels_prune_selection], labels_train_prune_predict)
      print(confmat_postprune)
      print('Number of SV after pruning [{:d},{:d}] ({:d})'.format(clf.n_support_[0],clf.n_support_[1],np.sum(clf.n_support_)))
  
  return clf

def train_and_test(features_train, labels_train, features_test, labels_test, params, prune = False, verbose = False, save_clf = False) :
  # Select features
  if not (np.array(params['FS_2']).any() == None):
    features_train2 = features_train[:,params['FS_2']]
    features_test2 = features_test[:,params['FS_2']]
  if not (np.array(params['FS']).any() == None):
    features_train = features_train[:,params['FS']]
    features_test = features_test[:,params['FS']]
  
  # Select classification type
  if params['type']=='NvSvV':
    labels_train_type = labels_train
    features_train_type = features_train
  elif params['type']=='NvX':
    labels_train_type = labels_train==BEAT_N
    features_train_type = features_train
  elif params['type']=='VvX':
    labels_train_type = labels_train==BEAT_V
    features_train_type = features_train
  elif params['type']=='SvV':
    labels_train_type = labels_train[labels_train!=BEAT_N]==BEAT_V
    features_train_type = features_train[labels_train!=BEAT_N,:]
  elif params['type']=='NvS':
    labels_train_type = labels_train[labels_train!=BEAT_V]==BEAT_N
    features_train_type = features_train[labels_train!=BEAT_V,:]
  elif params['type']=='hier':
    labels_train_type = labels_train==BEAT_N
    features_train_type = features_train
  else:
    print('Unexpected type. Fatal error!')
    exit()
  
  # Train and test classifier
  clf = train(features_train_type, labels_train_type, params, prune = prune, verbose = verbose)
  if save_clf:
    dump(clf,'clf.sav')
  
  # If hier, train second classifier
  if params['type']=='hier':
    labels_train_type = labels_train[labels_train!=BEAT_N]==BEAT_V
    features_train_type = features_train2[labels_train!=BEAT_N,:]
    clf2 = train(features_train_type, labels_train_type, params, prune = prune, verbose = verbose)
  
  if len(features_test)>0:
    labels_predict = clf.predict(features_test)
    labels_predict3 = np.zeros_like(labels_predict).astype(int)
    # Update class depending on type
    if params['type']=='NvSvV':
      labels_predict3 = labels_predict
    elif params['type']=='NvX':
      labels_predict3[np.logical_and(labels_test==BEAT_S,labels_predict==False)] = BEAT_S
      labels_predict3[np.logical_and(labels_test==BEAT_V,labels_predict==False)] = BEAT_V
      labels_predict3[np.logical_and(labels_test==BEAT_N,labels_predict==False)] = BEAT_S
      labels_predict3[labels_predict==True] = BEAT_N
    elif params['type']=='VvX':
      labels_predict3[np.logical_and(labels_test==BEAT_S,labels_predict==False)] = BEAT_S
      labels_predict3[np.logical_and(labels_test==BEAT_N,labels_predict==False)] = BEAT_N
      labels_predict3[np.logical_and(labels_test==BEAT_V,labels_predict==False)] = BEAT_S
      labels_predict3[labels_predict==True] = BEAT_V
    elif params['type']=='SvV':
      labels_predict3[labels_test==BEAT_N] = BEAT_N
      labels_predict3[np.logical_and(labels_test!=BEAT_N,labels_predict==False)] = BEAT_S
      labels_predict3[np.logical_and(labels_test!=BEAT_N,labels_predict==True)] = BEAT_V
    elif params['type']=='NvS':
      labels_predict3[labels_test==BEAT_V] = BEAT_V
      labels_predict3[np.logical_and(labels_test!=BEAT_V,labels_predict==False)] = BEAT_S
      labels_predict3[np.logical_and(labels_test!=BEAT_V,labels_predict==True)] = BEAT_N
    elif params['type']=='hier':
      labels_predict2 = clf2.predict(features_test2)
      labels_predict3[np.logical_and(labels_predict2==False,labels_predict==False)] = BEAT_S
      labels_predict3[np.logical_and(labels_predict2==True,labels_predict==False)] = BEAT_V
      labels_predict3[labels_predict==True] = BEAT_N
    confmat_test = metrics.confusion_matrix(labels_test, labels_predict3, labels=[BEAT_N,BEAT_S,BEAT_V])
  else:
    confmat_test = np.zeros((3,3))
  
  return confmat_test

def train_and_test_fold(features, folds, fold_idx, subset, labels,params, N_folds = 0, verbose = False) :
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
  
  # Train and test classifier
  confmat_test=train_and_test(features_train, labels_train, features_test, labels_test, params, verbose = verbose)
  
  if(verbose):
    print("----  Fold {:d}/{:d} done  ----".format(fold_idx+1,N_folds))
    print('Classification result on test set')
    print(confmat_test)
  
  return confmat_test

def cross_val(features, subset, labels, params, verbose = False) :
  N_rec = int(np.max(subset)+1)
  rec_idx = list(np.random.permutation(N_rec))

  folds = np.reshape(rec_idx, (-1,2))
  N_folds = len(folds)
  print(" ******* Cross-validation of classifier : {:d} folds ******* ".format(N_folds))
  
  if not (np.array(params['FS']).any() == None) :
    print("Features used : [{:s}] (C = {:4.2f}, G = {})".format(', '.join(map(str,params['FS'])),params['C'],params['gamma']),flush=True)
  else :
    print("No features filter")
 
  result = Parallel(n_jobs=12)(delayed(train_and_test_fold)(features, folds, fold_idx, subset, labels, params,N_folds = N_folds, verbose = False) for fold_idx in range(0,N_folds)) 
  confmat = np.sum(result,axis=0)
  
  # Extract output metric
  output = ECGClassEval.evalStats_3(confmat)
  
  return output
  
def fit_C(features, subset, labels, params, verbose = True) :
  #Cs = np.logspace(-1.5,1,6)
  #Cs=[0.1]
  Cs = np.power(2.,np.arange(-6,8,2))
  N_C = len(Cs)
  jk = np.zeros_like(Cs)
  print("Fit C parameter")
  print("Search grid : {:s}".format(', '.join(map(str,Cs))))
  for i in range(0,N_C) :
    C = Cs[i]
    print("Testing C parameter : {:f}".format(C))
    params['C']=C
    jk[i] = cross_val(features, subset, labels, params, verbose = verbose)
  
  max_sen_idx = np.argmax(jk)
  
  print("Maximum jk index {:4.2f} - C = {:4.2f}".format(jk[max_sen_idx],Cs[max_sen_idx]))
  
  params['C']=Cs[max_sen_idx]
  
  return jk[max_sen_idx],params

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
  
def select_features(features, subset, labels, params, N_select_feat = 10) :
  N_beat,N_feat = np.shape(features)
  if not (np.array(params['FS']).any() == None):
    selected_features = np.array(params['FS']).astype(int)
  else:
    selected_features = np.array([]).astype(int)
  jk_FS = np.zeros(N_select_feat)

  for i in range(0,N_select_feat) :
    jk = np.zeros(N_feat)
    #C = np.zeros(N_feat)
    for j in range(0,N_feat) :
      if np.isin(j,selected_features) :
        jk[j] = 0
      else :
        current_features = np.append(selected_features,j).astype(int)
        #jk[j],C[j] = fit_C(features, subset, labels, features_select = current_features, G=0.1/len(current_features))
        params['FS'] = current_features
        jk[j] = cross_val(features, subset, labels, params)
    new_feature = np.argmax(jk)
    print("New selected feature is {:d} with jk {:4.2f}".format(new_feature,jk[new_feature]))
    selected_features = np.append(selected_features,new_feature)
    print("New feature set is {:s}".format(', '.join(map(str,selected_features))))
    jk_FS[i] = jk[new_feature]
    
  print("Final feature set is {:s}".format(', '.join(map(str,selected_features))))
  print(jk_FS)
  params['FS'] = selected_features
  return jk_FS, params
   