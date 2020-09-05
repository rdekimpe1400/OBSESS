#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from sklearn.preprocessing import StandardScaler
from sklearn import svm

from src.SVM.defines import *

def updateModel(labels, features, params = {}, verbose = True):
  C_file = params['SVM_model_file']
  
  CFileInit(file_name = C_file) 
  
  # Train model
  labels, features = modifLabels(labels, features) # Keep relevant classes only 
  model, scaler, FS = trainSVM(labels, features, params)
  
  if verbose:
    print("--------------------------------------------------------")
    print("Open SVM model")
    print("\tClasses : {}".format(model.classes_))
    print("\tNumber of SV : {} (total: {})".format(model.n_support_, np.sum(model.n_support_)))
    sv_shape = np.shape(model.support_vectors_)
    print("\tNumber of features : {}".format(sv_shape[1]))
    print("\tSelected features : {}".format(FS))
    params = model.get_params()
    print("\tKernel : {}".format(params['kernel']))
    if params['kernel']=='rbf':
      print("\tC parameter : {}".format(params['C']))
      print("\tGamma parameter : {}".format(params['gamma']))
    else:
      print("\033[91mUnsupported kernel type!! \033[0m")
      sys.exit()
    print("--------------------------------------------------------")
  
  SVMtranslate(model,scaler,FS, file_name = C_file)
  
  CFileClose(file_name = C_file)

def trainSVM(labels, features, params = {}):
  scaler = StandardScaler() # Scale
  features_scale=scaler.fit_transform(features)

  FS = params['SVM_feature_select']
  features_select = features_scale[:,FS] #Select
  
  model_unpruned = svm.SVC(kernel='rbf', class_weight = 'balanced', verbose=False, C = params['SVM_C'], gamma=params['SVM_gamma']/len(FS), decision_function_shape='ovo') # SVM training
  model_unpruned.fit(features_select,labels)
  
  labels_predict = model_unpruned.predict(features_select)
  
  keep = np.where(labels_predict==labels)
  
  features_reduce = features_select[keep]
  labels_reduce = labels[keep]

  model = svm.SVC(kernel='rbf', gamma= params['SVM_gamma']/len(FS), C=100000)
  model.fit(features_reduce,labels_reduce)
  
  return model, scaler, FS

def CFileInit(file_name = "SVM.h"):
  file = open(file_name,'w')
  file.write('//**********************************************\n')
  file.write('//\n')
  file.write('//  SVM model data\n')
  file.write('//\n')
  file.write('//**********************************************\n\n')
  file.write('#ifndef SVM_H_\n')
  file.write('#define SVM_H_\n\n')
  
  return 1

def CFileClose(file_name = "SVM.h"):
  file = open(file_name,'a')
  file.write('\n#endif //SVM_H_\n')
  file.close()
  
  return 1

def SVMmodelOpen(file_name = "SVM.sav", scale_file_name = "scale.sav", FS_file_name = "FS.sav", verbose = False):
  model=load(file_name)
  scaler=load(scale_file_name)
  FS=load(FS_file_name)
  if verbose:
    print("--------------------------------------------------------")
    print("Open SVM model")
    print("\tClasses : {}".format(model.classes_))
    print("\tNumber of SV : {} (total: {})".format(model.n_support_, np.sum(model.n_support_)))
    sv_shape = np.shape(model.support_vectors_)
    print("\tNumber of features : {}".format(sv_shape[1]))
    print("\tSelected features : {}".format(FS))
    params = model.get_params()
    print("\tKernel : {}".format(params['kernel']))
    if params['kernel']=='rbf':
      print("\tC parameter : {}".format(params['C']))
      print("\tGamma parameter : {}".format(params['gamma']))
    else:
      print("\033[91mUnsupported kernel type!! \033[0m")
      sys.exit()
    print("--------------------------------------------------------")
  return model, scaler, FS

def SVMtranslate(model,scaler,FS, file_name = "svm_int.h"):
  sv = model.support_vectors_
  sv_shape = np.shape(sv)
  sv_params = model.get_params()
  n_sv = np.sum(model.n_support_)
  n_feat = sv_shape[1]
  file = open(file_name,'a')
  if sv_params['gamma']=='scale':
    gamma = 1/n_feat
  else:
    gamma = sv_params['gamma']
  
  feature_data = 16
  feature_dist_data = 2*feature_data
  kernel_data = 32
  decision_data = 32
  
  file.write("typedef int{:d}_t feature_data_t;\n".format(feature_data))
  file.write("typedef int{:d}_t feature_dist_data_t;\n".format(feature_dist_data))
  file.write("typedef int{:d}_t kernel_data_t;\n".format(kernel_data))
  file.write("typedef int{:d}_t decision_data_t;\n".format(decision_data))
  
  # Compute quantization range
  feature_max_range = 20 # from dataset
  feature_scale = (2**(feature_data-1))/feature_max_range
  feature_shift = np.floor(np.log2(feature_scale)).astype(int) #Avoid feature saturation
  
  feature_acc_shift = np.ceil(np.log2(n_feat)).astype(int) #Avoid accumulate saturation
  
  scale_shift = 16
  
  coef_shift = 10
  
  kernel_acc_shift = 16 #Avoid accumulate saturation
  
  # Write model data
  file.write('// Feature pre-processing\n')
  file.write('const int feature_select_idx[{:d}] = {{'.format(n_feat))
  for i in range(0,n_feat):
    file.write("{:d}".format(FS[i]))
    if i<(n_feat-1):
      file.write(", ")
  file.write("};\n")
  file.write('const feature_data_t scale_mean[{:d}] = {{'.format(n_feat))
  for i in range(0,n_feat):
    file.write("{:d}".format(np.round(scaler.mean_[FS[i]]).astype(int)))
    if i<(n_feat-1):
      file.write(", ")
  file.write("};\n")
  file.write('const int32_t scale_std[{:d}] = {{'.format(n_feat))
  for i in range(0,n_feat):
    file.write("{:d}".format(np.round((2**(feature_shift+scale_shift))/np.sqrt(scaler.var_[FS[i]])).astype(int)))
    if i<(n_feat-1):
      file.write(", ")
  file.write("};\n")
  file.write('const int scale_shift = {};\n'.format(scale_shift))
  
  file.write('// Exponential pre-calc coeff\n')
  exp_shift = 16
  ai_min = -15
  ai_max = 5
  ai = np.exp(-np.power(2.0,np.arange(ai_min,ai_max+1)))*(2**exp_shift)
  ai = ai.astype(int)
  
  file.write('const kernel_data_t exp_ai[{:d}] = {{'.format(len(ai)))
  for i in range(0,len(ai)):
    file.write("{:d}".format(ai[i]))
    if i<(len(ai)-1):
      file.write(", ")
  file.write("};\n")
  
  file.write('const int exp_ai_min = {};\n'.format(ai_min))
  file.write('const int exp_ai_max = {};\n'.format(ai_max))
  file.write('const int exp_shift = {};\n'.format(exp_shift))
  
  
  file.write('// SVM model data\n')
  
  file.write('const int n_feat = {};\n'.format(n_feat))
  
  file.write('const int gam_inv = {};\n'.format(int(1/gamma)))
  
  file.write('const int feature_shift = {};\n'.format(int(feature_shift)))
  
  file.write('const int feature_acc_shift = {};\n'.format(int(feature_acc_shift)))
  
  file.write('const int kernel_acc_shift = {};\n'.format(int(kernel_acc_shift)))
  
  file.write('const int n_sv[3] = {{{}, {}, {}}};\n'.format(model.n_support_[0], model.n_support_[1], model.n_support_[2]))
  
  file.write('const int n_sv_tot = {};\n'.format(n_sv))
  
  file.write('const int start_sv[3] = {{{}, {}, {}}};\n'.format(0, model.n_support_[0], model.n_support_[0]+model.n_support_[1]))
  
  interp_int = np.round(model.intercept_*(2**(exp_shift+coef_shift-kernel_acc_shift))).astype(int)
  file.write('const decision_data_t rho[3] = {{{}, {}, {}}};\n'.format(interp_int[0], interp_int[1], interp_int[2]))
  
  file.write('const decision_data_t sv_coef[2][{:d}] = {{{{'.format(n_sv))
  for i in range(0,n_sv):
    file.write("{}".format(int(model.dual_coef_[0,i]*(2**coef_shift))))
    if i<(n_sv-1):
      file.write(", ")
  file.write("}, {")
  for i in range(0,n_sv):
    file.write("{}".format(int(model.dual_coef_[1,i]*(2**coef_shift))))
    if i<(n_sv-1):
      file.write(", ")
  file.write("}};\n")
  
  file.write('const feature_data_t sv[{:d}][{:d}] = {{\n'.format(n_sv,n_feat))
  for i in range(0,n_sv):
    file.write("{")
    for j in range(0,n_feat):
      file.write("{}".format(int(sv[i,j]*(2**feature_shift))))
      if j<(n_feat-1):
        file.write(", ")
      elif i<(n_sv-1):
        file.write("},\n")
      else: 
        file.write("}\n")
  file.write("};\n")
  
  
  return 1

def modifLabels(labels, features,reject_NOTBEAT=True,reject_Q=True,merge_F = True,remove_V = False,merge_S = False, verbose = True ):

  classes = np.array([NOT_BEAT,BEAT_N,BEAT_S,BEAT_V,BEAT_F,BEAT_Q])


  if reject_NOTBEAT :
    idx_reject = np.where(labels == NOT_BEAT)
    features = np.delete(features,idx_reject,axis=0)
    labels = np.delete(labels,idx_reject,axis=0)
    classes = np.delete(classes,np.where(classes==NOT_BEAT))
  
  # Reject Q
  if reject_Q :
    idx_reject = np.where(labels == BEAT_Q)
    features = np.delete(features,idx_reject,axis=0)
    labels = np.delete(labels,idx_reject,axis=0)
    classes = np.delete(classes,np.where(classes==BEAT_Q))
    
  # Merge F
  if merge_F :
    labels[labels == BEAT_F] = BEAT_V
    classes = np.delete(classes,np.where(classes==BEAT_F))

  # Remove V
  if remove_V :
    idx_reject = np.where(labels == BEAT_V)
    features = np.delete(features,idx_reject,axis=0)
    labels = np.delete(labels,idx_reject,axis=0)
    classes = np.delete(classes,np.where(classes==BEAT_V))

  # Merge S
  if merge_S :
    labels[labels == BEAT_S] = BEAT_N
    classes = np.delete(classes,np.where(classes==BEAT_S))
    
  N_beat_train,N_feat = np.shape(features)

  if verbose :
    print("####################################################")
    print("Data set information")
    print("")
    print("Features : {:d}".format(N_feat))
    print("")
    print("Training set : {:d}".format(N_beat_train))
    print(" - N\t\t\t : %d" % (np.sum(labels==BEAT_N)))
    print(" - S\t\t\t : %d" % (np.sum(labels==BEAT_S)))
    print(" - V\t\t\t : %d" % (np.sum(labels==BEAT_V)))
    print(" - F\t\t\t : %d" % (np.sum(labels==BEAT_F)))
    print(" - Q\t\t\t : %d" % (np.sum(labels==BEAT_Q)))
    print("####################################################")
    
  return labels, features