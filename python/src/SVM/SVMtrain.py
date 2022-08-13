#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from sklearn.preprocessing import StandardScaler
from sklearn import svm

from sklearn import metrics

from joblib import Parallel, delayed
from joblib import dump, load

from src import evaluation
from src.SVM import FE
from src.defines import *
from src.SVM import ECGclassification
from src.SVM import ECGClassEval

def updateModel(params = {}, verbose = True):
  C_file = params['SVM_library']+params['SVM_model_file']
  
  CFileInit(file_name = C_file) 
  
  # Train model
  model_V = load('clf_V.sav')
  params_V = load('params_V.sav')
  model_S = load('clf_S.sav')
  params_S = load('params_S.sav')
  scaler = load('scaler_S.sav')
  
  FS_V = params_V['FS']
  FS_S = params_S['FS']
  
  if verbose:
    print("--------------------------------------------------------")
    print("Open SVM model (V)")
    print("\tClasses : {}".format(model_V.classes_))
    print("\tNumber of SV : {} (total: {})".format(model_V.n_support_, np.sum(model_V.n_support_)))
    sv_shape = np.shape(model_V.support_vectors_)
    print("\tNumber of features : {}".format(sv_shape[1]))
    print("\tSelected features : {}".format(FS_V))
    params = model_V.get_params()
    print("\tKernel : {}".format(params['kernel']))
    if params['kernel']=='rbf':
      print("\tC parameter : {}".format(params['C']))
      print("\tGamma parameter : {}".format(params['gamma']))
    else:
      print("\033[91mUnsupported kernel type!! \033[0m")
      sys.exit()
    print("--------------------------------------------------------")
    print("Open SVM model (S)")
    print("\tClasses : {}".format(model_S.classes_))
    print("\tNumber of SV : {} (total: {})".format(model_S.n_support_, np.sum(model_S.n_support_)))
    sv_shape = np.shape(model_S.support_vectors_)
    print("\tNumber of features : {}".format(sv_shape[1]))
    print("\tSelected features : {}".format(FS_S))
    params = model_S.get_params()
    print("\tKernel : {}".format(params['kernel']))
    if params['kernel']=='rbf':
      print("\tC parameter : {}".format(params['C']))
      print("\tGamma parameter : {}".format(params['gamma']))
    else:
      print("\033[91mUnsupported kernel type!! \033[0m")
      sys.exit()
    print("--------------------------------------------------------")
  
  SVMtranslate(model_S,FS_S, model_V,FS_V,scaler, file_name = C_file)
  
  CFileClose(file_name = C_file)
  return

def trainSVM(params = None):

  labels_train, features_train, subset_train, labels_test, features_test, subset_test = open_features(train_file = "output/features_run_all_train.dat", test_file = "output/features_run_all_test.dat")
  
  scaler = StandardScaler()
  features_train=scaler.fit_transform(features_train)
  features_test=scaler.transform(features_test)
  
  #V
  FE_preselect_V = np.array([0,1,2,3,4,5,6,7,21,22,23,24,25,26,27,28,29,30,31,32,33,34,118,119,120,121,122,123,124,144,145,146,147,148,149,150,151,152,156,157,158,159,160])
  features_train_V = features_train[:,FE_preselect_V]
  features_test_V = features_test[:,FE_preselect_V]

  FE_set_V = np.array([32, 34, 18, 23, 30, 21, 37, 27, 42, 40])
  SV_params_V = {'kernel':'rbf','C':0.1, 'gamma':0.1, 'degree':3, 'FS':FE_set_V[0:params['SVM_feature_N_V']], 'C_2':0.1, 'gamma_2':0.1, 'FS_2':None, 'pruning_D': params['SVM_pruning_D'], 'class_weight':'balanced','type':'VvX'}

  cm = ECGclassification.train_and_test(features_train_V, labels_train, features_test_V, labels_test, SV_params_V, prune=True, verbose=True, save_clf_file = 'clf_V.sav')
  j_prune= ECGClassEval.evalStats_3(cm)

  SV_params_V['FS'] = FE_preselect_V[SV_params_V['FS']]
  dump(scaler,'scaler_V.sav')
  dump(SV_params_V,'params_V.sav')
  
  #S
  FE_preselect_S = np.array([0,1,2,3,4,5,6,7,12,13,14,15,16,17,18,137,138,139,140,141,158,159,160,161,172,173,174])
  features_train_S = features_train[:,FE_preselect_S]
  features_test_S = features_test[:,FE_preselect_S]
  
  FE_set_S = np.array([0, 4, 5, 15, 1, 23, 20, 24, 21, 19])
  SV_params_S = {'kernel':'rbf','C':0.1, 'gamma':0.1, 'degree':3, 'FS':FE_set_S[0:params['SVM_feature_N_S']], 'C_2':0.1, 'gamma_2':0.1, 'FS_2':None, 'pruning_D': params['SVM_pruning_D'], 'class_weight':'balanced','type':'NvS'}
  
  cm = ECGclassification.train_and_test(features_train_S, labels_train, features_test_S, labels_test, SV_params_S, prune=True, verbose=True, save_clf_file = 'clf_S.sav')
  j_prune= ECGClassEval.evalStats_3(cm)
  
  SV_params_S['FS'] = FE_preselect_S[SV_params_S['FS']]
  dump(scaler,'scaler_S.sav')
  dump(SV_params_S,'params_S.sav')
  
  return


def CFileInit(file_name = "SVM.h"):
  file = open(file_name,'w')
  file.write('//**********************************************\n')
  file.write('//\n')
  file.write('//  SVM model data\n')
  file.write('//\n')
  file.write('//**********************************************\n\n')
  file.write('#ifndef SVM_MOD_H_\n')
  file.write('#define SVM_MOD_H_\n\n')
  
  return 1

def CFileClose(file_name = "SVM.h"):
  file = open(file_name,'a')
  file.write('\n#endif //SVM_MOD_H_\n')
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

def SVMtranslate(model_S,FS_S, model_V,FS_V,scaler, file_name = "svm_int.h"):
  sv_V = model_V.support_vectors_
  sv_shape_V = np.shape(sv_V)
  sv_params_V = model_V.get_params()
  n_sv_V = np.sum(model_V.n_support_)
  n_feat_V = sv_shape_V[1]
  file = open(file_name,'a')
  if sv_params_V['gamma']=='scale':
    gamma_V = 1/n_feat_V
  else:
    gamma_V = sv_params_V['gamma']
  
  sv_S = model_S.support_vectors_
  sv_shape_S = np.shape(sv_S)
  sv_params_S = model_S.get_params()
  n_sv_S = np.sum(model_S.n_support_)
  n_feat_S = sv_shape_S[1]
  file = open(file_name,'a')
  if sv_params_S['gamma']=='scale':
    gamma_S = 1/n_feat_S
  else:
    gamma_S = sv_params_S['gamma']
  
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
  #feature_max_range = np.maximum(np.max(np.abs(sv_S)),np.max(np.abs(sv_V)))
  #print("Feature max {}".format(feature_max_range))  
  feature_scale = (2**(feature_data-1))/feature_max_range
  feature_shift = np.floor(np.log2(feature_scale)).astype(int) #Avoid feature saturation
  
  feature_acc_shift = np.ceil(np.log2(np.maximum(n_feat_S,n_feat_S))).astype(int) #Avoid accumulate saturation
  
  scale_shift = 16
  
  coef_shift = 10
  
  kernel_acc_shift = 16 #Avoid accumulate saturation
  
  # Write model data
  file.write('// Feature pre-processing\n')
  file.write('const int feature_select_idx_V[{:d}] = {{'.format(n_feat_V))
  for i in range(0,n_feat_V):
    file.write("{:d}".format(FS_V[i]))
    if i<(n_feat_V-1):
      file.write(", ")
  file.write("};\n")
  file.write('const feature_data_t scale_mean_V[{:d}] = {{'.format(n_feat_V))
  for i in range(0,n_feat_V):
    file.write("{:d}".format(np.round(scaler.mean_[FS_V[i]]).astype(int)))
    if i<(n_feat_V-1):
      file.write(", ")
  file.write("};\n")
  file.write('const int32_t scale_std_V[{:d}] = {{'.format(n_feat_V))
  for i in range(0,n_feat_V):
    file.write("{:d}".format(np.round((2**(feature_shift+scale_shift))/np.sqrt(scaler.var_[FS_V[i]])).astype(int)))
    if i<(n_feat_V-1):
      file.write(", ")
  file.write("};\n")
  file.write('const int feature_select_idx_S[{:d}] = {{'.format(n_feat_S))
  for i in range(0,n_feat_S):
    file.write("{:d}".format(FS_S[i]))
    if i<(n_feat_S-1):
      file.write(", ")
  file.write("};\n")
  file.write('const feature_data_t scale_mean_S[{:d}] = {{'.format(n_feat_S))
  for i in range(0,n_feat_S):
    file.write("{:d}".format(np.round(scaler.mean_[FS_S[i]]).astype(int)))
    if i<(n_feat_S-1):
      file.write(", ")
  file.write("};\n")
  file.write('const int32_t scale_std_S[{:d}] = {{'.format(n_feat_S))
  for i in range(0,n_feat_S):
    file.write("{:d}".format(np.round((2**(feature_shift+scale_shift))/np.sqrt(scaler.var_[FS_S[i]])).astype(int)))
    if i<(n_feat_S-1):
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
  
  file.write('const int n_feat_V = {};\n'.format(n_feat_V))
  
  file.write('const int gam_inv_V = {};\n'.format(int(1/gamma_V)))
  
  file.write('const int n_feat_S = {};\n'.format(n_feat_S))
  
  file.write('const int gam_inv_S = {};\n'.format(int(1/gamma_S)))
  
  file.write('const int feature_shift = {};\n'.format(int(feature_shift)))
  
  file.write('const int feature_acc_shift = {};\n'.format(int(feature_acc_shift)))
  
  file.write('const int kernel_acc_shift = {};\n'.format(int(kernel_acc_shift)))
  
  file.write('const int n_sv_class_V[2] = {{{}, {}}};\n'.format(model_V.n_support_[0], model_V.n_support_[1]))
  
  file.write('const int n_sv_V = {};\n'.format(n_sv_V))
  
  file.write('const int start_sv_V[2] = {{{}, {}}};\n'.format(0, model_V.n_support_[0]))
  
  interp_int_V = np.round(model_V.intercept_*(2**(exp_shift+coef_shift-kernel_acc_shift))).astype(int)
  file.write('const decision_data_t rho_V = {};\n'.format(interp_int_V[0]))
  
  file.write('const decision_data_t sv_coef_V[{:d}] = {{'.format(n_sv_V))
  for i in range(0,n_sv_V):
    file.write("{}".format(int(model_V.dual_coef_[0,i]*(2**coef_shift))))
    if i<(n_sv_V-1):
      file.write(", ")
  file.write("};\n")
  
  file.write('const feature_data_t sv_V[{:d}][{:d}] = {{\n'.format(n_sv_V,n_feat_V))
  for i in range(0,n_sv_V):
    file.write("{")
    for j in range(0,n_feat_V):
      file.write("{}".format(int(sv_V[i,j]*(2**feature_shift))))
      if j<(n_feat_V-1):
        file.write(", ")
      elif i<(n_sv_V-1):
        file.write("},\n")
      else: 
        file.write("}\n")
  file.write("};\n")
  
  file.write('const int n_sv_class_S[2] = {{{}, {}}};\n'.format(model_S.n_support_[0], model_S.n_support_[1]))
  
  file.write('const int n_sv_S = {};\n'.format(n_sv_S))
  
  file.write('const int start_sv_S[2] = {{{}, {}}};\n'.format(0, model_S.n_support_[0]))
  
  interp_int_S = np.round(model_S.intercept_*(2**(exp_shift+coef_shift-kernel_acc_shift))).astype(int)
  file.write('const decision_data_t rho_S = {};\n'.format(interp_int_S[0]))
  
  file.write('const decision_data_t sv_coef_S[{:d}] = {{'.format(n_sv_S))
  for i in range(0,n_sv_S):
    file.write("{}".format(int(model_S.dual_coef_[0,i]*(2**coef_shift))))
    if i<(n_sv_S-1):
      file.write(", ")
  file.write("};\n")
  
  file.write('const feature_data_t sv_S[{:d}][{:d}] = {{\n'.format(n_sv_S,n_feat_S))
  for i in range(0,n_sv_S):
    file.write("{")
    for j in range(0,n_feat_S):
      file.write("{}".format(int(sv_S[i,j]*(2**feature_shift))))
      if j<(n_feat_S-1):
        file.write(", ")
      elif i<(n_sv_S-1):
        file.write("},\n")
      else: 
        file.write("}\n")
  file.write("};\n")
  
  
  return 1

def open_features(train_file = "output/train.dat", test_file = "output/test.dat", reject_Q=True,merge_F = True,remove_V = False,merge_S = False, reject_NOT = True, verbose = True ):

  subset_train, features_train, labels_train,names = FE.readFE(file_name = train_file,read_header=False)
  subset_test, features_test, labels_test,_ = FE.readFE(file_name = test_file,read_header=False)

  classes = np.array([BEAT_N,BEAT_S,BEAT_V,BEAT_F,BEAT_Q])

  # Reject non beats
  if reject_NOT :
    idx_reject = np.where(labels_train == NOT_BEAT)
    features_train = np.delete(features_train,idx_reject,axis=0)
    labels_train = np.delete(labels_train,idx_reject,axis=0)
    subset_train = np.delete(subset_train,idx_reject,axis=0)
    idx_reject = np.where(labels_test == NOT_BEAT)
    features_test = np.delete(features_test,idx_reject,axis=0)
    labels_test = np.delete(labels_test,idx_reject,axis=0)
    subset_test = np.delete(subset_test,idx_reject,axis=0)

  # Reject Q
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
  if merge_F :
    labels_train[labels_train == BEAT_F] = BEAT_V
    labels_test[labels_test == BEAT_F] = BEAT_V
    classes = np.delete(classes,np.where(classes==BEAT_F))

  # Remove V
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
  if merge_S :
    labels_train[labels_train == BEAT_S] = BEAT_V
    labels_test[labels_test == BEAT_S] = BEAT_V
    classes = np.delete(classes,np.where(classes==BEAT_S))
    
  N_beat_train,N_feat = np.shape(features_train)
  N_beat_test,_ = np.shape(features_test)
  N_rec_train = int(np.max(subset_train)+1)
  N_rec_test = int(np.max(subset_test)+1)

  if verbose :
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
    
  return labels_train, features_train, subset_train, labels_test, features_test, subset_test