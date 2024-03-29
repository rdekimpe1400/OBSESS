import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.interpolate import interp1d
import random
#import framework
import re
from src import evaluation
from contextlib import contextmanager
import subprocess
import sys
from joblib import dump, load
from importlib import reload
import os
from src import default
from src import data
import shutil
from changeFrequency import changeFrequency

#######################
# Optimization parameters
# Ranges
IA_noise_range = [0.6e-6, 30e-6]
#ADC_resolution_range = [9, 15]
ADC_resolution_range = [7, 14]
SVM_pruning_range = [-1.5, 3]
FREQ_range = [10, 200]
FEAT_range = [1,10]

ref_x = np.array([1.0,1.0,0.6,1.0,1.0])

# Constraints
alpha = 0.5
th = np.array([3.65750459e-02,2.30333372e-01,9.6127057408,25.7326009026,17.0771761517,65.2602614876]) * (1+alpha)

# Data log file
exec_file_name = 'exec_log.log'
data_file_name = 'data_log.log'
opti_file_name = 'opti_log.log'
fram_file_name = 'fram_log.log'
out_dir = './temp/'
exec_log_file = out_dir+exec_file_name
data_log_file = out_dir+data_file_name
opti_log_file = out_dir+opti_file_name
fram_log_file = out_dir+fram_file_name

# Parameters
params = default.default_parameters()

# Global variables
iteration = 0
accumulator_perf = list()
accumulator_constr = list()
accumulator_state = list()

#######################

def norm_param(x):
  #IA
  noise_IA = np.power(10.0,np.log10(IA_noise_range[1])-x[0]*(np.log10(IA_noise_range[1])-np.log10(IA_noise_range[0])))
  params["IA_thermal_noise"] = noise_IA/10
  
  #FREQ
  freq = FREQ_range[0] + x[3]*(FREQ_range[1]-FREQ_range[0])
  params['ADC_Fs'] = freq
  changeFrequency(freq)
  
  #ADC
  res_ADC = ADC_resolution_range[0] + x[1]*(ADC_resolution_range[1]-ADC_resolution_range[0])
  VCO_freq = np.power(2.0,res_ADC)*params['ADC_Fs']
  params["ADC_VCO_f0"] = VCO_freq
  
  #SVM
  SVM_pruning = SVM_pruning_range[1] - x[2]*(SVM_pruning_range[1]-SVM_pruning_range[0])
  params['SVM_pruning_D'] = SVM_pruning
  
  #FEAT
  feat = FEAT_range[0] + x[4]*(FEAT_range[1]-FEAT_range[0])
  params['SVM_feature_N_V'] = round(feat)
  params['SVM_feature_N_S'] = round(feat)
  
  dump(params,out_dir+'params.sav')
  f = open(fram_log_file, "a")
  subprocess.run("python framework.py -l {}".format(out_dir), shell=True,stdout=f,stderr=subprocess.STDOUT)
  f.close()
  
def power(x,train=True):
  log_exec("Power evaluation on point {}...\n".format(x))
  if len(x)<len(ref_x):
    x=np.append(np.array(x), ref_x[len(x):])
  _, power_perf = execute_framework(x,train=train)
  return np.log10(power_perf)
  
def inference(x,train=True):
  if len(x)<len(ref_x):
    x=np.append(np.array(x), ref_x[len(x):])
  inference_perf, _ = execute_framework(x,train=train)
  return inference_perf

def power_gradient(x):
  log_exec("Power gradient evaluation on point {}...\n".format(x))
  power_0 = power(x)
  train_vec = [False, False, True, False, True]
  steps = [-0.05,-0.05,-0.05,-0.05,-0.1]
  grad = np.zeros(len(x))
  for i in range(0,len(x)):
    delta = np.zeros(len(x))
    delta[i] = 1
    delta = delta*steps[i]
    if not train_vec[i]:
      load_cls(np.append(np.array(x), ref_x[len(x):]))
      norm_param(np.append(np.array(x), ref_x[len(x):]))
      dump(params,out_dir+'params.sav')
      f = open(fram_log_file, "a")
      subprocess.run("python framework.py -l {}".format(out_dir), shell=True,stdout=f,stderr=subprocess.STDOUT)
      f.close()
    power_delta = power(x+delta,train = train_vec[i])
    grad[i] = (power_delta-power_0)/steps[i]
  log_exec("Gradient value = [{}]\n".format(grad))
  return grad

def constraint_gradient(x,c):
  log_exec("Constraint ({:d}) gradient evaluation on point {}...\n".format(c,x))
  constr_0 = constraint(x,c)
  train_vec = [False, False, True, False, True]
  steps = [-0.05,-0.05,-0.05,-0.05,-0.1]
  grad = np.zeros(len(x))
  for i in range(0,len(x)):
    delta = np.zeros(len(x))
    delta[i] = 1
    delta = delta*steps[i]
    if not train_vec[i]:
      load_cls(np.append(np.array(x), ref_x[len(x):]))
      norm_param(np.append(np.array(x), ref_x[len(x):]))
      dump(params,out_dir+'params.sav')
      f = open(fram_log_file, "a")
      subprocess.run("python framework.py -l {}".format(out_dir), shell=True,stdout=f,stderr=subprocess.STDOUT)
      f.close()
    constr_delta = constraint(x+delta,c,train = train_vec[i])
    grad[i] = (constr_delta-constr_0)/steps[i]
  log_exec("Gradient value = [{}]\n".format(grad))
  return grad
  
def constraint(x,i,train=True):
  log_exec("Constraint ({}) evaluation on point {}...\n".format(i,x))
  accumulator_constr.append([i,x])
  dump(accumulator_constr,out_dir+'accumulator_constr.sav')
  
  perf = inference(x,train=train)
  g_val = np.log10(th[i]) - np.log10(perf[i])
  
  log_exec("Constraint_value = {:f}\n".format(g_val))
  return g_val

def log_exec(string):
  f = open(exec_log_file, "a")
  f.write(string)
  f.close()
  print(string)

def dump_data(x,inference_perf, power_perf):
  f = open(data_log_file, "a")
  f.write("[{:.10f} {:.10f} {:.10f} {:.10f} {:.10f}] ".format(x[0],x[1],x[2],x[3],x[4]))
  for i in range(0,len(inference_perf)):
    f.write("{:.10f} ".format(inference_perf[i]))
  f.write("{:.10f}".format(power_perf))
  f.write("\n")
  f.close()
  
def fetch_data(x):
  with open(data_log_file) as fp:
    for line in fp:
        p=re.findall("\[.*\]", line)
        d=re.findall("\].*", line)
        if p[0]=="[{:.10f} {:.10f} {:.10f} {:.10f} {:.10f}]".format(x[0],x[1],x[2],x[3],x[4]):
          return np.array(d[0][2:].split(" ")[:-1]).astype(np.float), np.array(d[0][2:].split(" ")[-1]).astype(np.float)
  return None, None

def save_cls(x):
  shutil.copyfile('scaler_S.sav',out_dir+'cls/scaler_S_{:.10f}_{:.10f}_{:.10f}_{:.10f}_{:.10f}.sav'.format(x[0],x[1],x[2],x[3],x[4]))
  shutil.copyfile('clf_S.sav',out_dir+'cls/clf_S_{:.10f}_{:.10f}_{:.10f}_{:.10f}_{:.10f}.sav'.format(x[0],x[1],x[2],x[3],x[4]))
  shutil.copyfile('params_S.sav',out_dir+'cls/params_S_{:.10f}_{:.10f}_{:.10f}_{:.10f}_{:.10f}.sav'.format(x[0],x[1],x[2],x[3],x[4]))
  shutil.copyfile('scaler_V.sav',out_dir+'cls/scaler_V_{:.10f}_{:.10f}_{:.10f}_{:.10f}_{:.10f}.sav'.format(x[0],x[1],x[2],x[3],x[4]))
  shutil.copyfile('clf_V.sav',out_dir+'cls/clf_V_{:.10f}_{:.10f}_{:.10f}_{:.10f}_{:.10f}.sav'.format(x[0],x[1],x[2],x[3],x[4]))
  shutil.copyfile('params_V.sav',out_dir+'cls/params_V_{:.10f}_{:.10f}_{:.10f}_{:.10f}_{:.10f}.sav'.format(x[0],x[1],x[2],x[3],x[4]))

def load_cls(x):
  shutil.copyfile(out_dir+'cls/scaler_S_{:.10f}_{:.10f}_{:.10f}_{:.10f}_{:.10f}.sav'.format(x[0],x[1],x[2],x[3],x[4]),'scaler_S.sav')
  shutil.copyfile(out_dir+'cls/clf_S_{:.10f}_{:.10f}_{:.10f}_{:.10f}_{:.10f}.sav'.format(x[0],x[1],x[2],x[3],x[4]),'clf_S.sav')
  shutil.copyfile(out_dir+'cls/params_S_{:.10f}_{:.10f}_{:.10f}_{:.10f}_{:.10f}.sav'.format(x[0],x[1],x[2],x[3],x[4]),'params_S.sav')
  shutil.copyfile(out_dir+'cls/scaler_V_{:.10f}_{:.10f}_{:.10f}_{:.10f}_{:.10f}.sav'.format(x[0],x[1],x[2],x[3],x[4]),'scaler_V.sav')
  shutil.copyfile(out_dir+'cls/clf_V_{:.10f}_{:.10f}_{:.10f}_{:.10f}_{:.10f}.sav'.format(x[0],x[1],x[2],x[3],x[4]),'clf_V.sav')
  shutil.copyfile(out_dir+'cls/params_V_{:.10f}_{:.10f}_{:.10f}_{:.10f}_{:.10f}.sav'.format(x[0],x[1],x[2],x[3],x[4]),'params_V.sav')

def save_state(x):
  global iteration
  iteration = iteration+1
  power_perf = power(x)
  inference_perf = inference(x)
  log_exec("--- ITERATION {:d} ---\n".format(iteration))
  accumulator_state.append([power_perf,inference_perf,x])
  dump(accumulator_state,out_dir+'accumulator_state.sav')
  f = open(opti_log_file, "a")
  f.write("{:4d} ".format(iteration))
  if len(x)<len(ref_x):
    x=np.append(np.array(x), ref_x[len(x):])
  f.write("[{:.10f} {:.10f} {:.10f} {:.10f} {:.10f}] ".format(x[0],x[1],x[2],x[3],x[4]))
  for i in range(0,len(inference_perf)):
    f.write("{:.10f} ".format(inference_perf[i]))
  f.write("{:.10f}".format(power_perf))
  f.write("\n")
  f.close()

def execute_framework(x,train=True):
  log_exec("Fetching data for point [{:.10f} {:.10f} {:.10f} {:.10f} {:.10f}]\n".format(x[0],x[1],x[2],x[3],x[4]))
  inference_perf, power_perf = fetch_data(x)
  if inference_perf is not None:
    log_exec('Previous data found : Power [{:f}] - Inference [{:s}]\n'.format(power_perf, ' '.join([str(elem) for elem in inference_perf])))
    return inference_perf, power_perf
  else:
    norm_param(x)
    log_exec('Previous data not found. Executing framework with {:s}...\n'.format(str(params)))
    if(train):
      train_inference_framework(params)
      save_cls(x)
    inference_perf = run_inference_framework()
    power_perf = run_power_framework(params)
    dump_data(x,inference_perf, power_perf)
    log_exec('Data obtained : Power [{:f}] - Inference [{:s}]\n'.format(power_perf, ' '.join([str(elem) for elem in inference_perf])))
    return inference_perf, power_perf
  return inference_perf, power_perf

def run_inference_framework():
  global params
  log_exec('Inference evaluation with param {:s}...\n'.format(str(params)))
  dump(params,out_dir+'params.sav')
  f = open(fram_log_file, "a")
  subprocess.run("python framework.py -e {}".format(out_dir), shell=True,stdout=f,stderr=subprocess.STDOUT)
  f.close()
  if os.path.isfile(out_dir+'perf.sav'):
    det_sen,det_ppv,cm,params = load(out_dir+'perf.sav')
    os.remove(out_dir+'perf.sav')
    _,_,_,class_perf=evaluation.statClass(cm)
    perf = np.zeros(6)
    perf[0] = 100-np.average(det_sen)
    perf[1] = 100-np.average(det_ppv)
    perf[2] = 100-class_perf[0]
    perf[3] = 100-class_perf[1]
    perf[4] = 100-class_perf[2]
    perf[5] = 100-class_perf[3]
  else:
    perf = np.zeros(6)
    perf[0] = 100
    perf[1] = 100
    perf[2] = 100
    perf[3] = 100
    perf[4] = 100
    perf[5] = 100
    
  return perf

def train_inference_framework(params):
  log_exec('Train framework with param {:s}...\n'.format(str(params)))
  dump(params,out_dir+'params.sav')
  f = open(fram_log_file, "a")
  subprocess.run("python framework.py -l {}".format(out_dir), shell=True,stdout=f,stderr=subprocess.STDOUT)
  subprocess.run("python framework.py -z {}".format(out_dir), shell=True,stdout=f,stderr=subprocess.STDOUT)
  f.close()

def run_power_framework(params):
  log_exec('Power evaluation with param {:s}...\n'.format(str(params)))
  dump(params,out_dir+'params.sav')
  f = open(fram_log_file, "a")
  subprocess.run("python framework.py -p {}".format(out_dir), shell=True,stdout=f,stderr=subprocess.STDOUT)
  f.close()
  power_perf = load(out_dir+'power.sav')
  return power_perf

def clear_output_dir(update_output_dir):
  global out_dir
  out_dir = update_output_dir
  try:
    os.mkdir(out_dir)
    os.mkdir(out_dir+'cls/')
  except FileExistsError:
    print(out_dir ,  " already exists")
  global exec_log_file
  global data_log_file
  global opti_log_file
  global fram_log_file
  exec_log_file = out_dir+exec_file_name
  data_log_file = out_dir+data_file_name
  opti_log_file = out_dir+opti_file_name
  fram_log_file = out_dir+fram_file_name
  
  f = open(exec_log_file, "w")
  f.close()
  f = open(data_log_file, "w")
  f.close()
  f = open(opti_log_file, "w")
  f.close()
  f = open(fram_log_file, "w")
  f.close()

def run_optimization(update_output_dir):
  clear_output_dir(update_output_dir)
  print('Start optimization')
  
  global accumulator_perf
  global accumulator_constr
  global accumulator_state
  accumulator_perf = list()
  accumulator_constr = list()
  accumulator_state = list()
  
  global iteration
  iteration = 0
  
  constraints = list()
  constraints.append({"fun": constraint, "type": "ineq", "jac": constraint_gradient, "args":([2])})
  constraints.append({"fun": constraint, "type": "ineq", "jac": constraint_gradient, "args":([3])})
  constraints.append({"fun": constraint, "type": "ineq", "jac": constraint_gradient, "args":([4])})
  constraints.append({"fun": constraint, "type": "ineq", "jac": constraint_gradient, "args":([5])})
  
  start = [1.0,1.0,0.6,1.0,1.0]
  save_state(start)
  
  bounds = list()
  for i in range(0,len(start)):
    bounds.append((0.05,1.0))
  
  res=optimize.minimize(power, np.array(start), method="SLSQP", jac=power_gradient,
                     constraints=constraints, bounds=bounds, callback = save_state, options={'disp': True ,'eps' : 5e-2, 'ftol' : 1e-3, 'maxiter' : 15, 'iprint': 100})
                     
  print('Optimization done')
  
  print(res)
  
  dump(res,out_dir+'result.sav')
  
  return(res)
