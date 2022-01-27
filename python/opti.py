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

#######################
# Optimization parameters
# Ranges
IA_noise_range = [0.6e-6, 30e-6]
#ADC_resolution_range = [9, 15]
ADC_resolution_range = [7, 14]
SVM_pruning_range = [-1.5, 3]

# Constraints
alpha = 0.5
th = np.array([3.65750459e-02,2.30333372e-01,9.6127057408,25.7326009026,17.0771761517,65.2602614876]) * (1+alpha)

# Data log file
full_file_name = 'full_log.log'
exec_file_name = 'exec_log.log'
data_file_name = 'data_log.log'
opti_file_name = 'opti_log.log'
out_dir = './temp/'
full_log_file = out_dir+full_file_name
exec_log_file = out_dir+exec_file_name
data_log_file = out_dir+data_file_name
opti_log_file = out_dir+opti_file_name

# Parameters
params = default.default_parameters()

# Global variables
iteration = 0
accumulator_perf = list()
accumulator_constr = list()
accumulator_state = list()

#######################

@contextmanager
def nullify_stdout():
    stdout = sys.stdout
    redirect = open(full_log_file, "a")
    try:
        sys.stdout = redirect
        yield
    finally:
        sys.stdout = stdout

def norm_param(x):
  #IA
  noise_IA = np.power(10.0,np.log10(IA_noise_range[1])-x[0]*(np.log10(IA_noise_range[1])-np.log10(IA_noise_range[0])))
  params["IA_thermal_noise"] = noise_IA/10
  
  #ADC
  res_ADC = ADC_resolution_range[0] + x[1]*(ADC_resolution_range[1]-ADC_resolution_range[0])
  VCO_freq = np.power(2.0,res_ADC)*params['ADC_Fs']
  params["ADC_VCO_f0"] = VCO_freq
  
  #SVM
  SVM_pruning = SVM_pruning_range[1] - x[2]*(SVM_pruning_range[1]-SVM_pruning_range[0])
  params['SVM_pruning_D'] = SVM_pruning
  
def power(x):
  _, power_perf = execute_framework(x)
  return np.log10(power_perf)
  
def inference(x):
  inference_perf, _ = execute_framework(x)
  return inference_perf

def constraint_0(x):
  val = constraint(x,0)
  return val

def constraint_1(x):
  val = constraint(x,1)
  return val

def constraint_2(x):
  val = constraint(x,2)
  return val

def constraint_3(x):
  val = constraint(x,3)
  return val

def constraint_4(x):
  val = constraint(x,4)
  return val

def constraint_5(x):
  val = constraint(x,5)
  return val

def constraint(x,i):
  log_exec("Constraint ({}) evaluation on point [{:.10f} {:.10f} {:.10f}]...\n".format(i,x[0],x[1],x[2]))
  accumulator_constr.append([i,x])
  dump(accumulator_constr,out_dir+'accumulator_constr.sav')
  
  perf = inference(x)
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
  f.write("[{:.10f} {:.10f} {:.10f}] ".format(x[0],x[1],x[2]))
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
        if p[0]=="[{:.10f} {:.10f} {:.10f}]".format(x[0],x[1],x[2]):
          return np.array(d[0][2:].split(" ")[:-1]).astype(np.float), np.array(d[0][2:].split(" ")[-1]).astype(np.float)
  return None, None

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
  f.write("[{:.10f} {:.10f} {:.10f}] ".format(x[0],x[1],x[2]))
  for i in range(0,len(inference_perf)):
    f.write("{:.10f} ".format(inference_perf[i]))
  f.write("{:.10f}".format(power_perf))
  f.write("\n")
  f.close()

def execute_framework(x):
  log_exec("Fetching data for point [{:.10f} {:.10f} {:.10f}]\n".format(x[0],x[1],x[2]))
  inference_perf, power_perf = fetch_data(x)
  if inference_perf is not None:
    log_exec('Previous data found : Power [{:f}] - Inference [{:s}]\n'.format(power_perf, ' '.join([str(elem) for elem in inference_perf])))
    return inference_perf, power_perf
  else:
    norm_param(x)
    log_exec('Previous data not found. Executing framework with {:s}...\n'.format(str(params)))
    with nullify_stdout():
      train_inference_framework(params)
      inference_perf = run_inference_framework()
      power_perf = run_power_framework(params)
    dump_data(x,inference_perf, power_perf)
    log_exec('Data obtained : Power [{:f}] - Inference [{:s}]\n'.format(power_perf, ' '.join([str(elem) for elem in inference_perf])))
    return inference_perf, power_perf
  return inference_perf, power_perf

def run_inference_framework():
  global params
  dump(params,out_dir+'params.sav')
  subprocess.run("python framework.py -e {}".format(out_dir), shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
  det_sen,det_ppv,cm,params = load(out_dir+'perf.sav')
  _,_,_,class_perf=evaluation.statClass(cm)
  perf = np.zeros(6)
  perf[0] = 100-np.average(det_sen)
  perf[1] = 100-np.average(det_ppv)
  perf[2] = 100-class_perf[0]
  perf[3] = 100-class_perf[1]
  perf[4] = 100-class_perf[2]
  perf[5] = 100-class_perf[3]
  return perf

def train_inference_framework(params):
  dump(params,out_dir+'params.sav')
  subprocess.run("python framework.py -z {}".format(out_dir), shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)

def run_power_framework(params):
  log_exec('Power evaluation with param {:s}...\n'.format(str(params)))
  dump(params,out_dir+'params.sav')
  subprocess.run("python framework.py -p {}".format(out_dir), shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
  power_perf = load(out_dir+'power.sav')
  return power_perf

def run_optimization(update_output_dir):
  global out_dir
  out_dir = update_output_dir
  try:
    os.mkdir(out_dir)
  except FileExistsError:
    print(out_dir ,  " already exists")
  global full_log_file
  global exec_log_file
  global data_log_file
  global opti_log_file
  full_log_file = out_dir+full_file_name
  exec_log_file = out_dir+exec_file_name
  data_log_file = out_dir+data_file_name
  opti_log_file = out_dir+opti_file_name
  
  f = open(exec_log_file, "w")
  f.close()
  f = open(data_log_file, "w")
  f.close()
  f = open(full_log_file, "w")
  f.close()
  f = open(opti_log_file, "w")
  f.close()
  
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
  constraints.append({"fun": constraint_2, "type": "ineq"})
  constraints.append({"fun": constraint_3, "type": "ineq"})
  constraints.append({"fun": constraint_4, "type": "ineq"})
  constraints.append({"fun": constraint_5, "type": "ineq"})
  
  start = [0.79+0.02*random.random(),0.79+0.02*random.random(),0.79+0.02*random.random()]
  save_state(start)
  
  bounds = list()
  bounds.append((0.01,0.99))
  bounds.append((0.01,0.99))
  bounds.append((0.01,0.99))
  
  res=optimize.minimize(power, np.array(start), method="SLSQP",
                     constraints=constraints, bounds=bounds, callback = save_state, options={'disp': True ,'eps' : 1e-2, 'ftol' : 1e-3, 'maxiter' : 30})
                     
  print('Optimization done')
  
  print(res)
  
  dump(res,out_dir+'result.sav')
