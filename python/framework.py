#####################################################
#                                                   #
#   OBSESS evaluation and optimization framework    #
#                                                   #
#####################################################
#
# R. Dekimpe (UCLouvain)
# Last update: 06.2020
#

import os
import sys
import getopt
import matplotlib.pyplot as plt
import numpy as np

from src import database
from src import inferenceModel
from src import powerModel
from src import evaluation
from src import data_IO
from src import default
from src.SVM import SVMtrain
import time

from joblib import dump, load

def run_framework_single_record(record_ID = 100, params = None, save_features=True, save_signal=False, append_features=False, ID=0, run_name='run_single', showFigures = False): 
  
  # Set parameters
  if params is None:
    params = default.default_parameters()
  
  # Get signals and annotations from database
  ECG, time, annotations = database.openRecord(record_ID = record_ID, params=params, N_db = None, showFigures = showFigures, verbose = True, stopwatch=True)
  print(params)
  # Save features to file
  if save_features:
    params["save_feature"] = True
    params["subset"] = ID
    params["feature_file"] = 'output/features_'+run_name+'.dat'
    if not append_features:
      open(params["feature_file"], 'w').close()
    params["feature_file"] = 'output/features_'+run_name+'.dat'
  
  # Run smart sensor model
  ECG_dig, detections, features = inferenceModel.inferenceModel(ECG,time,annotations, params=params, showFigures = showFigures)
  
  # Comparison of output with annotations
  det_sensitivity, det_PPV, matched_labels, confmat = evaluation.compareAnnotations(annotations, detections, time, ECG,showFigures = showFigures)
  
  #if save_features:
  #  data_IO.save_features(detections, features,matched_labels,subset=ID,file_name= 'output/features_'+run_name+'.dat',append=append_features)
  
  # Save signal to file
  if save_signal:
    data_IO.save_signal(ECG_dig,file_name= 'output/data_'+run_name+'.dat')
  print(params)
  return det_sensitivity, det_PPV, matched_labels, confmat, params

def run_framework_all_records(params = None, save_features=True, run_set = "both"): 
  
  sets = []
  
  if run_set=="train" or run_set=="both":
    sets.append(("train",[101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]))
  if run_set=="test" or run_set=="both":
    sets.append(("test",[100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]))
  
  for set in sets :
    sen = []
    ppv = []
    cm = np.zeros((6,6)).astype(int)
    set_name = set[0]
    print("Running set {:s}...".format(set_name))
    record_list = set[1]
    print('Record list : [{}]'.format(', '.join(map(str,record_list))))
    for count,signalID in enumerate(record_list) :
      print('Record {}'.format(signalID),flush=True)
      det_sensitivity, det_PPV, matched_labels, confmat,_=run_framework_single_record(params = params, record_ID = signalID, ID=count, save_features=save_features, run_name='run_all_'+set_name,append_features=(count>0))
      sen = sen+[det_sensitivity]
      ppv = ppv+[det_PPV]
      cm = cm + confmat
    _,j,_,_=evaluation.statClass(cm)
    print(np.average(sen))
    print(np.average(ppv))
    print(cm)
    print(j)
  return sen,ppv,cm,j

def embedded_run(path=None):
  params = load(path+'params.sav')
  print(params)
  print('Evaluating inference for {:s}...\n'.format(str(params)),flush=True)
  det_sen,det_ppv,cm,_ = run_framework_all_records(params = params,save_features=False,run_set="test")
  print(params)
  dump([det_sen,det_ppv,cm,params],path+'perf.sav')

def embedded_train(path=None):
  params = load(path+'params.sav')
  print('Training framework with {:s}...\n'.format(str(params)),flush=True)
  run_framework_all_records(params = params.copy(),save_features=True,run_set="train")
  SVMtrain.trainSVM(params = params)
  SVMtrain.updateModel(params = params)
  os.system("cd {} && rm -r build && python setupECG.py install".format(params['SVM_library']))

def embedded_reload(path=None):
  params = load(path+'params.sav')
  print('Reloading classifier with {:s}...\n'.format(str(params)),flush=True)
  SVMtrain.updateModel(params = params)
  os.system("cd {} && rm -r build && python setupECG.py install".format(params['SVM_library']))

def embedded_power(path=None):
  params = load(path+'params.sav')
  print('Evaluating power for {:s}...\n'.format(str(params)),flush=True)
  power_tot,_,_= powerModel.powerModel(params = params)
  dump(power_tot,path+'power.sav')

def trainModel(): 
  params = default.default_parameters()
  SVMtrain.trainSVM(params = params)
  return

def compile_DBE(): 
  params = default.default_parameters()
  SVMtrain.updateModel(params = params)
  os.system("cd {} && rm -r build && python setupECG.py install".format(params['SVM_library']))

def print_header():
  print('--------------------------------------')
  print('|                                    |')
  print('|              OBSESS                |')
  print('|                                    |')
  print('--------------------------------------')

def print_help():
  print('OBSESS system model')
  print('framework.py [-h] [-a] [-r <recordNumber>]')
  print("")
  print("Default: Run system model for record 100")
  print("")
  print("Arguments: ")
  print(" -h (--help)    >> Display help ")
  print(" -a (--all)     >> Run model for all records ")
  print(" -r (--record)  >> Run model for specified record only (overriden by -a) ")
  print(" -t (--train)   >> Run model for all records, training included ")
  print(" -c (--compile) >> Compile DBE C software ")
  print(" -f (--figures) >> Save figures ")
  print(" -v (--verbose) >> Increase verbosity ")
  print(" -s (--save)    >> Save features ")
  print(" -e (--embed_run)     >> Run framework from other python thread")
  print(" -z (--embed_train)   >> Train framework from other python thread")
  print(" -l (--embed_reload)   >> Reload classifier from other python thread")
  print(" -p (--embed_power)   >> Run power framework from other python thread")

if __name__ == "__main__":
  try:
    opts, args = getopt.getopt(sys.argv[1:],"ha:r:vtcfe:z:sp:l:",["help","all=","record=","verbose","train","compile","figures","embed_run=","embed_train=","save","embed_power=","embed_reload="])
  except getopt.GetoptError:
    print_help()
    sys.exit(2)
  # Default settings
  run = False
  compile = False
  single = True
  train = False
  figures = False
  verbose = False
  record = 100
  embed_run=False
  embed_train=False
  embed_power = False
  embed_reload = False
  set = 'test'
  save = False
  
  # Optional settings
  if(len(opts)>0):
    for opt, arg in opts:
      if opt in ("-h", "--help"):
        print_help()
        sys.exit()
      elif opt in ("-a", "--all"):
        run = True
        single = False
        set = arg
      elif opt in ("-r", "--record"):
        run = True
        record = int(arg)
      elif opt in ("-t", "--train"):
        single = False
        train = True
      elif opt in ("-c", "--compile"):
        compile = True
      elif opt in ("-f", "--figures"):
        figures = True
      elif opt in ("-v", "--verbose"):
        verbose = True
      elif opt in ("-s", "--save"):
        save = True
      elif opt in ("-e", "--embed_run"):
        embed_run=True
        path = arg
      elif opt in ("-z", "--embed_train"):
        embed_train=True
        path = arg
      elif opt in ("-p", "--embed_power"):
        embed_power=True
        path = arg
      elif opt in ("-l", "--embed_reload"):
        embed_reload=True
        path = arg
  else:
    run = True
  
  # Start
  print_header()
  start_time = time.time()
  if run:
    if single:
      run_framework_single_record(record_ID = record, showFigures = figures, save_signal=True)
    else:
      run_framework_all_records(save_features=save, run_set = set)
  if train:
    trainModel()
  if compile:
    compile_DBE()
  if embed_run:
    embedded_run(path=path)
  if embed_train:
    embedded_train(path=path)
  if embed_power:
    embedded_power(path=path)
  if embed_reload:
    embedded_reload(path=path)
      
  end_time = time.time()
  print('Total execution time: {:5.3f} seconds'.format(end_time-start_time))
  #plt.show()
