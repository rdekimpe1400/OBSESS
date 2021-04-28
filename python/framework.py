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
from src import model
from src import evaluation
from src import data_IO
from src.SVM import SVMtrain
import time

def run_framework_single_record(record_ID = 100, params = None, save_features=True, save_signal=False, append_features=False, ID=0, run_name='run_single', showFigures = False): 
  
  # Set parameters
  if params is None:
    params = default_parameters()
  
  # Get signals and annotations from database
  ECG, time, annotations = database.openRecord(record_ID = record_ID, params=params, N_db = None, showFigures = showFigures, verbose = True, stopwatch=True)
  
  # Run smart sensor model
  ECG_dig, power, detections, features = model.systemModel(ECG,time,annotations, params=params, showFigures = showFigures)
  
  # Comparison of output with annotations
  det_sensitivity, det_PPV, matched_labels, confmat = evaluation.compareAnnotations(annotations, detections, time, ECG,showFigure = False)
  
  # Print power results
  evaluation.printPower(power, params=params)
  
  # Save features to file
  if save_features:
    data_IO.save_features(detections, features,matched_labels,subset=ID,file_name= 'output/features_'+run_name+'.dat',append=append_features)
  
  # Save signal to file
  if save_signal:
    data_IO.save_signal(ECG_dig,file_name= 'output/data_'+run_name+'.dat')
    
  return det_sensitivity, det_PPV, matched_labels, confmat, params

def run_framework_all_records(params = None, save_features=True, run_set = "both"): 
  
  sets = []
  
  if run_set=="train" or run_set=="both":
    sets.append(("train",[101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]))
  if run_set=="test" or run_set=="both":
    sets.append(("test",[100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]))
  
  sen = []
  ppv = []
  cm = np.zeros((6,6)).astype(int)
  for set in sets :
    set_name = set[0]
    print("Running set {:s}...".format(set_name))
    record_list = set[1]
    print('Record list : [{}]'.format(', '.join(map(str,record_list))))
    for count,signalID in enumerate(record_list) :
      det_sensitivity, det_PPV, matched_labels, confmat,_=run_framework_single_record(params = params, record_ID = signalID, ID=count, save_features=save_features, run_name='run_all_'+set_name,append_features=(count>0))
      if(set_name=="test"):
        sen = sen+[det_sensitivity]
        ppv = ppv+[det_PPV]
        cm = cm + confmat
  _,j,_,_=evaluation.statClass(cm)
  print(sen)
  print(ppv)
  print(cm)
  print(j)
  return sen,ppv,cm,j

def trainModel(): 
  params = default_parameters()
  SVMtrain.trainSVM()
  return

def compile_DBE(): 
  params = default_parameters()
  SVMtrain.updateModel(params = params)
  os.system("cd c_lib && rm -r build && python setupECG.py install")

def default_parameters():
  params = {"analog_resample":1000,
            "IA_TF_file":'./src/AFE/AFE_data/IA_dist.dat',
            "IA_DCout": 0.6,
            "IA_thermal_noise" : 0.095e-6,
            "IA_flicker_noise_corner" : None, #1,
            "ADC_Fs": 200,
            "ADC_VCO_TF_file":'./src/AFE/AFE_data/VCO_TF.dat',
            "ADC_thermal_noise" : 0.01e-3,
            "average_bpm": 100,
            "SVM_model_file":'./c_lib/c_src/svm_model.h',
            "SVM_feature_select":[0,1,43,33,55,39,57,38,54,58,56,37,40,60,52,61,36,48,32,50], #[0,1,18,8,30,14,32,13,29,33,31,12,15,35,28,36,11,23,37,25]
            "SVM_C":0.1,
            "SVM_gamma":0.1}
  return params

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
  print(" -v (--verbose) >> Increase verbosity ")

if __name__ == "__main__":
  try:
    opts, args = getopt.getopt(sys.argv[1:],"har:vtc",["help","all","record=","verbose","train","compile"])
  except getopt.GetoptError:
    print_help()
    sys.exit(2)
  # Default settings
  run = False
  compile = False
  single = True
  train = False
  verbose = False
  record = 100
  
  # Optional settings
  if(len(opts)>0):
    for opt, arg in opts:
      if opt in ("-h", "--help"):
        print_help()
        sys.exit()
      elif opt in ("-a", "--all"):
        run = True
        single = False
      elif opt in ("-r", "--record"):
        run = True
        record = int(arg)
      elif opt in ("-t", "--train"):
        single = False
        train = True
      elif opt in ("-c", "--compile"):
        compile = True
      elif opt in ("-v", "--verbose"):
        verbose = True
  else:
    run = True
  
  # Start
  print_header()
  start_time = time.time()
  if run:
    if single:
      run_framework_single_record(record_ID = record, showFigures = False, save_signal=True)
    else:
      run_framework_all_records(save_features=True)
  if train:
    trainModel()
  if compile:
    compile_DBE()
      
  end_time = time.time()
  print('Total execution time: {:5.3f} seconds'.format(end_time-start_time))
  #plt.show()
