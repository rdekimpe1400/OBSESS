#####################################################
#                                                   #
#   OBSESS evaluation and optimization framework    #
#                                                   #
#####################################################
#
# R. Dekimpe (UCLouvain)
# Last update: 06.2020
#

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

def run_framework_single_record(record_ID = 100, save_features=True, append_features=False, ID=0, run_name='run_single'): 
  
  # Set parameters
  params = default_parameters()
  
  # Get signals and annotations from database
  ECG, time, annotations = database.openRecord(record_ID = record_ID, params=params, N_db = None, showFigures = False, verbose = True, stopwatch=True)
  
  # Run smart sensor model
  power, detections, features = model.systemModel(ECG,time,params=params)
  
  # Comparison of output with annotations
  det_sensitivity, det_PPV, matched_labels, confmat = evaluation.compareAnnotations(annotations, detections, time, ECG,showFigure = False)
  
  # Print power results
  evaluation.printPower(power, params=params)
  
  # Save features to file
  if save_features:
    data_IO.save_features(detections, features,matched_labels,subset=ID,file_name= 'output/features_'+run_name+'.dat',append=append_features)
    
  return det_sensitivity, det_PPV, matched_labels, confmat, params

def run_framework_all_records(save_features=True): 
  
  sets = []
  
  #sets.append(("train",[101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]))
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
      det_sensitivity, det_PPV, matched_labels, confmat,_=run_framework_single_record(record_ID = signalID, ID=count, save_features=True, run_name='run_all_'+set_name,append_features=(count>0))
      if(set_name=="test"):
        sen = sen+[det_sensitivity]
        ppv = ppv+[det_PPV]
        cm = cm + confmat
  evaluation.statClass(cm)
  print(sen)
  print(ppv)
  print(cm)

def run_framework_with_training(): 
  sets = []
  
  #sets.append(("train",[101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]))
  #sets.append(("test",[100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]))
  
  sets.append(("train",[101,106,108]))
  sets.append(("test",[100,103]))
  
  # Train
  set = sets[0]
  set_name = set[0]
  run_name = 'run_all_'+set_name
  print("Training with set {:s}...".format(set_name))
  record_list = set[1]
  print('Record list : [{}]'.format(', '.join(map(str,record_list))))
  for count,signalID in enumerate(record_list) :
    det_sensitivity, det_PPV, matched_labels, confmat,params=run_framework_single_record(record_ID = signalID, ID=count, save_features=True, run_name=run_name,append_features=(count>0))
  print("Update SVM model")
  _,features, labels,_=data_IO.read_features(file_name='output/features_'+run_name+'.dat')
  SVMtrain.updateModel(labels, features, params = params, verbose = True)
  
  # Test
  sen = []
  ppv = []
  cm = np.zeros((6,6)).astype(int)
  set = sets[1]
  set_name = set[0]
  print("Running set {:s}...".format(set_name))
  record_list = set[1]
  print('Record list : [{}]'.format(', '.join(map(str,record_list))))
  for count,signalID in enumerate(record_list) :
    det_sensitivity, det_PPV, matched_labels, confmat,_=run_framework_single_record(record_ID = signalID, ID=count, save_features=True, run_name='run_all_'+set_name,append_features=(count>0))
    sen = sen+[det_sensitivity]
    ppv = ppv+[det_PPV]
    cm = cm + confmat
  evaluation.statClass(cm)
  print(sen)
  print(ppv)
  print(cm)

def default_parameters():
  params = {"analog_resample":1000,
            "IA_TF_file":'./src/AFE/AFE_data/IA_dist.dat',
            "IA_DCout": 0.6,
            "IA_thermal_noise" : 0.095e-6,
            "IA_flicker_noise_corner" : 1,
            "ADC_Fs": 200,
            "ADC_VCO_TF_file":'./src/AFE/AFE_data/VCO_TF.dat',
            "ADC_thermal_noise" : 0.01e-3,
            "average_bpm": 100,
            "SVM_model_file":'./c_lib/c_src/svm_model_test.h',
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
  print('main.py [-h] [-a] [-r <recordNumber>]')
  print("")
  print("Default: Run system model for record 100")
  print("")
  print("Arguments: ")
  print(" -h (--help)    >> Display help ")
  print(" -a (--all)     >> Run model for all records ")
  print(" -r (--record)  >> Run model for specified record only (overriden by -a) ")
  print(" -t (--train)   >> Run model for all records, training included ")
  print(" -v (--verbose) >> Increase verbosity ")

if __name__ == "__main__":
  try:
    opts, args = getopt.getopt(sys.argv[1:],"har:vt",["help","all","record=","verbose","train"])
  except getopt.GetoptError:
    print_help()
    sys.exit(2)
  # Default settings
  single = True
  train = False
  verbose = False
  record = 100
  # Optional settings
  for opt, arg in opts:
    if opt in ("-h", "--help"):
      print_help()
      sys.exit()
    elif opt in ("-a", "--all"):
      single = False
    elif opt in ("-r", "--record"):
      record = int(arg)
    elif opt in ("-t", "--train"):
      single = False
      train = True
    elif opt in ("-v", "--verbose"):
      verbose = True
  
  # Start
  print_header()
  start_time = time.time()
  if single:
    run_framework_single_record(record_ID = record)
  elif train:
    run_framework_with_training()
  else:
    run_framework_all_records(save_features=False)
  end_time = time.time()
  print('Total execution time: {:5.3f} seconds'.format(end_time-start_time))
  plt.show()