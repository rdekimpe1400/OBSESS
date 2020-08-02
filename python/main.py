#####################################################
#                                                   #
#   OBSESS evaluation and optimization framework    #
#                                                   #
#####################################################
#
# R. Dekimpe (UCLouvain)
# Last update: 06.2020
#

import matplotlib.pyplot as plt
import numpy as np

from src import database
from src import model
from src import evaluation
from src import data_IO
import time

def run_framework_single_record(record_ID = 100, save_features=True, append_features=False, ID=0, run_name='run_single'): 
  
  # Set parameters
  params = default_parameters()
  
  # Get signals and annotations from database
  ECG, time, annotations = database.openRecord(record_ID = record_ID, showFigures = False, verbose = True)
  
  # Run smart sensor model
  power, detections, features = model.systemModel(ECG,time)
  
  # Comparison of output with annotations
  det_sensitivity, det_PPV, matched_labels, confmat = evaluation.compareAnnotations(annotations, detections, time, ECG,showFigure = False)
  
  # Save features to file
  if save_features:
    data_IO.save_features(detections, features,matched_labels,subset=ID,file_name= 'output/features_'+run_name+'.dat',append=append_features)
    
  return det_sensitivity, det_PPV, matched_labels, confmat

def run_framework_all_records(save_features=True): 
  
  sets = []
  
  #sets.append(("train",[101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]))
  sets.append(("test",[100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]))
  #sets.append(("train",[112,114,115]))
  #sets.append(("test",[100,103]))
  
  sen = []
  ppv = []
  cm = np.zeros((6,6)).astype(int)
  for set in sets :
    set_name = set[0]
    print("Running set {:s}...".format(set_name))
    record_list = set[1]
    print('Record list : [{}]'.format(', '.join(map(str,record_list))))
    for count,signalID in enumerate(record_list) :
      det_sensitivity, det_PPV, matched_labels, confmat=run_framework_single_record(record_ID = signalID, ID=count, save_features=True, run_name='run_all_'+set_name,append_features=(count>0))
      if(set_name=="test"):
        sen = sen+[det_sensitivity]
        ppv = ppv+[det_PPV]
        cm = cm + confmat
  evaluation.statClass(cm)
  print(sen)
  print(ppv)
  print(cm)
  

def default_parameters():
  params = {}
  return params

def print_header():
  print('--------------------------------------')
  print('|                                    |')
  print('|              OBSESS                |')
  print('|                                    |')
  print('--------------------------------------')

if __name__ == "__main__":
  print_header()
  start_time = time.time()
  #run_framework_single_record(record_ID = 100)
  run_framework_all_records(save_features=False)
  end_time = time.time()
  print('Total execution time: {:5.3f} seconds'.format(end_time-start_time))
  plt.show()