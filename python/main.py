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
#from src import evaluation
import time

def run_framework_single_record(record_ID = 100): 
  
  # Set parameters
  params = {}
  
  # Get signals and annotations from database
  ECG, time, annotations = database.openRecord(record_ID = record_ID, showFigures = False, verbose = True)
  
  # Run smart sensor model
  power, detections = model.systemModel(ECG,time)
  
  # Comparison of output with annotations
  
  
def print_header():
  print('--------------------------------------')
  print('|                                    |')
  print('|              OBSESS                |')
  print('|                                    |')
  print('--------------------------------------')

if __name__ == "__main__":
  print_header()
  start_time = time.time()
  run_framework_single_record()
  end_time = time.time()
  print('Total execution time: {:5.3f} seconds'.format(end_time-start_time))
  plt.show()