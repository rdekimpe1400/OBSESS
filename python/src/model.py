# Smart ECG sensor model
#
# R. Dekimpe
# Last update: 06.2020

from src import AFEmodel
from src import DBEmodel

#
def systemModel(ECG,time, params = {}, verbose = False, showFigures = False):
  # Run AFE model
  # Input is the signal from the database with time vector
  # Output is the digitized signal with time vector after AFE non-idealities
  ECG_dig0, time_dig = AFEmodel.analogFrontEndModel(ECG[0],time, showFigures = showFigures, stopwatch=True)
  ECG_dig1, _ = AFEmodel.analogFrontEndModel(ECG[1],time, showFigures = showFigures)
  
  ECG_dig = [ECG_dig0,ECG_dig1]
  
  # Run DBE model
  # Input is the digitized signal with time vector
  # Output is the label of detected beats with corresponding time 
  labels, time_det, features = DBEmodel.digitalBackEndModel(ECG_dig,time_dig, showFigures = showFigures)
  
  detect = {'label': labels,'time':time_det} 
  
  power = 0
  return power, detect, features