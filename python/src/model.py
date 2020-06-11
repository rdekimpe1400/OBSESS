# Smart ECG sensor model
#
# R. Dekimpe
# Last update: 06.2020

from src import AFEmodel

#
def systemModel(ECG,time, verbose = False, showFigures = False):
  # Run AFE model
  # Input is the signal from the database with time vector
  # Output is the digitized signal with time vector after AFE non-idealities
  ECG_dig0, time_dig = AFEmodel.analogFrontEndModel(ECG[0],time, showFigures = True)
  ECG_dig1, _ = AFEmodel.analogFrontEndModel(ECG[1],time, time_dig = time_dig, showFigures = True)
  
  # Run DBE model
  # Input is the digitized signal with time vector
  # Output is the label of detected beats with corresponding time 
  
  power = 0
  detections = 0
  return power, detections