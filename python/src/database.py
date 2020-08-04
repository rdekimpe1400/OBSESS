# ECG database for OBSESS
# Wrapper for wfdb functions for accessing PhysioNet MIT-BIH database
#
# R. Dekimpe
# Last update: 06.2020

from os import path
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import time

from src import annotations 
from src.defines import *

# Fetch record in local MIT-BIH (must be placed in OBSESS/python/mitdb)
# Returns:
# - ECG = [2xN_db] list containing samples of ECG signal for each lead, in mV
# - time = [N_db] list of corresponding time instants
# - annotations = dictionary with annotation information ('label' for the N/S/V/F/Q/other annotation [cfr defines], 'time' for the time location [seconds])
def openRecord(record_ID = 100, Fs_resample = None, N_db = None, showFigures = False, verbose = False, stopwatch = False):
  
  t_start = time.time()
  
  # Full record path
  database = 'mitdb'
  record_path = database+'/'+str(record_ID)
  if not path.exists(record_path+'.hea'):
    print('Record {} does not exist. Load MIT-BIH database in "python/mitdb" directory.'.format(record_path))
    return
  
  # Open record
  ecg_db, fields = wfdb.rdsamp(record_path, sampto=N_db)
  
  t_open = time.time()
  
  # Reorder signals to have MLII lead first
  lead_MLII_idx = fields.get("sig_name").index('MLII')
  if lead_MLII_idx==0 :
    ECG = ecg_db
  else :
    ECG = ecg_db[:,[1,0]]
  
  # Create time vector
  Fs_db = fields.get("fs")
  N_db = len(ECG)
  t_max = (N_db-1)/Fs_db
  time_db = np.linspace(0,t_max,N_db)
  lead0 = fields.get("sig_name")[lead_MLII_idx]
  lead1 = fields.get("sig_name")[1-lead_MLII_idx]
  
  # Transpose ECG signal matrix (axis(0) = leads, axis(1) = time)
  ECG = np.transpose(ECG)

  # Read annotations
  ann_db = wfdb.rdann(record_path, 'atr', sampto=N_db,summarize_labels=True)
  
  # Map beat types to standard classes
  ann_label = annotations.annotMap(ann_db.symbol)
  
  # Map annotation samples to time value
  ann_time = ann_db.sample/Fs_db
  
  # Remove not beat annotations
  ann_time = ann_time[ann_label!=NOT_BEAT]
  ann_label = ann_label[ann_label!=NOT_BEAT]
  
  # Concatenate annotation data in dict structure
  annot = {'label': ann_label,'time':ann_time} 
  
  t_process = time.time()
  
  # Plot signals
  if showFigures:
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle("ECG signal #{} from database".format(record_ID))
    axs[0].plot(time_db,ECG[0])
    axs[0].set(ylabel="ECG (lead {}) [mV]".format(lead0))
    axs[1].plot(time_db,ECG[1])
    axs[1].set(ylabel="ECG (lead {}) [mV]".format(lead1))
    axs[2].plot(annot['time'],annot['label'],'r.')
    axs[2].set(ylabel="Annotations",xlabel="Time[s]",ylim=[0.5,5.5])
    axs[2].set_yticks(np.arange(1,6))
    axs[2].set_yticklabels(['N','S','V','F','Q'])
  
  t_plot = time.time()
  
  if Fs_resample is None:
    ECG_out = ECG
    time_out = time_db
  else:
    n_resample = int( Fs_resample / Fs_db * len(ECG[0]))
    time_out = np.arange(0,n_resample)/Fs_resample
    tck0 = interpolate.splrep(time_db,ECG[0], s=0)
    tck1 = interpolate.splrep(time_db,ECG[1], s=0)
    ECG_out = np.zeros((2,n_resample))
    ECG_out[0] = interpolate.splev(time_out, tck0, der=0)
    ECG_out[1] = interpolate.splev(time_out, tck1, der=0)
  
  t_resample = time.time()
  
  # Print information
  if verbose :
    print("####################################################")
    print("Waveform information for record %d" %(record_ID))
    print("")
    print("Lead 0\t\t\t : %s" %(lead0))
    print("Lead 1\t\t\t : %s" %(lead1))
    print("Sampling frequency\t : %d Hz" %(Fs_db))
    print("Signal length\t\t : %3.1f s" %(t_max))
    print("")
    print("Beat types distribution")
    print(" - N\t\t\t : %d" % (np.sum(ann_label==BEAT_N)))
    print(" - S\t\t\t : %d" % (np.sum(ann_label==BEAT_S)))
    print(" - V\t\t\t : %d" % (np.sum(ann_label==BEAT_V)))
    print(" - F\t\t\t : %d" % (np.sum(ann_label==BEAT_F)))
    print(" - Q\t\t\t : %d" % (np.sum(ann_label==BEAT_Q)))
    print("####################################################")
  
  if stopwatch :
    print("####################################################")
    print("Database access timing report" )
    print("")
    print("- Open database     {:4.3f}s".format(t_open-t_start))
    print("- Data pre-process  {:4.3f}s".format(t_process-t_open))
    print("- Plot signals      {:4.3f}s".format(t_plot-t_process))
    print("- Resample signals  {:4.3f}s".format(t_resample-t_plot))
    print("")
    print("- Total             {:4.3f}s".format(t_resample-t_start))
    print("####################################################")
  
  return ECG_out,time_out,annot
