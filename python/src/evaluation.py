# Evaluation of inference performance of smart sensor model
#
# R. Dekimpe
# Last update: 07.2020

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics

from src import annotations 
from src.defines import *

# Compare annotated and detected beats and labels
def compareAnnotations(annot, detect, time, ECG, showFigure = False):
  
  # Evaluate detections
  matched, match_idx, missed, init = matchAnnot(annot['time'], detect['time'])
  
  if showFigure :      
    fig, axs = plt.subplots(8,sharex=True)
    fig.suptitle("Detection evaluation")
    axs[0].plot(time,ECG[0])
    axs[0].set(ylabel="ECG #1 [mV]")
    axs[1].plot(time,ECG[1])
    axs[1].set(ylabel="ECG #2 [mV]")
    axs[2].plot(detect['time'],np.zeros_like(detect['time']),'.')
    axs[2].set(ylabel="Detected", ylim=[-0.5,0.5])
    axs[2].set_yticks([])
    axs[3].plot(annot['time'],np.zeros_like(annot['time']),'.')
    axs[3].set(ylabel="Annotated", ylim=[-0.5,0.5])
    axs[3].set_yticks([])
    axs[4].plot(np.array(detect['time'])[matched],np.zeros_like(np.array(detect['time'])[matched]),'.')
    axs[4].set(ylabel="Matched", ylim=[-0.5,0.5])
    axs[4].set_yticks([])
    axs[5].plot(np.array(detect['time'])[np.logical_not(matched)],np.zeros_like(np.array(detect['time'])[np.logical_not(matched)]),'.')
    axs[5].set(ylabel="Unmatched", ylim=[-0.5,0.5])
    axs[5].set_yticks([])
    axs[6].plot(np.array(annot['time'])[missed],np.zeros_like(np.array(annot['time'])[missed]),'.')
    axs[6].set(ylabel="Missed", ylim=[-0.5,0.5])
    axs[6].set_yticks([])
    axs[7].plot(np.array(annot['time'])[init],np.zeros_like(np.array(annot['time'])[init]),'.')
    axs[7].set(ylabel="Init", ylim=[-0.5,0.5])
    axs[7].set_yticks([])
  
  det_sensitivity, det_PPV=statDetect(matched, missed, init)
  
  # Match golden class annotations to detections 
  matched_labels = np.full(len(detect['time']),NOT_BEAT)
  matched_labels[matched] = annot['label'][match_idx[matched]]
  
  # Evaluate classification
  confmat = metrics.confusion_matrix(matched_labels, detect['label'], labels=[NOT_BEAT, BEAT_N, BEAT_S, BEAT_V, BEAT_F, BEAT_Q])
  statClass(confmat)
  
  return det_sensitivity, det_PPV, matched_labels, confmat


def matchAnnot(annot_time, detect_time):
  
  maxTol = 0.2 #max 200ms of tolerance for detection
  
  # Match detected beats to reference annotations
  N_det = len(detect_time)
  matched = np.zeros(N_det).astype(bool)
  idx = np.zeros(N_det).astype(int)
  for i in range(0,N_det):
    diff = np.absolute(annot_time-detect_time[i])
    min_idx = np.argmin(diff)
    min_diff = diff[min_idx]
    if min_diff<maxTol :
      idx[i] = min_idx
      matched[i] = 1
  
  # Find undetected beats
  N_ref = len(annot_time)
  count = np.zeros(N_ref);
  for i in range(0,N_det):
    if matched[i] :
      count[idx[i]]=count[idx[i]]+1
  missed = count==0
  
  # Remove missed beats during initialization
  init = np.logical_and((annot_time < 8),missed);
  missed[init] = 0;
  
  # Remove 2 last beat of record if missed
  if missed[-1] :
    missed[-1] = 0
    init[-1] = 1
  if missed[-2] :
    missed[-2] = 0
    init[-2] = 1
  
  return matched, idx, missed, init
  
def statDetect(matched, missed, init,verbose = True):
  
  N_init = np.sum(init)
  N_ann = len(missed)
  N_det = len(matched)
  
  TP = np.sum(matched)
  FP = np.sum(np.logical_not(matched))
  FN = np.sum(missed)
  
  sensitivity = 100*TP/(TP+FN)
  PPV = 100*TP/(TP+FP)
  
  if verbose:
    print("####################################################")
    print("Detection statistics\n")
    print("Total beat annotations \t\t: %d" % (N_ann))
    print("Total beat detections \t\t: %d" % (N_det))
    print(" - Missed during init \t\t: %d" % (N_init))
    print("")
    print(" ------------------------------------------------")
    print("| \t\t|\t\tAnn\t\t |")
    print("| \t\t|\tT\t\tF\t |")
    print(" ------------------------------------------------")
    print("| \tT\t|\t%d\t\t%d\t |" % (TP,FP))
    print("| Det\t\t|\t\t\t\t |")
    print("| \tF\t|\t%d\t\t--\t |" % (FN))
    print(" ------------------------------------------------")
    print("")
    print("Sensitivity\t\t : {:6.2f} %".format(sensitivity))
    print("Positive pred. value\t : {:6.2f} %".format(PPV))
    print("")
    print("####################################################")
  
  return sensitivity, PPV

def statClass(confmat,verbose = True):
  Nn = confmat[1,1]
  Ns = confmat[1,2]
  Nv = confmat[1,3]
  Sn = confmat[2,1]
  Ss = confmat[2,2]
  Sv = confmat[2,3]
  Vn = confmat[3,1]+confmat[4,1]
  Vs = confmat[3,2]+confmat[4,2]
  Vv = confmat[3,3]+confmat[4,3]
  
  tot = Nn + Ns + Nv + Sn + Ss + Sv + Vn + Vs + Vv
  
  TNV = Nn + Ns + Sn + Ss
  FNV = Vn + Vs
  TPV = Vv
  FPV = Nv + Sv
  SenV = 100*TPV/(TPV+FNV+0.00001)
  PPV = 100*TPV/(TPV+FPV+0.00001)
  FPRV = 100*FPV/(TNV+FPV+0.00001)
  AccV = 100*(TPV + TNV)/(TPV+TNV+FPV+FNV+0.00001)
  
  TNS = Nn + Nv + Vn + Vv
  FNS = Sn + Sv
  TPS = Ss
  FPS = Ns + Vs
  SenS = 100*TPS/(TPS+FNS+0.00001)
  PPS = 100*TPS/(TPS+FPS+0.00001)
  FPRS = 100*FPS/(TNS+FPS+0.00001)
  AccS = 100*(TPS + TNS)/(TPS+TNS+FPS+FNS+0.00001)
  
  TN = Nn
  Sp = 100*TN/(Nn+Ns+Nv+0.00001)
  Acc = 100*(TN + TPS + TPV)/(np.sum(np.sum(confmat)))
  
  j = (SenS + SenV + PPS + PPV)/4
  DN = (Nn + Ns + Nv)*(Nn + Sn + Vn)/tot
  DS = (Sn + Ss + Sv)*(Ns + Ss + Vs)/tot
  DV = (Vn + Vs + Vv)*(Nv + Sv + Vv)/tot
  k = 100*(((Nn + Vv + Ss) - (DN + DS + DV))/(tot - (DN + DS + DV)))
  jk = k/2 + j/2
  
  if verbose :
    print("####################################################")
    print("Classification statistics")
    print("")
    print("Confusion matrix")
    print(" ------------------------------------------------")
    print("{:<38}  Algo".format(""))
    print("")
    print("{:<15}{:<15} {:<8}{:<8}{:<8}".format("",""," N"," S"," V"))
    print("{:<30} ------------------------".format(""))
    print("{:<15}{:<15}|{:<8}{:<8}{:<8}".format("","Not beat",confmat[0,1],confmat[0,2],confmat[0,3]))
    print("{:<15}{:<15}|\033[94m{:<8}\033[0m{:<8}{:<8}".format("","N",confmat[1,1],confmat[1,2],confmat[1,3]))
    print("{:<15}{:<15}|{:<8}\033[94m{:<8}\033[0m{:<8}".format("Annot","S",confmat[2,1],confmat[2,2],confmat[2,3]))
    print("{:<15}{:<15}|{:<8}{:<8}\033[94m{:<8}\033[0m".format("","V",confmat[3,1],confmat[3,2],confmat[3,3]))
    print("{:<15}{:<15}|{:<8}{:<8}\033[94m{:<8}\033[0m".format("","F",confmat[4,1],confmat[4,2],confmat[4,3]))
    print(" ------------------------------------------------")
    print("")
    print("Ventricular arrhythmia")
    print("Sensitivity\t\t : {:6.2f} %".format(SenV))
    print("Positive pred. value\t : {:6.2f} %".format(PPV))
    print("False positive rate\t : {:6.2f} %".format(FPRV))
    print("Accuracy\t\t : {:6.2f} %".format(AccV))
    print("")
    print("Supraventricular arrhythmia")
    print("Sensitivity\t\t : {:6.2f} %".format(SenS))
    print("Positive pred. value\t : {:6.2f} %".format(PPS))
    print("False positive rate\t : {:6.2f} %".format(FPRS))
    print("Accuracy\t\t : {:6.2f} %".format(AccS))
    print("")
    print("Global accuracy\t\t : {:6.2f} %".format(Acc))
    print("j index\t\t\t : {:6.2f} %".format(j))
    print("kappa index\t\t : {:6.2f} %".format(k))
    print("Ijk index\t\t : {:6.2f} %".format(jk))
    print("####################################################",flush=True)
  
  return jk,j,k

def printPower(power, params = {}):  
  print("####################################################")
  print("Power statistics")
  print("")
  print("Average power\t\t : {:6.2f} uW".format(power["total"]*1e6))
  print(" - AFE \t\t\t : {:6.2f} uW".format(power["AFE"]*1e6))
  print(" - DBE \t\t\t : {:6.2f} uW".format(power["DBE"]*1e6))
  print("")
  print("Average energy per beat\t : {:6.2f} uJ".format(power["total"]*1e6*60/params["average_bpm"]))
  print("####################################################",flush=True)
  