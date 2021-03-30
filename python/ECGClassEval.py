#!/usr/bin/env python3

from defines import *
import numpy as np
from sklearn import metrics

def evalStats(labels_true, labels_predict, verbose = True):
  
  confmat = metrics.confusion_matrix(labels_true, labels_predict, labels=[1,2,3,4,5])
  
  Nn = confmat[0,0]
  Ns = confmat[0,1]
  Nv = confmat[0,2]
  Nf = confmat[0,3]
  Nq = confmat[0,4]
  Sn = confmat[1,0]
  Ss = confmat[1,1]
  Sv = confmat[1,2]
  Sf = confmat[1,3]
  Sq = confmat[1,4]
  Vn = confmat[2,0]
  Vs = confmat[2,1]
  Vv = confmat[2,2]
  Vf = confmat[2,3]
  Vq = confmat[2,4]
  Fn = confmat[3,0]
  Fs = confmat[3,1]
  Fv = confmat[3,2]
  Ff = confmat[3,3]
  Fq = confmat[3,4]
  Qn = confmat[4,0]
  Qs = confmat[4,1]
  Qv = confmat[4,2]
  Qf = confmat[4,3]
  Qq = confmat[4,4]
  
  TNV = Nn + Ns + Nf + Nq + Sn + Ss + Sf + Sq + Fn + Fs + Ff + Fq + Qn + Qs + Qf + Qq
  FNV = Vn + Vs + Vf + Vq
  TPV = Vv
  FPV = Nv + Sv
  SenV = 100*TPV/(TPV+FNV)
  PPV = 100*TPV/(TPV+FPV)
  FPRV = 100*FPV/(TNV+FPV)
  AccV = 100*(TPV + TNV)/(TPV+TNV+FPV+FNV)
  
  TNS = Nn + Nv + Nf + Nq + Vn + Vv + Vf + Vq + Fn + Fv + Ff + Fq + Qn + Qv + Qf + Qq
  FNS = Sn + Sv + Sf + Sq
  TPS = Ss
  FPS = Ns + Vs + Fs
  SenS = 100*TPS/(TPS+FNS)
  PPS = 100*TPS/(TPS+FPS)
  FPRS = 100*FPS/(TNS+FPS)
  AccS = 100*(TPS + TNS)/(TPS+TNS+FPS+FNS)
  
  TN = Nn
  TPF = Ff
  TPQ = Qq
  Sp = 100*TN/(Nn+Ns+Nv+Nf+Nq)
  #SenF = 100*TPF/(Fn+Fs+Fv+Ff+Fq)
  #SenQ = 100*TPQ/(Qn+Qs+Qv+Qf+Qq)
  Acc = 100*(TN + TPS + TPV + TPF + TPQ)/(np.sum(np.sum(confmat)))
  
  if verbose :
    print("####################################################")
    print("Classification statistics")
    print("")
    print("Confusion matrix")
    print(" ------------------------------------------------")
    print(confmat)
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
    print("####################################################")

def evalStats_3(confmat, verbose = True):
  
  Nn = confmat[0,0]
  Ns = confmat[0,1]
  Nv = confmat[0,2]
  Sn = confmat[1,0]
  Ss = confmat[1,1]
  Sv = confmat[1,2]
  Vn = confmat[2,0]
  Vs = confmat[2,1]
  Vv = confmat[2,2]
  tot = Nn + Ns + Nv + Sn + Ss + Sv + Vn + Vs + Vv
  
  TNV = Nn + Ns + Sn + Ss
  FNV = Vn + Vs
  TPV = Vv
  FPV = Nv + Sv
  SenV = 100*TPV/(TPV+FNV)
  PPV = 100*TPV/(TPV+FPV+0.00001)
  FPRV = 100*FPV/(TNV+FPV)
  AccV = 100*(TPV + TNV)/(TPV+TNV+FPV+FNV)
  
  TNS = Nn + Nv + Vn + Vv
  FNS = Sn + Sv
  TPS = Ss
  FPS = Ns + Vs
  SenS = 100*TPS/(TPS+FNS)
  PPS = 100*TPS/(TPS+FPS+0.00001)
  FPRS = 100*FPS/(TNS+FPS)
  AccS = 100*(TPS + TNS)/(TPS+TNS+FPS+FNS)
  
  TN = Nn
  Sp = 100*TN/(Nn+Ns+Nv)
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
    print(confmat)
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

def evalStats_2(confmat, verbose = True):
  
  Nn = confmat[0,0]
  Nx = confmat[0,1]
  Xn = confmat[1,0]
  Xx = confmat[1,1]
  
  TN = Nn
  FN = Xn
  TP = Xx
  FP = Nx
  Sen = 100*TP/(TP+FN+0.00001)
  PPV = 100*TP/(TP+FP+0.00001)
  FPR = 100*FP/(TN+FP+0.00001)
  Acc = 100*(TP + TN)/(TP+TN+FP+FN+0.00001)
  
  if verbose :
    print("####################################################")
    print("Classification statistics")
    print("")
    print("Confusion matrix")
    print(" ------------------------------------------------")
    print(confmat)
    print(" ------------------------------------------------")
    print("")
    print("Sensitivity\t\t : {:6.2f} %".format(Sen))
    print("Positive pred. value\t : {:6.2f} %".format(PPV))
    print("False positive rate\t : {:6.2f} %".format(FPR))
    print("Accuracy\t\t : {:6.2f} %".format(Acc))
    print("")
    print("####################################################")
    
  return Sen,PPV,FPR


