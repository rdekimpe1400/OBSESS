
from src.defines import *
import numpy as np

def annotMap(wfdbAnnot):
  N = len(wfdbAnnot)
  refAnnot = np.zeros(N)
  switch = {
    'N' : BEAT_N,
    'L' : BEAT_N,
    'R' : BEAT_N,
    'B' : BEAT_N,
    'A' : BEAT_S,
    'a' : BEAT_S,
    'J' : BEAT_S,
    'S' : BEAT_S,
    'V' : BEAT_V,
    'r' : BEAT_V,
    'F' : BEAT_F,
    'e' : BEAT_N,
    'j' : BEAT_N,
    'n' : BEAT_N,
    'E' : BEAT_V,
    '/' : BEAT_Q,
    'f' : BEAT_Q,
    'Q' : BEAT_Q,
    '?' : NOT_BEAT,
    '[' : NOT_BEAT,
    '!' : NOT_BEAT,
    ']' : NOT_BEAT,
    'x' : NOT_BEAT,
    '(' : NOT_BEAT,
    ')' : NOT_BEAT,
    'p' : NOT_BEAT,
    't' : NOT_BEAT,
    'u' : NOT_BEAT,
    '`' : NOT_BEAT,
    "'" : NOT_BEAT,
    '^' : NOT_BEAT,
    '|' : NOT_BEAT,
    '~' : NOT_BEAT,
    '+' : NOT_BEAT,
    's' : NOT_BEAT,
    'T' : NOT_BEAT,
    '*' : NOT_BEAT,
    'D' : NOT_BEAT,
    '=' : NOT_BEAT,
    '"' : NOT_BEAT,
    '@' : NOT_BEAT
  }
  for i in range(0,N):
    refAnnot[i] = switch.get(wfdbAnnot[i])
    if(np.isnan(refAnnot[i])):
      print('Error annotation {:c} unknown'.format(wfdb[i]))
  return refAnnot

def mapNoise(symbol, sample, subtype):
  noise_samples = sample[np.array(symbol)=='~']
  sub = np.bitwise_and(subtype[np.array(symbol)=='~'],1)
  noise = []
  state = 0
  for i in range(0,len(sub)) :
    if sub[i]==(not state) :
      noise.append(noise_samples[i])
      state = not state
  if state :
    noise.append(100000000)
  if len(noise)%2 != 0 :
    print('!!!!!!!!!!!!!!!!!!!!!!! Noise annotations are not paired !!!!!!!!!!!!!!!!!!!!')
  noise = np.array(noise).reshape((-1,2))
  return noise

def mapVFIB(symbol, sample):
  vfib_start = sample[np.array(symbol)=='[']
  vfib_stop = sample[np.array(symbol)==']']
  vfib = [vfib_start, vfib_stop]
  return np.array(vfib).transpose()

def mapAFIB(symbol, sample, aux_note):
  afib_samples = sample[np.array(symbol)=='+']
  aux_note = np.array(aux_note)
  start_stop = np.array(aux_note[np.array(symbol)=='+'])=='(AFIB\x00'
  afib = []
  state = 0
  for i in range(0,len(start_stop)) :
    if start_stop[i]==(not state) :
      afib.append(afib_samples[i])
      state = not state
  if state :
    afib.append(100000000)
  if len(afib)%2 != 0 :
    print('!!!!!!!!!!!!!!!!!!!!!!! AFIB annotations are not paired !!!!!!!!!!!!!!!!!!!!')
  afib = np.array(afib).reshape((-1,2))
  return afib

def matchAnnot(refAnnot, detAnnot, Fs=200):
  
  maxTol = 0.2*Fs #max 200ms of tolerance for detection
  
  # Match detected beats to reference annotations
  N_det = len(detAnnot)
  matched = np.zeros(N_det).astype(bool)
  idx = np.zeros(N_det).astype(int)
  for i in range(0,N_det):
    diff = np.absolute(refAnnot-detAnnot[i])
    min_idx = np.argmin(diff)
    min_diff = diff[min_idx]
    if min_diff<maxTol :
      idx[i] = min_idx
      matched[i] = 1
  
  # Find undetected beats
  N_ref = len(refAnnot)
  count = np.zeros(N_ref);
  for i in range(0,N_det):
    if matched[i] :
      count[idx[i]]=count[idx[i]]+1
  missed = count==0
  
  # Remove missed beats during initialization
  init = np.logical_and((refAnnot < 8*Fs),missed);
  missed[init] = 0;
  
  # Remove last beat of record if missed
  if missed[-1] :
    missed[-1] = 0
    init[-1] = 1
  
  return matched, idx, missed, init

def statAnnot(matched, missed, init, ref_rejected, det_rejected,verbose = 1, record=0):
  
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
    print("Detection statistics for record %d\n" % (record))
    print("Total beat annotations \t\t: %d (+ %d in noise segments)" % (N_ann, ref_rejected))
    print("Total beat detections \t\t: %d (+ %d in noise segments)" % (N_det, det_rejected))
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









