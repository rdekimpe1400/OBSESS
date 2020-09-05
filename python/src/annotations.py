
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










