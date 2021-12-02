# Smart ECG sensor model
#
# R. Dekimpe
# Last update: 06.2020

from src import data
import numpy as np
from scipy.interpolate import interp1d

def powerModel(params = {}):
  
  beat_freq = params["beat_freq"]
  
  ## AFE
  #IA
  f_power_IA = interp1d(np.log10(data.IA_data[:,1]),np.log10(data.IA_data[:,0]))
  iaPower = np.power(10,f_power_IA(np.log10(params['IA_thermal_noise']*10)))
  
  #ADC
  counterPower = data.ADCcounterEnergy * params['ADC_VCO_f0']
  adcPower = 2*(counterPower+data.ADCbbcoPower)
  
  #Total
  AFEpower = iaPower + adcPower
  
  
  ## DBE 
  #Detect
  detectPower = data.detectEnergyPerSample*params['ADC_Fs']
  
  #Feature extraction
  fePower = data.feEnergyPerBeat*params["beat_freq"]
  
  #SVM
  n_sv = params["SVM_SV_N_V"] + params["SVM_SV_N_S"]
  svmEnergyPerBeat = data.svmEnergyPerBeatFixed + data.svmEnergyPerBeatPerSV*n_sv
  svmPower = svmEnergyPerBeat*params["beat_freq"]
  
  #Total
  DBEpower = detectPower + fePower + svmPower
  
  averagePower = AFEpower + DBEpower
  
  return averagePower, AFEpower, DBEpower