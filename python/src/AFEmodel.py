# Analog front-end model
#
# R. Dekimpe
# Last update: 06.2020

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import time

from src.AFE import IA
from src.AFE import ADC

# Analog front-end signal transfer function model with non-idealities
# Inputs:
# - ECG: vector with single-lead analog signal [mV] from database
# - time: vector with corresponding time (same size as ECG)
# - time_dig: optional time vector for sampling (None if unused)
# - parameters
# Outputs:
# - ECG_dig: vector of digitized ECG samples
# - time_dig: sampling time vector (same as input time_dig if provided)
def analogFrontEndModel(ECG,time_analog, params = {}, IA_TF = None, VCO_TF=None, showFigures = False, stopwatch = False):
  
  t_start = time.time()
  
  # Create transfer functions
  if IA_TF is None:
    IA_TF= IA.IA_dist(params = params)
  if VCO_TF is None:
    VCO_TF= ADC.VCO_dist(params = params)
  
  t_tf = time.time()
  
  # Apply IA model
  IA_Vin_diff = ECG * 1e-3 # (mV)    
  IA_Vout_p = params["IA_DCout"] + IA_TF(IA_Vin_diff)/2 
  IA_Vout_m = params["IA_DCout"] - IA_TF(IA_Vin_diff)/2 
  
  t_ia = time.time()
  
  # Apply ADC model
  dt = time_analog[1]-time_analog[0]
  freq_VCO_p = VCO_TF(IA_Vout_p)
  ADC_out_p,time_dig = ADC.Counter_ADC(freq_VCO_p, time_analog, params["ADC_Fs"])
  freq_VCO_m = VCO_TF(IA_Vout_m)
  ADC_out_m,_ = ADC.Counter_ADC(freq_VCO_m, time_analog, params["ADC_Fs"])
  ADC_out = ADC_out_p - ADC_out_m

  t_adc = time.time()
  
  ECG_dig = ADC_out.astype(np.int32)
  
  # Plot signals
  if showFigures:
    fig, axs = plt.subplots(4, sharex=True)
    fig.suptitle("ECG signal digitization")
    axs[0].plot(time_analog,IA_Vin_diff)
    axs[0].set(ylabel="Differential IA input [V]")
    axs[1].plot(time_analog,IA_Vout_p,label="+")
    axs[1].plot(time_analog,IA_Vout_m,label="-")
    axs[1].set(ylabel="IA output [V]")
    axs[1].legend(loc='upper right')
    axs[2].plot(time_dig,ADC_out_p,label="+")
    axs[2].plot(time_dig,ADC_out_m,label="-")
    axs[2].set(ylabel="ADC output [#]",xlabel="Time[s]")
    axs[2].legend(loc='upper right')
    axs[3].plot(time_dig,ADC_out)
    axs[3].set(ylabel="Differential ADC output [V]")
  
  t_plot = time.time()
  
  if stopwatch :
    print("####################################################")
    print("AFE model timing report" )
    print("")
    print("- Create transfer functions     {:4.3f}s".format(t_tf-t_start))
    print("- IA model                      {:4.3f}s".format(t_ia-t_tf))
    print("- ADC model                     {:4.3f}s".format(t_adc-t_ia))
    print("- Plot signals                  {:4.3f}s".format(t_plot-t_adc))
    print("")
    print("- Total                         {:4.3f}s".format(t_plot-t_start))
    print("####################################################")
  
  return ECG_dig, time_dig