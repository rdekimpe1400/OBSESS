# Analog front-end model
#
# R. Dekimpe
# Last update: 06.2020

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal

from src.AFE import IA
from src.AFE import ADC
from src.AFE import noise

# Analog front-end signal transfer function model with non-idealities
# Inputs:
# - ECG: vector with single-lead analog signal [mV] from database
# - time: vector with corresponding time (same size as ECG)
# - time_dig: optional time vector for sampling (None if unused)
# - parameters
# Outputs:
# - ECG_dig: vector of digitized ECG samples
# - time_dig: sampling time vector (same as input time_dig if provided)
def analogFrontEndModel(ECG,time_analog, params = {}, IA_TF = None, VCO_TF=None, showFigures = False, stopwatch = False, channel = 0):
  
  t_start = time.time()
  
  # Input signal properties
  f_analog = params['analog_resample']
  dt_analog = 1/params['analog_resample']
  n_analog = len(ECG)
  
  # Create transfer functions
  if IA_TF is None:
    IA_TF= IA.IA_dist(params = params)
  if VCO_TF is None:
    VCO_TF= ADC.VCO_dist(params = params)
  
  t_tf = time.time()
  
  # ECG from mV to V
  ECG = ECG * 1e-3
  
  # Add IA noise
  IA_thermal_noise, IA_flicker_noise, IA_noise = noise.input_noise(f_analog,n_analog,params['IA_thermal_noise'],params['IA_flicker_noise_corner'])
  ECG = ECG + IA_noise
  
  # Plot noise
  if showFigures:
    f, Pxx_den_therm = signal.periodogram(IA_thermal_noise, f_analog)
    f, Pxx_den_total = signal.periodogram(IA_noise, f_analog)
    fig, axs = plt.subplots(2)
    fig.suptitle("IA input-referred noise")
    axs[0].loglog(f, np.sqrt(Pxx_den_therm),label="Thermal (sim)")
    axs[0].loglog(f, np.ones_like(Pxx_den_therm)*params['IA_thermal_noise'],label="Thermal (ideal)")
    
    fx = np.fft.rfftfreq(n_analog,dt_analog)
    fx[0] = fx[1]
    fx = fx**(-1/2.)
    s = np.sqrt(np.mean(fx**2))
    if params['IA_flicker_noise_corner'] is not None:
      flick = fx*params['IA_thermal_noise']/(params['IA_flicker_noise_corner']**(-1/2))
      f, Pxx_den_flick = signal.periodogram(IA_flicker_noise, f_analog)
      axs[0].loglog(f, np.sqrt(Pxx_den_flick),label="Flicker (sim)")
      axs[0].loglog(np.fft.rfftfreq(n_analog,dt_analog),flick,label="Flicker (ideal)")
    
    axs[0].set_ylim([1e-10, 1e-5])
    axs[0].set_xlabel('frequency [Hz]')
    axs[0].set_ylabel('PSD [V/sqrt(Hz)]')
    axs[0].legend(loc='upper right')
    axs[1].loglog(f, np.sqrt(Pxx_den_total))
    axs[1].set_xlabel('frequency [Hz]')
    axs[1].set_ylabel('PSD [V/sqrt(Hz)]')
    axs[1].set_ylim([1e-10, 1e-5])
    plt.savefig('plots/IANoise_'+str(channel)+'.png')
  
  # Apply IA model
  IA_Vin_diff = ECG   
  IA_Vout_p = params["IA_DCout"] + IA_TF(IA_Vin_diff)/2 
  IA_Vout_m = params["IA_DCout"] - IA_TF(IA_Vin_diff)/2 
  
  t_ia = time.time()
  
  # Add ADC noise
  _,_, ADC_noise_p = noise.input_noise(f_analog,n_analog,params['ADC_thermal_noise'],None)
  IA_Vout_p = IA_Vout_p + ADC_noise_p
  _,_, ADC_noise_m = noise.input_noise(f_analog,n_analog,params['ADC_thermal_noise'],None)
  IA_Vout_m = IA_Vout_m + ADC_noise_m
  
  # Apply ADC model
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
    axs[3].plot(time_dig,np.left_shift(ECG_dig,np.log2(np.floor(6600000/(params["input_scale"]*params["ADC_VCO_f0"]))).astype(int)))
    #axs[3].plot(time_dig,ECG_dig)
    axs[3].set(ylabel="Differential ADC output [V]")
    axs[3].set(xlim=(0,50))
    plt.savefig('plots/AFE_'+str(channel)+'.png')
  
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
    print("####################################################",flush=True)
  
  return ECG_dig, time_dig