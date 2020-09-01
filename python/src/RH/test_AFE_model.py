# -*- coding: utf-8 -*-
"""
Created on Thu Jul  14 12:00:00 2020

@author: saeidi
"""

from AFEmodel_functions import *

#####################################################################
####                      *    TEST CODES    *
#####################################################################    

#################################### AFE Test codes

signalID = 209
CLK_freq = 250 
#oversampling_factor is just a simulation parameter for integral math
# an integer >= 2
oversampling_factor = 4
ECG_resamplerate = CLK_freq * oversampling_factor
(noise_BL, pink_noise) = input_noise(ECG_resamplerate, oversampling_factor)
IA_TF= IA_dist()
VCO_TF = VCO_dist()

t_stop = 60   # [sec] you can put "None" to use the whole ECG for simulation
import timeit
start = timeit.default_timer()
Output_digital = AFE_model(signalID, ECG_resamplerate, noise_BL, pink_noise, IA_TF, VCO_TF, CLK_freq, t_stop)
stop = timeit.default_timer()
print('Simulation Time is ', stop - start, 'sec with dt=', 1e6/(CLK_freq*oversampling_factor),'µsec') 

import matplotlib.pyplot as plt
t_CLK = np.linspace (0, (len(Output_digital)-1) /CLK_freq, len(Output_digital))
plt.figure(facecolor="white")
plt.title('Digital output with a sample rate of '+str(ECG_resamplerate)+' [sample/sec]')
plt.plot(t_CLK, Output_digital, 'blue')
plt.xlabel('time (Sec)')
plt.ylabel('Dig_out ')
plt.xlim([0.0, 1])

plt.show()