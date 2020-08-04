# -*- coding: utf-8 -*-
"""
Created on Thu Jul  10 12:00:01 2020

@author: saeidi
"""
import numpy as np
from scipy.interpolate import interp1d

#################################### VCO Transfer Function  
def VCO_dist():            
    X, Y , Y0 = [], [], []
    for line in open('./src/AFE/AFE_data/VCO_TF.dat', 'r'):
      values = [float(s) for s in line.split()]
      X.append(values[0])
      Y.append(values[1])
      Y0.append(values[2])
    # to extend interpolation rage     
    x_min_extend = -4
    x_max_extend = 4
    n_extend = 20
    Xnew=np.append(np.linspace(x_min_extend,min(X)-1e-4,n_extend), X)
    Ynew=np.append(np.ones(n_extend)* min(Y), Y)    
    Xnew=np.append(Xnew, np.linspace(max(X)+1e-4,x_max_extend,n_extend))
    Ynew=np.append(Ynew, np.ones(n_extend) * max(Y))      
    VCO_TF = interp1d(Xnew, Ynew, kind = 'cubic')
    return VCO_TF

#################################### Counter
def Counter_ADC(freq_VCO, t, CLK_freq): 
    dt = t[1]-t[0]
    CLK_period = 1/CLK_freq
    N_dt = int(CLK_period / dt)      
    i = 0
    SUM_phi = 0
    Counter = np.array([])
    Counting_per_clock = 0
    res_counter = 0 
    # Integrate (freq -> phase)
    dphi = dt*2*np.pi*(freq_VCO[1:]+freq_VCO[:-1])/2
    dphi = np.concatenate(([0],dphi))
    phi = np.cumsum(dphi)
    # Quantize
    phi_q = np.floor(phi/np.pi)
    # Sample
    T_s = np.arange(t[0],t[-1],1/CLK_freq)
    phi_q_s = np.floor(np.interp(T_s,t,phi_q))
    
    T_s = np.delete(T_s,0)
    Counter = np.diff(phi_q_s)
    
    return Counter, T_s
