# -*- coding: utf-8 -*-
"""
Created on Thu Jul  10 12:00:01 2020

@author: saeidi
"""
import numpy as np
from scipy.interpolate import interp1d

#################################### IA Distortion function
def IA_dist(params = {}):          
    X, X0 , Y = [], [], []
    for line in open(params["IA_TF_file"], 'r'):
      values = [float(s) for s in line.split()]
      if round(values[2]*1e6)/1e6 != 0:
        X.append(values[0])
        X0.append(values[1])
        Y.append(values[2])
#   # to extend interpolation rage  
    x_min_extend = -0.1
    x_max_extend = 0.1
    n_extend = 20
    Xnew=np.append(np.linspace(x_min_extend,min(X)-1e-4,n_extend), X)
    Ynew=np.append(np.ones(n_extend)* min(Y), Y)    
    Xnew=np.append(Xnew, np.linspace(max(X)+1e-4,x_max_extend,n_extend))
    Ynew=np.append(Ynew, np.ones(n_extend) * max(Y))    
    IA_Gain = interp1d(Xnew, Ynew, kind = 'cubic')        
    return IA_Gain
