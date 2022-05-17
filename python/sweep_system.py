
import os
import opti
import numpy as np
from src import default

if __name__ == "__main__":
  
  scale_signal = 0.1
  
  opti.out_dir = './temp/sweep_all/'
  opti.clear_output_dir(opti.out_dir)

  opti.params = default.default_parameters()
  opti.params['input_scale'] = scale_signal
  
  opti.execute_framework([0,0,0])
  
  scale_max=4
  for scale in range(0,scale_max+1):
    n_steps = np.power(2,scale)
    for i in range(0,n_steps+1):
      for j in range(0,n_steps+1):
        for k in range(0,n_steps+1):
          if (np.mod(i,2)>0) or (np.mod(j,2)>0) or (np.mod(k,2)>0):
            opti.execute_framework([i/n_steps, j/n_steps, k/n_steps])

#  D = 0.5
#  opti.params = default.default_parameters()
#  opti.params['input_scale'] = scale_signal
#  opti.norm_param([1,1,D])
#  opti.norm_param([0.5,0.5,D])
#  opti.train_inference_framework(opti.params)
#  for i in range(0,50):
#    opti.params = default.default_parameters()
#    opti.params['input_scale'] = scale_signal
#    #opti.norm_param([1,1,D])
#    #opti.norm_param([0.5,0.5,D])
#    #opti.train_inference_framework(opti.params)
#    opti.execute_framework([0.5, 0.5, D])

#  scale_max=4
#  D = 0
#  print("D = {}".format(D),flush=True)
#  opti.params = default.default_parameters()
#  opti.params['input_scale'] = scale_signal
#  opti.norm_param([1,1,D])
#  opti.train_inference_framework(opti.params)
#  opti.execute_framework([0, 0, D])
#  for scale_j in range(0,scale_max+1):
#    n_steps_j = np.power(2,scale_j)
#    for j in range(0,n_steps_j+1):
#      if (np.mod(j,2)>0):
#          opti.execute_framework([j/n_steps_j, j/n_steps_j, D])
#  
#  for scale_i in range(0,scale_max+1):
#    n_steps_i = np.power(2,scale_i)
#    for i in range(0,n_steps_i+1):
#      if (np.mod(i,2)>0):
#        D = i/n_steps_i
#        print("D = {}".format(D),flush=True)
#        opti.params = default.default_parameters()
#        opti.params['input_scale'] = scale_signal
#        opti.norm_param([1,1,D])
#        opti.train_inference_framework(opti.params)
#        opti.execute_framework([0, 0, D])
#        for scale_j in range(0,scale_max+1):
#          n_steps_j = np.power(2,scale_j)
#          for j in range(0,n_steps_j+1):
#            if (np.mod(j,2)>0):
#                opti.execute_framework([j/n_steps_j, j/n_steps_j, D])
  
#  for i in range(0,len(Ds)):
#    D = Ds[i]
#    opti.params = default.default_parameters()
#    opti.params['input_scale'] = scale
#    opti.norm_param([1,1,D])
#    opti.train_inference_framework(opti.params)
#    opti.execute_framework([0, 0, D])
#    scale_max=4
#    for scale in range(0,scale_max+1):
#      n_steps = np.power(2,scale)
#      for i in range(0,n_steps+1):
#        for j in range(0,n_steps+1):
#          if (np.mod(i,2)>0) or (np.mod(j,2)>0):
#              opti.execute_framework([i/n_steps, j/n_steps, D])
  
  