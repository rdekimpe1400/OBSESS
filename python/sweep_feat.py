
import os
import opti
import numpy as np
from src import default

if __name__ == "__main__":
  
  scale_signal = 0.1
  
  opti.out_dir = './temp/sweep_feat/'
  opti.clear_output_dir(opti.out_dir)

  opti.params = default.default_parameters()
  opti.params['input_scale'] = scale_signal
  
  ref = [0.5,0.5,0.6,1.0,1.0]
  opti.execute_framework([ref[0],ref[1],ref[2],ref[3],0])
  
  scale_max=6
  for scale in range(0,scale_max+1):
    n_steps = np.power(2,scale)
    for i in range(0,n_steps+1):
      if (np.mod(i,2)>0):
        opti.execute_framework([ref[0],ref[1],ref[2],ref[3],i/n_steps])
