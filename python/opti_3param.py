
import os
import opti
import numpy as np

if __name__ == "__main__":
  
  scale = 0.1
  for i in range(0,10):
    opti.out_dir = './temp/opti_3param_{}/'.format(i)
    opti.params['input_scale'] = scale
    result = opti.run_optimization(opti.out_dir)
    print(result)
  
  
  