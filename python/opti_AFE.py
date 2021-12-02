
import os
import opti
import numpy as np

if __name__ == "__main__":
  
  scale = 0.05
  opti.out_dir = './temp/opti_{}/'.format(scale)
  opti.params['input_scale'] = scale
  result = opti.run_optimization(opti.out_dir)
  print(result)
  
  
  