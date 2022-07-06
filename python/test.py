
import os
import opti
import numpy as np
from src import default
from joblib import dump, load

if __name__ == "__main__":
  dir_name = './temp/test/'
  scale = 0.1
  opti.out_dir = dir_name
  opti.clear_output_dir(opti.out_dir)
  opti.params = default.default_parameters()
  opti.params['input_scale'] = scale
  opti.power([.5,.5,.5])
  print(opti.power_gradient([.5,.5,.5]))
  
  
  