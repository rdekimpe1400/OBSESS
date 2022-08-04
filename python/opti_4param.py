
import os
import opti
import numpy as np
from src import default

if __name__ == "__main__":
  for i in range(0,10):
    dir_name = './temp/opti_4param_{:d}/'.format(i)
    scale = 0.1
    opti.out_dir = dir_name
    opti.clear_output_dir(opti.out_dir)
    opti.params = default.default_parameters()
    opti.params['input_scale'] = scale
    #opti.train_inference_framework(opti.params)
    opti.params['input_scale'] = scale
    result = opti.run_optimization(opti.out_dir)
    print(result)
    f = open('{}result.log'.format(dir_name), "w")
    f.write("{} ".format(str(result)))
    f.close()
    
  
  
  