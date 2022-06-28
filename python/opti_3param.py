
import os
import opti
import numpy as np
from src import default
from joblib import dump, load

if __name__ == "__main__":
  dir_name = './temp/opti_3param_lowPower_sweep2_{}/'
  scale = 0.1
  for i in range(3,20):
    opti.out_dir = dir_name.format(i)
    opti.clear_output_dir(opti.out_dir)
    opti.params = default.default_parameters()
    opti.params['input_scale'] = scale
    #opti.train_inference_framework(opti.params)
    opti.params['input_scale'] = scale
    result = opti.run_optimization(opti.out_dir)
    print(result)
    f = open('{}result.log'.format(dir_name.format(i)), "w")
    f.write("{} ".format(str(result)))
    f.close()
    opti.execute_framework(result.x+[0,0,0])
    opti.execute_framework(result.x+[0.025,0,0])
    opti.execute_framework(result.x+[0.05,0,0])
    opti.execute_framework(result.x+[0.075,0,0])
    opti.execute_framework(result.x+[0.1,0,0])
    opti.execute_framework(result.x+[0.125,0,0])
    opti.execute_framework(result.x+[0.15,0,0])
    opti.execute_framework(result.x+[-0.025,0,0])
    opti.execute_framework(result.x+[-0.05,0,0])
    opti.execute_framework(result.x+[-0.075,0,0])
    opti.execute_framework(result.x+[-0.1,0,0])
    opti.execute_framework(result.x+[-0.125,0,0])
    opti.execute_framework(result.x+[-0.15,0,0])
    opti.execute_framework(result.x+[0,0.025,0])
    opti.execute_framework(result.x+[0,0.05,0])
    opti.execute_framework(result.x+[0,0.075,0])
    opti.execute_framework(result.x+[0,0.1,0])
    opti.execute_framework(result.x+[0,0.125,0])
    opti.execute_framework(result.x+[0,0.15,0])
    opti.execute_framework(result.x+[0,-0.025,0])
    opti.execute_framework(result.x+[0,-0.05,0])
    opti.execute_framework(result.x+[0,-0.075,0])
    opti.execute_framework(result.x+[0,-0.1,0])
    opti.execute_framework(result.x+[0,-0.125,0])
    opti.execute_framework(result.x+[0,-0.15,0])
    opti.execute_framework(result.x+[0,0,0.025])
    opti.execute_framework(result.x+[0,0,0.05])
    opti.execute_framework(result.x+[0,0,0.075])
    opti.execute_framework(result.x+[0,0,0.1])
    opti.execute_framework(result.x+[0,0,0.125])
    opti.execute_framework(result.x+[0,0,0.15])
    opti.execute_framework(result.x+[0,0,-0.025])
    opti.execute_framework(result.x+[0,0,-0.05])
    opti.execute_framework(result.x+[0,0,-0.075])
    opti.execute_framework(result.x+[0,0,-0.1])
    opti.execute_framework(result.x+[0,0,-0.125])
    opti.execute_framework(result.x+[0,0,-0.15])
  
  
  