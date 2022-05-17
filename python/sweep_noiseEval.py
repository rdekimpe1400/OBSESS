
import os
import opti
import numpy as np
from src import default

if __name__ == "__main__":
  
  scale_signal = 0.1
  
  opti.out_dir = './temp/sweep_test_single_noTrain05/'
  opti.clear_output_dir(opti.out_dir)

  D = 0.5
  
  opti.params = default.default_parameters()
  opti.params['input_scale'] = scale_signal
  opti.norm_param([1,1,D])
  opti.train_inference_framework(opti.params)
  for i in range(0,50):
    opti.params = default.default_parameters()
    opti.params['input_scale'] = scale_signal
    opti.execute_framework([1, 1, D])

  opti.params = default.default_parameters()
  opti.params['input_scale'] = scale_signal
  opti.norm_param([1,1,D])
  opti.train_inference_framework(opti.params)
  for i in range(0,50):
    opti.params = default.default_parameters()
    opti.params['input_scale'] = scale_signal
    opti.execute_framework([0.5,0.5, D])
    
  opti.params = default.default_parameters()
  opti.params['input_scale'] = scale_signal
  opti.norm_param([0.5,0.5,D])
  opti.train_inference_framework(opti.params)
  for i in range(0,50):
    opti.params = default.default_parameters()
    opti.params['input_scale'] = scale_signal
    opti.execute_framework([0.5,0.5, D])
  
  for i in range(0,50):
    opti.params = default.default_parameters()
    opti.params['input_scale'] = scale_signal
    opti.norm_param([1,1,D])
    opti.train_inference_framework(opti.params)
    opti.execute_framework([0.5, 0.5, D])
  
  for i in range(0,50):
    opti.params = default.default_parameters()
    opti.params['input_scale'] = scale_signal
    opti.norm_param([0.5, 0.5,D])
    opti.train_inference_framework(opti.params)
    opti.execute_framework([0.5, 0.5, D])
  