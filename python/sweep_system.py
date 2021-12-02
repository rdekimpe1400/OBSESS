
import os
import opti
import numpy as np

if __name__ == "__main__":
  
  scale = 0.1
  
  opti.out_dir = './temp/data_{:4.3f}/'.format(scale)
  try:
    os.mkdir(opti.out_dir)
  except FileExistsError:
    print(opti.out_dir ,  " already exists")
  opti.full_log_file = opti.out_dir+opti.full_file_name
  opti.exec_log_file = opti.out_dir+opti.exec_file_name
  opti.data_log_file = opti.out_dir+opti.data_file_name
  
  f = open(opti.exec_log_file, "w")
  f.close()
  f = open(opti.data_log_file, "w")
  f.close()
  f = open(opti.full_log_file, "w")
  f.close()
  
  opti.params['input_scale'] = scale
  opti.execute_framework([0, 0, 0])
  
  #n_steps = 10
  #for k in range(0,n_steps+1):
  #  opti.execute_framework([k/n_steps,0.5,0.5])
  #
  #n_steps = 10
  #for k in range(0,n_steps+1):
  #  opti.execute_framework([0.5,k/n_steps,0.5])
  #
  #n_steps = 20
  #for k in range(0,n_steps+1):
  #  opti.execute_framework([0.5,0.5,k/n_steps])
  
  scale_max=5
  for scale in range(0,scale_max+1):
    n_steps = np.power(2,scale)
    for i in range(0,n_steps+1):
      for j in range(0,n_steps+1):
        for k in range(0,n_steps+1):
          if (np.mod(i,2)>0) or (np.mod(j,2)>0) or (np.mod(k,2)>0):
            opti.execute_framework([i/n_steps, j/n_steps, k/n_steps])
  
  
  