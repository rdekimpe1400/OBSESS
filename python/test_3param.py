
import os
import opti
import numpy as np
from joblib import dump, load

if __name__ == "__main__":
  
  scale = 0.1
  for i in range(0,10):
    result = load('./temp/opti_3param_dummy_{}/result.sav'.format(i))
    print(str(result))
    f = open('./temp/opti_3param_dummy_{}/result.log'.format(i), "w")
    f.write("{} ".format(str(result)))
    f.close()
  
  
  