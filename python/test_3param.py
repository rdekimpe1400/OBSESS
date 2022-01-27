
import os
import opti
import numpy as np
from joblib import dump, load

if __name__ == "__main__":
  
  scale = 0.1
  for i in range(0,3):
    result = load('./temp/opti_3param_{}/result.sav'.format(i))
    print(result)
  
  
  