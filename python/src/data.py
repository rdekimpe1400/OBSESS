
import numpy as np

# Data
#DBE
detectEnergyPerSample = 3.6e-9
feEnergyPerBeat = 183e-9
svmEnergyPerBeatFixed = 400e-9
#svmEnergyPerBeatPerSV = 4e-9
svmEnergyPerBeatPerSV = 1e-9

#AFE
IA_data = np.array([[32e-6,0.6e-6],[7.9e-6,0.71e-6],[1.75e-6,0.94e-6],[0.47e-6,1.59e-6],[0.099e-6,4.04e-6],[0.025e-6,30e-6]]);
ADCbbcoPower = 0.51e-6
ADCcounterEnergy = 157e-14