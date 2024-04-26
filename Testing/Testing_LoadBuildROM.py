import numpy as np
import os

import LoadData as ld
import ROM as ROM

#Set paths
home = os.path.dirname(os.getcwd())
datapath = home + '\Data\PatientData_ungrouped\\'
#Get tumor information in folder
files = os.listdir(datapath)

#Load the first patient
print(files[0])
tumor = ld.LoadTumor_mat(datapath + files[0], crop2D = 'false')

#constructROM_RXDIF_wAC(tumor, bounds, augmentation = 'average', depth = 8, samples = None, r = 0):
bounds = {}
bounds['d'] = np.array([1e-6, 1e-3])
bounds['k'] = np.array([1e-6, 0.1])
bounds['alpha'] = np.array([1e-6, 0.8])

ROM = ROM.constructROM_RXDIF(tumor, bounds)