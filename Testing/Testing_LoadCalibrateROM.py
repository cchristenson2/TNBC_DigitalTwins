import numpy as np
import os
import time

import LoadData as ld
import ForwardModels as fwd
import Calibrations as cal
import ReducedModel as rm

#Set paths
home = os.path.dirname(os.path.dirname(os.getcwd()))
datapath = home + '\Data\PatientData_ungrouped\\'
#Get tumor information in folder
files = os.listdir(datapath)

#Load the first patient
tumor = ld.LoadTumor_mat(datapath + files[0], crop2D = True, split = 2)

bounds = {}
bounds['d'] = np.array([1e-6, 1e-3])
bounds['k'] = np.array([1e-6, 0.1])
bounds['alpha'] = np.array([1e-6, 0.8])

start = time.time()
ROM = rm.constructROM_RXDIF(tumor, bounds)
print('Total time = ' + str(time.time() - start))

params = ([fwd.Parameter('d','g'), fwd.ReducedParameter('k','r',ROM['V']), fwd.Parameter('alpha','g'),
           fwd.Parameter('beta_a','f'), fwd.Parameter('beta_c','f')])

params[0].setBounds(np.array([1e-6,1e-3]))
params[1].setBounds(ROM['Library']['B']['coeff_bounds'])
params[1].setFullBounds(np.array([1e-6,0.1]))
params[2].setBounds(np.array([1e-6,0.8]))

start = time.time()
params, operators, stats = cal.calibrateRXDIF_LM_ROM(tumor, ROM, params, options = {'max_it': 200,'j_freq': 1})
print('Total time = ' + str(time.time() - start))

kp = np.reshape(ROM['V']@params[1].get(), tumor['Mask'].shape)

import matplotlib.pyplot as plt

if tumor['Mask'].ndim == 3:
    s = round(tumor['Mask'].shape[2]/2)
    kp = kp[:,:,s]
plt.figure()
plt.imshow(kp)
plt.colorbar()
