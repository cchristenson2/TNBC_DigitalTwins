import os
from scipy import io as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import LoadData as ld
import ForwardModels as fwd

matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 8

#Set paths
home = os.path.dirname(os.getcwd())
datapath = home + '\Data\PatientData_ungrouped\\'
#Get tumor information in folder
files = os.listdir(datapath)

#Load the first patient
tumor = ld.LoadTumor_mat(datapath + files[0], crop2D = 'true')

N0 = tumor['N'][:,:,0]
sy, sx = N0.shape

# Set up variables to run
kp = 0.05 * np.ones(N0.shape)
d = 1e-2
tspan = np.array([10,20])
h = (1,1)
dt = 0.5
bcs = tumor['bcs']

alpha = 0.20;
beta1 = 0.60;
beta2 = 3.25;
trx_params = {}
trx_params['t_trx'] = tumor['t_trx']
trx_params['beta'] = np.array([beta1, beta2])
trx_params['AUC'] = tumor['AUC']

#Run forward model
N_sim, _ = fwd.RXDIF_2D_wAC(N0, kp, d, alpha, trx_params, tspan, h, dt, bcs)

#Create subplot and visualize
figure, ax = plt.subplots(1,3)

p = ax[0].imshow(N0, clim=(0,1),cmap='jet')
ax[0].set_title('V1 Measured')
ax[0].set_xticks([]), ax[0].set_yticks([])
plt.colorbar(p,fraction=0.046, pad=0.04)

p = ax[1].imshow(N_sim[:,:,0], clim=(0,1),cmap='jet')
ax[1].set_title('V2 Simulation')
ax[1].set_xticks([]), ax[1].set_yticks([])
plt.colorbar(p,fraction=0.046, pad=0.04)

p = ax[2].imshow(N_sim[:,:,1], clim=(0,1),cmap='jet')
ax[2].set_title('V3 Simulation')
ax[2].set_xticks([]), ax[2].set_yticks([])
plt.colorbar(p,fraction=0.046, pad=0.04)