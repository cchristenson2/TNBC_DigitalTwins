import os
from scipy import io as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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
print(files[0])
tumor = io.loadmat(datapath + files[0])

#Remove tumor info from struct
temp1 = tumor['coarse_res_dat']
temp2 = temp1['NTC1'][0,0]

N0 = temp2

#Crop to 2D
[sx,sy,sz] = N0.shape
slice = int(np.round(sz/2))-1

N0_2D = N0[:,:,slice]
N0_2D = 0.5 * N0_2D / N0_2D.max()
# N0_2D = np.flip(N0_2D)

# Set up variables to run
kp = 0.05 * np.ones(N0_2D.shape)
d = 1e-2
tspan = np.array([10,20])
h = 1
dt = 0.5
bcs = np.zeros([sy,sx,2])
bcs[:,0,0] = -1;
bcs[:,-1,0] = 1;
bcs[0,:,1] = -1;
bcs[-1,:,1] = 1;

#Run forward model
N_sim = fwd.RXDIF_2D(N0_2D, kp, d, tspan, h, dt, bcs)

#Create subplot and visualize
figure, ax = plt.subplots(1,3)

p = ax[0].imshow(N0_2D, clim=(0,1),cmap='jet')
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