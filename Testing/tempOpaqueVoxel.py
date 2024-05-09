# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:34:36 2024

@author: Chase Christenson
"""

import numpy as np
import os
import LoadData as ld
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col

def _alphaToHex(n):
    string = '{0:x}'.format(round(n*255))
    if len(string) == 1:
        string = '0'+string
    return string


#Set paths
home = os.path.dirname(os.getcwd())
datapath = home + '\Data\PatientData_ungrouped\\'
#Get tumor information in folder
files = os.listdir(datapath)

#Load the first patient
tumor = ld.LoadTumor_mat(datapath + files[0], crop2D = False, split = 2)

N0 = tumor['N'][:,:,:,0]

N0_norm = (N0 - np.min(N0))/(np.max(N0) - np.min(N0))

colors = cm.jet(N0)
hex_colors = np.empty(N0.shape, dtype=object)
for i in range(N0.shape[0]):
    for j in range(N0.shape[1]):
        for k in range(N0.shape[2]):
            hex_colors[i,j,k] = col.rgb2hex(colors[i,j,k,:-1]) + _alphaToHex(np.exp(0.6931*N0_norm[i,j,k]) - 1)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
p = ax.voxels(N0, facecolors = hex_colors)
ax.set_aspect('equal')
plt.colorbar(ax)