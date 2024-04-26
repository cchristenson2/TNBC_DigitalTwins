import os
from scipy import io as io
import numpy as np
import matplotlib.pyplot as plt

#Set paths
home = os.path.dirname(os.getcwd())
datapath = home + '\Data\PatientData_ungrouped\\'
#Get tumor information in folder
files = os.listdir(datapath)

#Load the first patient
print(files[0])
tumor = io.loadmat(datapath + files[0])

Mask = tumor['full_res_dat']['BreastMask'][0,0]

bcs = np.zeros((Mask.shape+(3,)))
for z in range(Mask.shape[2]):
    for y in range(Mask.shape[0]):
        for x in range(Mask.shape[1]):
            if Mask[y,x,z] == 0:
                bcs[y,x,z,:] = np.array([2,2,2])
            else:
                boundary = np.array([0,0,0])
                #check X
                if x == 0:
                    boundary[0] = -1
                elif x == Mask.shape[1]-1:
                    boundary[0] = 1
                else:
                    if Mask[y,x-1,z] == 0:
                        boundary[0] = -1
                    elif Mask[y,x+1,z] == 0:
                        boundary[0] = 1            
                #check Y
                if y == 0:
                    boundary[1] = -1
                elif y == Mask.shape[0]-1:
                    boundary[1] = 1
                else:
                    if Mask[y-1,x,z] == 0:
                        boundary[1] = -1
                    elif Mask[y+1,x,z] == 0:
                        boundary[1] = 1
                #check Z
                if z == 0:
                    boundary[2] = -1
                elif z == Mask.shape[2]-1:
                    boundary[2] = 1
                else:
                    if Mask[y,x,z-1] == 0:
                        boundary[2] = -1
                    elif Mask[y,x,z+1] == 0:
                        boundary[2] = 1
                bcs[y,x,z,:] = boundary