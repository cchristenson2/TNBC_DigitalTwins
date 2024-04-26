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

#Remove tumor info from struct
N0 = tumor['full_res_dat']['NTC1'][0,0]
s = int(np.round(N0.shape[2]/2))-1
N0_2D = N0[:,:,s]

plt.imshow(N0_2D)
plt.show()

h = np.array([1,1])
inplane = 4
inslice = 4

sy,sx = N0_2D.shape
X,Y = np.meshgrid(np.linspace(h[0],sx*h[0],sx), np.linspace(h[1],sy*h[1],sy), indexing='ij')
h_new = h.copy()
h_new[0] = sx*h[0] / round(sx/inplane)
h_new[1] = sy*h[1] / round(sy/inplane)
X_coarse,Y_coarse = np.meshgrid(np.linspace(h_new[0],sx*h[0],round(sx/inplane)),
                                         np.linspace(h_new[1],sy*h[1],round(sy/inplane)), indexing='ij')

from scipy.interpolate import RegularGridInterpolator
N0_2D_norm = N0_2D / np.prod(h)
x = np.linspace(h[0],sx*h[0],sx)
y = np.linspace(h[1],sy*h[1],sy)

interp = RegularGridInterpolator((x,y), N0_2D_norm)

N0_2D_down_norm = interp((X_coarse, Y_coarse))
N0_2D_down = N0_2D_down_norm * np.prod(h_new)

plt.imshow(N0_2D_down)