# LoadData.py
""" Loading .mat files and processing for digital twins
*LoadData_mat(location, downsample='true', inplane=4, slice=1, crop2D = 'false')
"""

import os
import numpy as np
from scipy import io as io

pack_dens = 0.7405
cell_size = 4.189e-6

## Internal load for restructuring nested structs from MATLAB
def loadmat(filename):
    data = io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    for key in dict:
        if isinstance(dict[key], io.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, io.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def find(array, condition):
    for i, elem in np.ndenumerate(array):
        print(elem)
        if condition(elem):
            return i
        
def stackNTC(dat):
    NTCs = np.array([])
    for key in dat.keys():
        if 'NTC' in key:
            if NTCs.size == 0:
                NTCs = dat[key]
            else:
                if NTCs.shape[2] <= 1:
                    NTCs = np.append(NTCs, dat[key], 2)
                else:
                    NTCs = np.append(NTCs, dat[key], 3)


#Set paths
home = os.path.dirname(os.getcwd())
datapath = home + '\Data\PatientData_ungrouped\\'
#Get tumor information in folder
files = os.listdir(datapath)

#Load the first patient


data = loadmat(datapath + files[0])
tumor = {}

#Pull out treatment regimen info
sch = data['schedule_info']
times = sch['times']
times = np.cumsum(times)
times_labels = sch['schedule']
scan_times = times[times_labels=='S']
trx_times = times[times_labels==sch['drugtype']]
dimensions = sch['imagedims']
pcr = data['pcr']

#Pull out and prep scan data
# N = stackNTC(sorted(data['full_res_dat']))

# temp = data['full_res_dat']
# temp = temp[sorted(data['full_res_dat'])]

names = []
for key in data['full_res_dat'].keys():
    if 'NTC' in key:
        names.append(key)
        
for key in names:
    print(key)
    if 'NTCs' not in locals():
        NTCs = data['full_res_dat'][key]
        NTCs = np.expand_dims(NTCs,NTCs.ndim)
    else:
        NTCs = np.append(NTCs, np.expand_dims(data['full_res_dat'][key],NTCs.ndim - 1), NTCs.ndim - 1)
        
        
        
        
                
                
                
                
                