# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:42:51 2024

@author: Chase Christenson
"""
import os
import numpy as np

import LoadData as ld
import ReducedModel as rm
import Operators as ops
import Library as lib

#Set paths
home = os.path.dirname(os.getcwd())
datapath = home + '\Data\PatientData_ungrouped\\'
#Get tumor information in folder
files = os.listdir(datapath)

#Load the first patient
print(files[0])
tumor = ld.LoadTumor_mat(datapath + files[0], crop2D = True, split = 2)

bounds = {}
bounds['d'] = np.array([1e-6, 1e-3])
bounds['k'] = np.array([1e-6, 0.1])
bounds['alpha'] = np.array([1e-6, 0.8])

required_ops = ('A','B','H','T')
params_ops = ('d','k','k','alpha')
type_ops = ('G','l','l','G')
zipped = zip(required_ops, params_ops, type_ops)

ROM = rm.constructROM_RXDIF(tumor, bounds, zipped = zipped, num = 2)

rm.visualizeBasis(ROM, tumor['Mask'].shape)

test_d = 5e-4

n,r = ROM['V'].shape
test_k = ROM['V'] @ (ROM['V'].T @ np.array(np.linspace(1e-6,0.1,n)))
test_k_r = ROM['V'].T @ test_k

A_r = ops.buildReduced_A(ROM['V'], tumor, test_d)

B_r = ops.buildReduced_B(ROM['V'], tumor, test_k)

A_r_interp = lib.interpolateGlobal(ROM['Library']['A'], test_d)
B_r_interp = lib.interpolateLocal(ROM['Library']['B'], test_k_r)