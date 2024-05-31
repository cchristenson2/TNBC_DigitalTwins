# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:00:52 2024

@author: Chase Christenson
"""

import os
from scipy import io as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import LoadData as ld
import Optimize as opt

from scipy import integrate

matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 8

#Set paths
home = os.path.dirname(os.getcwd())
datapath = home + '\Data\PatientData_ungrouped\\'
#Get tumor information in folder
files = os.listdir(datapath)

#Load the first patient
tumor = ld.LoadTumor_mat(datapath + files[0], crop2D = 'true', split = 2)






problem = opt.problemSetup(tumor, objectives = ['final_cells'])


#Test get drug concentrations
trx_params = {'t_trx': tumor['t_trx'], 'beta':np.array([0.6,3.2])}
drugs = opt.getDrugConcentrations(trx_params)

auc = opt.getToxicity_ld50(drugs)
