import numpy as np
import os
import time

import LoadData as ld
import DigitalTwin as dtwin
import Calibrations as cal
import ReducedModel as rm

import matplotlib.pyplot as plt
import pyabc as pyabc

if __name__ == '__main__':
    #Set paths
    home = os.path.dirname(os.getcwd())
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
    print('ROM build time = ' + str(time.time() - start))
    
    params = {'d':dtwin.Parameter('d','g'), 'k':dtwin.ReducedParameter('k','r',ROM['V']), 'alpha':dtwin.Parameter('alpha','g'),
               'beta_a':dtwin.Parameter('beta_a','g'), 'beta_c': dtwin.Parameter('beta_c','g'), 'sigma':dtwin.Parameter('sigma','f')}
    
    params['d'].setBounds(np.array([1e-6,1e-3]))
    params['k'].setCoeffBounds(ROM['Library']['B']['coeff_bounds'])
    params['k'].setBounds(np.array([1e-6,0.1]))
    params['alpha'].setBounds(np.array([1e-6,0.8]))
    params['beta_a'].setBounds(np.array([0.35, 0.85]))
    params['beta_c'].setBounds(np.array([1.0, 5.5]))
    params['sigma'].update(1.626)
    
    priors = cal.generatePriors(params)
    
    start = time.time()
    test = cal.calibrateRXDIF_ABC_ROM(tumor, ROM, params, priors, dt = 0.5, options = {'n_pops': 3,'pop_size':50})
    print('MCMC time = ' + str(time.time() - start))
    