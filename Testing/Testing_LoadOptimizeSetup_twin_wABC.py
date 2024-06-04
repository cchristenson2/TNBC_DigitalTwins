
import numpy as np
import os
import time
import pickle

import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings(action = 'ignore', category=LinAlgWarning)

import DigitalTwin as dtwin
import Optimize as opt

if __name__ == '__main__':
    
    #Set paths
    home = os.path.dirname(os.getcwd())
    datapath = home + '\Data\PatientData_ungrouped\\'
    #Get tumor information in folder
    files = os.listdir(datapath)
    
    #set arguments for loading
    load_args = {'crop2D': True, 'split': 2}
    
    #set arguments for ROM building
    bounds = {'d': np.array([1e-6, 1e-3]), 'k': np.array([1e-6, 0.1]),
              'alpha': np.array([1e-6, 0.8])}
    
    #set arguments for what should be calibrated

    
    required_ops = ('A','B','H','T')
    params_ops = ('d','k','k','alpha')
    type_ops = ('g','l','l','g')
    zipped = zip(required_ops, params_ops, type_ops)
    
    ROM_args = {'bounds': bounds, 'zipped': zipped}
    
    #Load the first patient
    twin = dtwin.DigitalTwin(datapath + files[0], load_args = load_args,
                                 ROM = True, ROM_args = ROM_args)
    
    params = {'d':dtwin.Parameter('d','g'),
              'k':dtwin.ReducedParameter('k','r',twin.ROM['V']),
              'alpha':dtwin.Parameter('alpha','g'),
              'beta_a':dtwin.Parameter('beta_a','g'),
              'beta_c': dtwin.Parameter('beta_c','g')}
    
    params['d'].setBounds(np.array([1e-6,1e-3]))
    params['k'].setBounds(np.array([1e-6,0.1]))
    params['k'].setCoeffBounds(twin.ROM['Library']['B']['coeff_bounds'])
    params['alpha'].setBounds(np.array([1e-6,0.8]))
    params['beta_a'].setBounds(np.array([0.35, 0.85]))
    params['beta_c'].setBounds(np.array([1.0, 5.5]))
    twin.setParams(params)
    twin.getPriors(params)
    
    cal_args = {'dt': 0.5, 'options': {'n_pops': 10,'pop_size':500,'epsilon':'calibrated','distance':'MSE','burnin':0.1}}
    
    start = time.time()
    twin.calibrateTwin('ABC_ROM', cal_args)
    print('ABC calibration time = ' + str(time.time() - start))
    
    start = time.time()
    twin.simulations = twin.predict(dt = 0.5, threshold = 0.25, plot = True, visualize = False, parallel = False)
    print('Prediction time = ' + str(time.time() - start))
    
    problem = opt.problemSetup_cellMin(twin.tumor, twin.simulations, objectives = ['final_cells', 'max_cells'], threshold = 0.25)
    start = time.time()
    output = twin.optimize_cellMin(problem)
    print('Optimization time = ' + str(time.time() - start))
    optimal_simulation = twin.predict(treatment = output[1], threshold = 0.25, plot = True)
    
    dtwin.plotCI_optimized(twin.simulations, optimal_simulation)
    