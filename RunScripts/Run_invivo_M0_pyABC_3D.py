
import numpy as np
import os
import time
import pickle

import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings(action = 'ignore', category=LinAlgWarning)

import DigitalTwin as dtwin


if __name__ == '__main__':
    #Set paths
    home = os.path.dirname(os.getcwd())
    datapath = home + '\Data\PatientData_ungrouped\\'
    #Get tumor information in folder
    files = os.listdir(datapath)
    
    #set arguments for loading
    load_args = {'crop2D': False}
    
    #set arguments for ROM building
    bounds = {'d': np.array([1e-6, 1e-3]), 'k': np.array([1e-6, 0.1]),
              'alpha': np.array([1e-6, 0.8])}
    
    #set arguments for what should be calibrated

    
    required_ops = ('A','B','H','T')
    params_ops = ('d','k','k','alpha')
    type_ops = ('G','l','l','G')
    zipped = zip(required_ops, params_ops, type_ops)
    
    ROM_args = {'bounds': bounds, 'zipped': zipped}
    
    #Load the first patient
    twin = dtwin.DigitalTwin(datapath + files[39], load_args = load_args,
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
    
    #Test calibrate LM
    cal_args = {'dt': 0.5, 'options': {'n_pops': 20,'pop_size':1000,'epsilon':'calibrated','distance':'MSE'}}
    
    start = time.time()
    twin.calibrateTwin('ABC_ROM', cal_args)
    print('ABC calibration time = ' + str(time.time() - start))
    
    start = time.time()
    twin.simulations = twin.predict(dt = 0.5, threshold = 0.25, plot = False, visualize = False, parallel = False)
    print('Prediction time = ' + str(time.time() - start))
    # twin.simulationStats(threshold = 0.25)