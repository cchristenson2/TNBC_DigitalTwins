import numpy as np
import os
import time
import pickle

import warnings
warnings.filterwarnings(action = 'ignore', module = 'la.LinAlgWarning')

import DigitalTwin as dtwin


if __name__ == '__main__':
    #Set paths
    home = os.path.dirname(os.getcwd())
    datapath = home + '\Data\PatientData_ungrouped\\'
    #Get tumor information in folder
    files = os.listdir(datapath)
    
    #set arguments for loading
    load_args = {'crop2D': False, 'split': 2}
    
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
    twin = dtwin.DigitalTwin(datapath + files[0], load_args = load_args,
                                 ROM = True, ROM_args = ROM_args)
    
    ## FOM example
    #Set parameter variables for calibration, params for ROM must be set after basis is built
    # params = {'d':dtwin.Parameter('d','g'), 'k':dtwin.Parameter('k','l'), 'alpha':dtwin.Parameter('alpha','g'),
    #            'beta_a':dtwin.Parameter('beta_a','g'), 'beta_c': dtwin.Parameter('beta_c','g')}
    
    # params['d'].setBounds(np.array([1e-6,1e-3]))
    # params['k'].setBounds(np.array([1e-6,0.1]))
    # params['alpha'].setBounds(np.array([1e-6,0.8]))
    # params['beta_a'].setBounds(np.array([0.35, 0.85]))
    # params['beta_c'].setBounds(np.array([1.0, 5.5]))
    # twin.setParams(params)
    
    # #Test calibrate LM
    # cal_args = {'parallel': True, 'options':{'max_it':10,'j_freq':2}}
    # twin.calibrateTwin('LM_FOM', cal_args)
    # twin.predict(dt = 0.5, threshold = 0.25, plot = False, parallel = False)
    
    ## ROM example
    # Set parameter variables for calibration, params for ROM must be set after basis is built
    params = {'d':dtwin.Parameter('d','g'), 'k':dtwin.ReducedParameter('k','r',twin.ROM['V']), 'alpha':dtwin.Parameter('alpha','g'),
                'beta_a':dtwin.Parameter('beta_a','g'), 'beta_c': dtwin.Parameter('beta_c','g')}
    
    params['d'].setBounds(np.array([1e-6,1e-3]))
    params['k'].setBounds(np.array([1e-6,0.1]))
    params['k'].setCoeffBounds(twin.ROM['Library']['B']['coeff_bounds'])
    params['alpha'].setBounds(np.array([1e-6,0.8]))
    params['beta_a'].setBounds(np.array([0.35, 0.85]))
    params['beta_c'].setBounds(np.array([1.0, 5.5]))
    twin.setParams(params)
    
    #Test calibrate LM
    cal_args = {'options':{'max_it':100,'j_freq':2}}
    twin.calibrateTwin('LM_ROM', cal_args)
    twin.predict(dt = 0.5, threshold = 0.25, plot = False, visualize = True, parallel = False)
    twin.simulationStats(threshold = 0.25)
    
        
    
    ## MCMC example
    #Set parameter variables for calibration, params for ROM must be set after basis is built
    # params = {'d':dtwin.Parameter('d','g'), 'k':dtwin.ReducedParameter('k','r',twin.ROM['V']), 'alpha':dtwin.Parameter('alpha','g'),
    #             'beta_a':dtwin.Parameter('beta_a','g'), 'beta_c': dtwin.Parameter('beta_c','g'), 'sigma':dtwin.Parameter('sigma','f')}
    
    # params['d'].setBounds(np.array([1e-6,1e-3]))
    # params['k'].setBounds(np.array([1e-6,0.1]))
    # params['k'].setCoeffBounds(twin.ROM['Library']['B']['coeff_bounds'])
    # params['alpha'].setBounds(np.array([1e-6,0.8]))
    # params['beta_a'].setBounds(np.array([0.35, 0.85]))
    # params['beta_c'].setBounds(np.array([1.0, 5.5]))
    # params['sigma'].update(1.626)
    # twin.setParams(params)
    # twin.getPriors(params)
    
    # #Test calibrate LM
    # cal_args = {'options':{'samples':100}}
    # twin.calibrateTwin('gwMCMC_ROM', cal_args)
    # twin.predict(dt = 0.5, threshold = 0.25, plot = True, parallel = False)