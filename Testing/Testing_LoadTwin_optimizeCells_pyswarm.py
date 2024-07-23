

import numpy as np
import os
import time
import pickle

import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings(action = 'ignore', category=LinAlgWarning)
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter(action = 'ignore', category = NumbaPerformanceWarning)

import DigitalTwin as dtwin
import Optimize_pyswarm as opt
import Optimize_MIP as opt_mip

from pyswarm import pso

if __name__ == '__main__':
    #Set paths
    home = os.path.dirname(os.getcwd())
    datapath = home + '\Results\ABC_calibrated_twins_v2\\'
    #Get results in folder
    files = os.listdir(datapath)
    
    twin = pickle.load(open(datapath + files[0],'rb'))
    
    start = time.time()
    twin.simulations = twin.predict(dt = 0.5, threshold = 0.25, plot = False, visualize = False, parallel = False)
    print('Prediction time = ' + str(time.time() - start))
    
    metric = 'mean'
    
    problem_args = {'objectives':['final_cells'],
                    'constraints':['doses','max_concentration','ld50_toxicity','cumulative_cells'],
                    'threshold':0.25, 'max_dose':2.0, 'weights':np.array([1]),
                    'metric':metric,'estimated':True,'true_schedule':False,
                    'partial':True,'norm':True,'cycles':False,'interval':1}
    
    mip_problem_args = problem_args.copy()
    mip_problem_args['deliveries'] = 3
    mip_problem_args.pop('cycles', None)
    mip_problem_args.pop('interval', None)
    problem_args['MIP_problem'] = opt_mip.problemSetup_cellMin(twin, **mip_problem_args)
    
    problem = opt.problemSetup_cellMin(twin, **problem_args)
    
    # temp = opt_mip.randomizeInitialGuess(twin, problem['MIP_problem'])
    # print('Initial guess found.')
    # temp_doses, temp_days = opt_mip.reorganize_doses(temp, problem['MIP_problem'])
    # problem['doses_guess'] = opt.mipInitialGuess(temp_doses, temp_days, problem)
    
    Model = opt.CachedModel(twin, problem)
    
    bounds = np.array(problem['bounds'])
    lb = bounds[:,0]
    ub = bounds[:,1]
    
    
    start = time.time()
    
    xopt, fopt = pso(Model.objective, lb, ub, f_ieqcons = Model.constraints, debug = False)
    
    print('Optimization time = ' + str(time.time() - start))
    
    
    new_days = np.concatenate((problem['t_trx_soc'], problem['potential_days']),axis = 0)
    new_doses = np.concatenate((problem['doses_soc'], xopt), axis = 0)
    trx_params = {'t_trx': new_days, 'doses': new_doses}
    
    soc_simulation = twin.predict(threshold = 0.25, plot = False, change_t = problem['t_pred_end'])
    optimal_simulation = twin.predict(treatment = trx_params, threshold = 0.25, plot = False, change_t = problem['t_pred_end'])
    
    dtwin.plotCI_comparison(soc_simulation, optimal_simulation)
    opt.plotObj_comparison(problem, soc_simulation, optimal_simulation)
    cons = opt.plotCon_comparison(problem, soc_simulation, optimal_simulation)