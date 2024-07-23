

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
import Optimize as opt
import Optimize_MIP as opt_mip

if __name__ == '__main__':
    thresh = 0.25
    
    #Set paths
    home = os.path.dirname(os.getcwd())
    datapath = home + '\Results\ABC_calibrated_twins_v2\\'
    #Get results in folder
    files = os.listdir(datapath)
    
    twin = pickle.load(open(datapath + files[0],'rb'))
    
    start = time.time()
    twin.simulations = twin.predict(dt = 0.5, threshold = thresh, plot = False, visualize = False, parallel = False)
    print('Prediction time = ' + str(time.time() - start))
    
    metric = 'mean'
    
    # problem_args = {'objectives':['final_cells'],
    #                 'constraints':['max_concentration','ld50_toxicity','cumulative_cells'],
    #                 'threshold':0.25, 'max_dose':2.0, 'weights':np.array([1]),
    #                 'metric':metric,'estimated':True,'true_schedule':False,
    #                 'partial':True,'norm':True,'cycles':False,'interval':1,
    #                 'cum_cells_tol':0e-1}
    
    problem_args = {'objectives':['time_to_prog'],
                    'constraints':['total_dose','max_concentration','ld50_toxicity'],
                    'threshold':thresh, 'max_dose':2.0, 'weights':np.array([1]),
                    'metric':metric,'estimated':True,'true_schedule':False,
                    'partial':True,'norm':False,'cycles':False,'interval':1}
    
    # mip_problem_args = problem_args.copy()
    # mip_problem_args['deliveries'] = 5
    # mip_problem_args.pop('cycles', None)
    # mip_problem_args.pop('interval', None)
    # problem_args['MIP_problem'] = opt_mip.problemSetup_cellMin(twin, **mip_problem_args)
    
    problem = opt.problemSetup_cellMin(twin, **problem_args)
    start = time.time()
    output = twin.optimize_cellMin(problem, method = 1, initial_guess = 2, test_guess = True)
    print('Optimization time = ' + str(time.time() - start))
    soc_simulation = twin.predict(threshold = thresh, plot = False, change_t = problem['t_pred_end'])
    optimal_simulation = twin.predict(treatment = output[1], threshold = thresh, plot = False, change_t = problem['t_pred_end'])
    
    dtwin.plotCI_comparison(soc_simulation, optimal_simulation)
    objs = opt.plotObj_comparison(problem, soc_simulation, optimal_simulation)
    cons = opt.plotCon_comparison(problem, soc_simulation, optimal_simulation)
    
    # dtwin.plotCI_comparison(soc_simulation, output[3])
    # _ = opt.plotObj_comparison(problem, soc_simulation, output[3])
    # _ = opt.plotCon_comparison(problem, soc_simulation, output[3])
    
    # mean_cells = np.mean(optimal_simulation['cell_tc'],axis=1)
    # mean_cells_soc = np.mean(soc_simulation['cell_tc'],axis=1)
    # max_cells = np.max(optimal_simulation['cell_tc'],axis=1)
    # max_cells_soc = np.max(soc_simulation['cell_tc'],axis=1)
    # mid_cells = np.median(optimal_simulation['cell_tc'],axis=1)
    # mid_cells_soc = np.median(soc_simulation['cell_tc'],axis=1)