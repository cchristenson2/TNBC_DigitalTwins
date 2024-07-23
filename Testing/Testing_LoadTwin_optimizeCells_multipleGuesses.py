

import numpy as np
import os
import time
import pickle

import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings(action = 'ignore', category=LinAlgWarning)

import DigitalTwin as dtwin
import Optimize as opt
import Optimize_MIP as opt_mip

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
                    'constraints':['max_concentration','ld50_toxicity','cumulative_cells'],
                    'threshold':0.25, 'max_dose':2.0, 'weights':np.array([1]),
                    'metric':metric,'estimated':True,'true_schedule':False,
                    'partial':True,'norm':True,'cycles':False,'cum_cells_tol':2e-1}
    
    mip_problem_args = problem_args.copy()
    mip_problem_args['deliveries'] = 3
    mip_problem_args.pop('cycles', None)
    problem_args['MIP_problem'] = opt_mip.problemSetup_cellMin(twin, **mip_problem_args) 
    problem = opt.problemSetup_cellMin(twin, **problem_args)
    for i in range(10):
        start = time.time()
        output = twin.optimize_cellMin(problem, method = 2, test_guess = True)
        print('Optimization time = ' + str(time.time() - start))
        soc_simulation = twin.predict(threshold = 0.25, plot = False, change_t = problem['t_pred_end'])
        optimal_simulation = twin.predict(treatment = output[1], threshold = 0.25, plot = False, change_t = problem['t_pred_end'])
        dtwin.plotCI_comparison(soc_simulation, output[3])
        dtwin.plotCI_comparison(soc_simulation, optimal_simulation)
        