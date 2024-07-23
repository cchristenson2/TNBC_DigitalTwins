

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
    
    tolerances = np.linspace(1e-1, 1e0, 10)
    
    results_grad = {}
    print('Gradient based:')
    for i, tol in enumerate(tolerances):
        problem_args = {'objectives':['final_cells'],
                        'constraints':['max_concentration','ld50_toxicity','cumulative_cells'],
                        'threshold':0.25, 'max_dose':2.0, 'weights':np.array([1]),
                        'metric':metric,'estimated':True,'true_schedule':False,
                        'partial':True,'norm':True,'cycles':False, 'cum_cells_tol':tol}
        
        mip_problem_args = problem_args.copy()
        mip_problem_args['deliveries'] = 3
        mip_problem_args.pop('cycles', None)
        problem_args['MIP_problem'] = opt_mip.problemSetup_cellMin(twin, **mip_problem_args)
        
        problem = opt.problemSetup_cellMin(twin, **problem_args)
        start = time.time()
        output = twin.optimize_cellMin(problem, method = 0)
        print('Optimization time = ' + str(time.time() - start))
        optimal_simulation = twin.predict(treatment = output[1], threshold = 0.25, plot = False, change_t = problem['t_pred_end'])
        
        temp = {'opt_output':output, 'simulation':optimal_simulation}
        results_grad[tol] = temp
        
    results_cobyla = {}
    print('COBYLA:')
    for i, tol in enumerate(tolerances):
        problem_args = {'objectives':['final_cells'],
                        'constraints':['max_concentration','ld50_toxicity','cumulative_cells'],
                        'threshold':0.25, 'max_dose':2.0, 'weights':np.array([1]),
                        'metric':metric,'estimated':True,'true_schedule':False,
                        'partial':True,'norm':True,'cycles':False, 'cum_cells_tol':tol}
        
        mip_problem_args = problem_args.copy()
        mip_problem_args['deliveries'] = 3
        mip_problem_args.pop('cycles', None)
        problem_args['MIP_problem'] = opt_mip.problemSetup_cellMin(twin, **mip_problem_args)
        
        problem = opt.problemSetup_cellMin(twin, **problem_args)
        start = time.time()
        output = twin.optimize_cellMin(problem, method = 1)
        print('Optimization time = ' + str(time.time() - start))
        optimal_simulation = twin.predict(treatment = output[1], threshold = 0.25, plot = False, change_t = problem['t_pred_end'])
        
        temp = {'opt_output':output, 'simulation':optimal_simulation}
        results_cobyla[tol] = temp
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1)
    for i, elem in enumerate(results_cobyla):
        data = results_cobyla[elem]['simulation']['cell_tc']
        tc = np.mean(data, axis = 1)
        t = results_cobyla[elem]['simulation']['tspan']
        ax.plot(t, tc, alpha = 0.5, label = 'Tol: '+str(elem))
    
    soc_simulation = twin.predict(dt = 0.5, threshold = 0.25, plot = False, change_t = problem['t_pred_end'])
    data = soc_simulation['cell_tc']
    tc = np.mean(data, axis = 1)
    t = results_cobyla[elem]['simulation']['tspan']
    ax.plot(t, tc, color = 'black', label = 'SoC')
    ax.legend(fontsize = 'xx-small')
    ax.set_title('Mean cell count')
    
    # results_basin = {}
    # print('Basin-hopping:')
    # for i, tol in enumerate(tolerances):
    #     problem_args = {'objectives':['final_cells'],
    #                     'constraints':['max_concentration','ld50_toxicity','cumulative_cells'],
    #                     'threshold':0.25, 'max_dose':2.0, 'weights':np.array([1]),
    #                     'metric':metric,'estimated':True,'true_schedule':False,
    #                     'partial':True,'norm':True,'cycles':False}
        
    #     mip_problem_args = problem_args.copy()
    #     mip_problem_args['deliveries'] = 3
    #     mip_problem_args.pop('cycles', None)
    #     problem_args['MIP_problem'] = opt_mip.problemSetup_cellMin(twin, **mip_problem_args)
        
    #     problem = opt.problemSetup_cellMin(twin, **problem_args)
    #     start = time.time()
    #     output = twin.optimize_cellMin(problem, method = 2)
    #     print('Optimization time = ' + str(time.time() - start))
    #     optimal_simulation = twin.predict(treatment = output[1], threshold = 0.25, plot = False, change_t = problem['t_pred_end'])
        
    #     temp = {'opt_output':output, 'simulation':optimal_simulation}
    #     results_basin[tol] = temp
    