
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
    
    metric = 'percentile'
    method = 0
    
    # problem_args = {'objectives':['final_cells', 'max_cells'], 'threshold':0.25, 'max_dose':2.0}
    problem_args = {'objectives':['final_cells','cumulative_cells'],
                    'constraints':['max_concentration','ld50_toxicity'],
                    'threshold':0.25,
                    'max_dose':2.0, 'weights':np.array([0.0, 1.0]), 'metric':metric,'estimated':True,
                    'metric':metric,'estimated':True,'true_schedule':False,
                    'partial':True,'norm':True}
    mip_problem_args = problem_args.copy()
    mip_problem_args['deliveries'] = 4
    problem_args['MIP_problem'] = opt_mip.problemSetup_cellMin(twin, **mip_problem_args)
    problem = opt.problemSetup_cellMin(twin, **problem_args)
    
    start = time.time()
    output = twin.optimize_cellMin(problem, method = method)
    print('Optimization time = ' + str(time.time() - start))
    optimal_simulation_01 = twin.predict(treatment = output[1], threshold = 0.25, plot = False)

    
    problem_args = {'objectives':['final_cells','cumulative_cells'],
                    'constraints':['max_concentration','ld50_toxicity'],
                    'threshold':0.25,
                    'max_dose':2.0, 'weights':np.array([0.2, 0.8]), 'metric':metric,'estimated':True,
                    'metric':metric,'estimated':True,'true_schedule':False,
                    'partial':True,'norm':True}
    mip_problem_args = problem_args.copy()
    mip_problem_args['deliveries'] = 4
    problem_args['MIP_problem'] = opt_mip.problemSetup_cellMin(twin, **mip_problem_args)
    problem = opt.problemSetup_cellMin(twin, **problem_args)
    
    start = time.time()
    output = twin.optimize_cellMin(problem, method = method)
    print('Optimization time = ' + str(time.time() - start))
    optimal_simulation_28 = twin.predict(treatment = output[1], threshold = 0.25, plot = False)
    
    problem_args = {'objectives':['final_cells','cumulative_cells'],
                    'constraints':['max_concentration','ld50_toxicity'],
                    'threshold':0.25,
                    'max_dose':2.0, 'weights':np.array([0.4, 0.6]), 'metric':metric,'estimated':True,
                    'metric':metric,'estimated':True,'true_schedule':False,
                    'partial':True,'norm':True}
    mip_problem_args = problem_args.copy()
    mip_problem_args['deliveries'] = 4
    problem_args['MIP_problem'] = opt_mip.problemSetup_cellMin(twin, **mip_problem_args)
    problem = opt.problemSetup_cellMin(twin, **problem_args)
    
    start = time.time()
    output = twin.optimize_cellMin(problem, method = method)
    print('Optimization time = ' + str(time.time() - start))
    optimal_simulation_46 = twin.predict(treatment = output[1], threshold = 0.25, plot = False)
    
    problem_args = {'objectives':['final_cells','cumulative_cells'],
                    'constraints':['max_concentration','ld50_toxicity'],
                    'threshold':0.25,
                    'max_dose':2.0, 'weights':np.array([0.5, 0.5]), 'metric':metric,'estimated':True,
                    'metric':metric,'estimated':True,'true_schedule':False,
                    'partial':True,'norm':True}
    mip_problem_args = problem_args.copy()
    mip_problem_args['deliveries'] = 4
    problem_args['MIP_problem'] = opt_mip.problemSetup_cellMin(twin, **mip_problem_args)
    problem = opt.problemSetup_cellMin(twin, **problem_args)
    
    start = time.time()
    output = twin.optimize_cellMin(problem, method = method)
    print('Optimization time = ' + str(time.time() - start))
    optimal_simulation_55 = twin.predict(treatment = output[1], threshold = 0.25, plot = False)
    
    problem_args = {'objectives':['final_cells','cumulative_cells'],
                    'constraints':['max_concentration','ld50_toxicity'],
                    'threshold':0.25,
                    'max_dose':2.0, 'weights':np.array([0.6, 0.4]), 'metric':metric,'estimated':True,
                    'metric':metric,'estimated':True,'true_schedule':False,
                    'partial':True,'norm':True}
    mip_problem_args = problem_args.copy()
    mip_problem_args['deliveries'] = 4
    problem_args['MIP_problem'] = opt_mip.problemSetup_cellMin(twin, **mip_problem_args)
    problem = opt.problemSetup_cellMin(twin, **problem_args)
    
    start = time.time()
    output = twin.optimize_cellMin(problem, method = method)
    print('Optimization time = ' + str(time.time() - start))
    optimal_simulation_64 = twin.predict(treatment = output[1], threshold = 0.25, plot = False)
    
    problem_args = {'objectives':['final_cells','cumulative_cells'],
                    'constraints':['max_concentration','ld50_toxicity'],
                    'threshold':0.25,
                    'max_dose':2.0, 'weights':np.array([0.8, 0.2]), 'metric':metric,'estimated':True,
                    'metric':metric,'estimated':True,'true_schedule':False,
                    'partial':True,'norm':True}
    mip_problem_args = problem_args.copy()
    mip_problem_args['deliveries'] = 4
    problem_args['MIP_problem'] = opt_mip.problemSetup_cellMin(twin, **mip_problem_args)
    problem = opt.problemSetup_cellMin(twin, **problem_args)
    
    start = time.time()
    output = twin.optimize_cellMin(problem, method = method)
    print('Optimization time = ' + str(time.time() - start))
    optimal_simulation_82 = twin.predict(treatment = output[1], threshold = 0.25, plot = False)

    
    problem_args = {'objectives':['final_cells','cumulative_cells'],
                    'constraints':['max_concentration','ld50_toxicity'],
                    'threshold':0.25,
                    'max_dose':2.0, 'weights':np.array([1.0, 0.0]), 'metric':metric,'estimated':True,
                    'metric':metric,'estimated':True,'true_schedule':False,
                    'partial':True,'norm':True}
    mip_problem_args = problem_args.copy()
    mip_problem_args['deliveries'] = 4
    problem_args['MIP_problem'] = opt_mip.problemSetup_cellMin(twin, **mip_problem_args)
    problem = opt.problemSetup_cellMin(twin, **problem_args)
    
    start = time.time()
    output = twin.optimize_cellMin(problem, method = method)
    print('Optimization time = ' + str(time.time() - start))
    optimal_simulation_10 = twin.predict(treatment = output[1], threshold = 0.25, plot = False)
    
    obj_soc, obj_01 = opt.plotObj_comparison(problem, twin.simulations, optimal_simulation_01)
    obj_28 = opt.plotObj_comparison(problem, twin.simulations, optimal_simulation_28)[1]
    obj_46 = opt.plotObj_comparison(problem, twin.simulations, optimal_simulation_46)[1]
    obj_55 = opt.plotObj_comparison(problem, twin.simulations, optimal_simulation_55)[1]
    obj_64 = opt.plotObj_comparison(problem, twin.simulations, optimal_simulation_64)[1]
    obj_82 = opt.plotObj_comparison(problem, twin.simulations, optimal_simulation_82)[1]
    obj_10 = opt.plotObj_comparison(problem, twin.simulations, optimal_simulation_10)[1]

    vals_soc = [np.mean(np.array(obj_soc),axis=1), np.std(np.array(obj_soc),axis=1)]

    vals_opt = np.zeros((7,2))
    std_opt = np.zeros((7,2))

    vals_opt[0,:] = np.mean(np.array(obj_01),axis=1)
    std_opt[0,:] = np.std(np.array(obj_01),axis=1)

    vals_opt[1,:] = np.mean(np.array(obj_28),axis=1)
    std_opt[1,:] = np.std(np.array(obj_28),axis=1)

    vals_opt[2,:] = np.mean(np.array(obj_46),axis=1)
    std_opt[2,:] = np.std(np.array(obj_46),axis=1)

    vals_opt[3,:] = np.mean(np.array(obj_55),axis=1)
    std_opt[3,:] = np.std(np.array(obj_55),axis=1)

    vals_opt[4,:] = np.mean(np.array(obj_64),axis=1)
    std_opt[4,:] = np.std(np.array(obj_64),axis=1)

    vals_opt[5,:] = np.mean(np.array(obj_82),axis=1)
    std_opt[5,:] = np.std(np.array(obj_82),axis=1)

    vals_opt[6,:] = np.mean(np.array(obj_10),axis=1)
    std_opt[6,:] = np.std(np.array(obj_10),axis=1)

    import matplotlib.pyplot as plt
    plt.figure(figsize = (4,4))
    plt.scatter(vals_opt[:,0],vals_opt[:,1])
    plt.scatter(vals_soc[0][0],vals_soc[0][1])
    plt.xlabel('Final cells (f1)')
    plt.ylabel('Cumulative cells (f2)')
    labels = ['', '', '',
              '', 'w1 = 0.6; w2 = 0.4', 'w1 = 0.8; w2 = 0.2',
              'w1 = 1.0; w2 = 0.0']
    for i, txt in enumerate(labels):
        plt.annotate(txt, (vals_opt[i,0], vals_opt[i,1]))
    plt.annotate('soc', (vals_soc[0][0],vals_soc[0][1]))
    
    
    #Plot a simulation
    dtwin.plotCI_comparison(twin.simulations, optimal_simulation_55)
    dtwin.plotCI_comparison(twin.simulations, optimal_simulation_64)
    dtwin.plotCI_comparison(twin.simulations, optimal_simulation_82)