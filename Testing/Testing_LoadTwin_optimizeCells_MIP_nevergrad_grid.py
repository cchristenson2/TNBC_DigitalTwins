

import numpy as np
import os
import time
import pickle

import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings(action = 'ignore', category=LinAlgWarning)

import DigitalTwin as dtwin
import Optimize_MIP_nevergrad as opt_mip

import nevergrad as ng

def printTreatment(doses, days, string):
    print(string)
    for i in range(doses.size):
        print(str(doses[i])+' on day '+str(days[i]))
        

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
    
    tol_vec = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    delivs_vec = np.array([2,3,4,5,6,7,8,9,10], dtype=int)
    
    # tol_vec = np.array([0.8,0.9,1.0])
    # delivs_vec = np.array([9,10], dtype=int)
    
    results = {}
    
    for i, tol in enumerate(tol_vec):
        for j, delivs in enumerate(delivs_vec):
            print('Deliveries = '+str(delivs)+'; Tolerance = '+str(tol))

            problem_args = {'deliveries':int(delivs), 
                            'objectives':['final_cells'],
                            'constraints':['doses','max_concentration','ld50_toxicity', 'cumulative_cells'],
                            'threshold':0.25, 'max_dose':2.0, 'weights':np.array([1]),
                            'metric':metric,'estimated':True,'true_schedule':False,
                            'partial':True,'norm':True,'cum_cells_tol':tol}
        
            problem = opt_mip.problemSetup_cellMin(twin, **problem_args)
            problem['initial_guess'] = opt_mip.randomizeInitialGuess(twin,problem)
            print('Initial guess found')
            
            Model = opt_mip.CachedModel(twin, problem)
            
            soc_con = Model.soc_constraints()
            problem['con_size'] = soc_con.size
            
            n = problem['deliveries']
            bounds = np.array(problem['bounds'])
            x = ng.p.Array(shape=(n,),lower=bounds[0,0],upper=bounds[0,1])
            x.value = problem['initial_guess'][:n]
            y = ng.p.Array(shape=(n,),lower=bounds[n,0],upper=bounds[n,1]).set_integer_casting()
            y.value = problem['initial_guess'][n:]
            
            instru = ng.p.Instrumentation(x, y)
            optimizer = ng.optimizers.NGOpt(parametrization = instru, budget = 500)
            
            start = time.time()
            
            solution = optimizer.minimize(Model.objective, constraint_violation = [Model.constraints])
            
            print('Optimization time = ' + str(time.time() - start))
            
            temp = solution.value[0]
            doses, days = temp
            new_days = np.concatenate((problem['t_trx_soc'], days),axis = 0)
            new_doses = np.concatenate((problem['doses_soc'], doses), axis = 0)
            opt_trx_params = {'t_trx': new_days, 'doses': new_doses}
            optimal_simulation = twin.predict(treatment = opt_trx_params, threshold = 0.25, plot = False, change_t = problem['t_pred_end'])
            
            temp = {}
            temp['treatment'] = opt_trx_params
            temp['simulation'] = optimal_simulation
            
            results[str(delivs)+'_'+str(tol)] = temp
            
    soc_simulation = twin.predict(threshold = 0.25, plot = False, change_t = problem['t_pred_end'])
            
    temp = np.array(soc_simulation['cell_tc'])
    soc_cells = np.mean(temp[-1,:])
    
    cells_grid = np.zeros((tol_vec.size, delivs_vec.size))
    con_grid = np.zeros((tol_vec.size, delivs_vec.size))
    for i, tol in enumerate(tol_vec):
        for j, delivs in enumerate(delivs_vec):
            temp = np.array(results[str(delivs)+'_'+str(tol)]['simulation']['cell_tc'])
            cells_grid[i,j] = np.mean(temp[-1,:])
            
            cons = opt_mip.plotCon_comparison(problem, soc_simulation, optimal_simulation)
    
    import matplotlib.pyplot as plt
    
    plt.imshow(100*(cells_grid - soc_cells)/soc_cells)