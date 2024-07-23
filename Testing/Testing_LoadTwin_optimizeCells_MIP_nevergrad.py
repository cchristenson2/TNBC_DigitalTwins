

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
    
    metric = 'percentile'

    problem_args = {'deliveries':10, 
                    'objectives':['final_cells'],
                    'constraints':['doses','max_concentration','ld50_toxicity', 'cumulative_cells'],
                    'threshold':0.25, 'max_dose':2.0, 'weights':np.array([1]),
                    'metric':metric,'estimated':True,'true_schedule':False,
                    'partial':True,'norm':True,'cum_cells_tol':3e-1}

    problem = opt_mip.problemSetup_cellMin(twin, **problem_args)
    problem['initial_guess'] = opt_mip.randomizeInitialGuess(twin,problem)
    
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
    # doses, days = opt_mip.reorganize_doses(np.array(solution['x']), problem)
    idx = np.argsort(days)
    new_days = np.concatenate((problem['t_trx_soc'], days[idx]),axis = 0)
    new_doses = np.concatenate((problem['doses_soc'], doses[idx]), axis = 0)
    opt_trx_params = {'t_trx': new_days, 'doses': new_doses}
    soc_simulation = twin.predict(threshold = 0.25, plot = False, change_t = problem['t_pred_end'])
    optimal_simulation = twin.predict(treatment = opt_trx_params, threshold = 0.25, plot = False, change_t = problem['t_pred_end'])
    dtwin.plotCI_comparison(soc_simulation, optimal_simulation)
    opt_mip.plotObj_comparison(problem, soc_simulation, optimal_simulation)
    cons = opt_mip.plotCon_comparison(problem, soc_simulation, optimal_simulation)
    
    print(optimizer. _select_optimizer_cls())
    
    
    
    
