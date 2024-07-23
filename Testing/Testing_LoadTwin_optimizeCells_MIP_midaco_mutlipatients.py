

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
import Optimize_MIP_midaco as opt_mip


import midaco

def printTreatment(doses, days, string):
    print(string)
    for i in range(doses.size):
        print(str(doses[i])+' on day '+str(days[i]))
        

if __name__ == '__main__':
    #Set paths
    home = os.path.dirname(os.getcwd())
    datapath = home + '\Results\ABC_calibrated_twins_v2\\'
    savepath = home + '\Results\ABC_optimized_twins_MIP_v1\\'
    #Get results in folder
    files = os.listdir(datapath)
    
    # for z in range(len(files)):
    for z in range(len(files)):
        twin = pickle.load(open(datapath + files[z],'rb'))
        
        start = time.time()
        twin.simulations = twin.predict(dt = 0.5, threshold = 0.25, plot = False, visualize = False, parallel = False)
        print('Prediction time = ' + str(time.time() - start))
    
        metric = 'mean'
    
        problem_args = {'deliveries':5, 
                        'objectives':['final_cells'],
                        'constraints':['doses','max_concentration','ld50_toxicity','cumulative_cells'],
                        'threshold':0.25, 'max_dose':2.0, 'weights':np.array([1]),
                        'metric':metric,'estimated':True,'true_schedule':False,
                        'partial':True,'norm':True,'cum_cells_tol':2e-1}
        
        workers = 8
        evals = 2000
        variables = problem_args['deliveries'] * 2 + 2
        pr = False
        
        results = {}
    
        delivs_vec = np.array([2,3,4,5,6,7,8,9,10], dtype=int)
    
        tol = 0.0
        
        start_loop = time.time()
    
        for i, delivs in enumerate(delivs_vec):
            problem_args = {'deliveries':delivs, 
                            'objectives':['final_cells'],
                            'constraints':['doses','max_concentration','ld50_toxicity','cumulative_cells'],
                            'threshold':0.25, 'max_dose':2.0, 'weights':np.array([1]),
                            'metric':metric,'estimated':True,'true_schedule':False,
                            'partial':True,'norm':True,'cum_cells_tol':tol}
            
            problem = opt_mip.problemSetup_cellMin(twin, **problem_args)
            n = problem['deliveries']
            problem['initial_guess'] = opt_mip.randomizeInitialGuess_multi(twin,problem,workers)
            print('Initial guesses found')
            
            
            Model = opt_mip.CachedModel(twin, problem)
            
            soc_con = Model.soc_constraints()
            problem['con_size'] = soc_con.size
            
            key = b'Chase_Christenson_____[TRIAL-License-valid-until-1-Oct-2024]'
            
            opt_prob = {}
            opt_options = {}
            
            opt_prob['@'] = Model.constrained_objective
            
            opt_prob['o']  = 1  # Number of objectives 
            opt_prob['n']  = int(problem['deliveries'] * 2) 
            opt_prob['ni'] = int(problem['deliveries'])
            opt_prob['m']  = soc_con.size + 1
            opt_prob['me'] = 1
            
            bounds = np.array(problem['bounds'])
            opt_prob['xl'] = bounds[:,0].tolist()
            opt_prob['xu'] = bounds[:,1].tolist()
            
            opt_prob['x'] = problem['initial_guess']
            
            evals = evals
            
            opt_options['maxeval'] = evals
            opt_options['maxtime'] = 60*60*24
            if pr:
                opt_options['printeval'] = evals
            else:
                opt_options['printeval'] = 0
            opt_options['save2file'] = 0
            
            opt_options['param1']  = 0.1  # ACCURACY  
            opt_options['param2']  = 0.0  # SEED  
            opt_options['param3']  = 0.0  # FSTOP  
            opt_options['param4']  = 0.0  # ALGOSTOP  
            opt_options['param5']  = 0.0  # EVALSTOP  
            opt_options['param6']  = 0.0  # FOCUS  
            opt_options['param7']  = 8.0  # ANTS  
            opt_options['param8']  = 4.0  # KERNEL  
            opt_options['param9']  = 0.0  # ORACLE  
            opt_options['param10'] = 0.0  # PARETOMAX
            opt_options['param11'] = 0.0  # EPSILON  
            opt_options['param12'] = 0.0  # BALANCE
            opt_options['param13'] = 0.0  # CHARACTER
            
            opt_options['parallel'] = workers
            
            if problem['initial_guess'].ndim == 1:
                mg = False
            else:
                mg = True
            if opt_options['param1'] == 0.0:
                Model.tol = 1e-3
            else:
                Model.tol = opt_options['param1']
                
            start = time.time()
            
            solution = midaco.run( opt_prob, opt_options, key , multiple_guess = mg, adaptive = True)
            
            print('Optimization time = ' + str(time.time() - start))
            
            doses, days = opt_mip.reorganize_doses(np.array(solution['x']), problem)
            
            idx = np.argsort(days)
            doses = doses[idx]
            days = days[idx]
            new_days = np.concatenate((problem['t_trx_soc'], days),axis = 0)
            new_doses = np.concatenate((problem['doses_soc'], doses), axis = 0)
            opt_trx_params = {'t_trx': new_days, 'doses': new_doses}
            
            optimal_simulation = twin.predict(treatment = opt_trx_params, threshold = 0.25, plot = False, change_t = problem['t_pred_end'])
            
            temp = {}
            temp['treatment'] = opt_trx_params
            temp['simulation'] = optimal_simulation
            temp['flag'] = solution['iflag']
            
            results[str(delivs)] = temp
                
        results['t_optimize']  = time.time() - start_loop
        results['soc_simulation'] = twin.predict(threshold = 0.25, plot = False, change_t = problem['t_pred_end'])
        
        
        save_name = savepath + files[z].replace('.mat','')
        with open(save_name,'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
        print('Patient '+str(z)+' complete.')
            