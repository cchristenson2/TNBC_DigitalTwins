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

if __name__ == '__main__':
    #Set paths
    home = os.path.dirname(os.getcwd())
    datapath = home + '\Results\ABC_calibrated_twins_v2\\'
    savepath = home + '\Results\ABC_optimized_twins_v2\\'
    #Get results in folder
    files = os.listdir(datapath)
    
    thresh = 0.25
    
    # for z in range(len(files)):
    for z in range(len(files)):
        twin = pickle.load(open(datapath + files[z],'rb'))
        
        start = time.time()
        twin.simulations = twin.predict(dt = 0.5, threshold = thresh, plot = False, visualize = False, parallel = False)
        print('Prediction time = ' + str(time.time() - start))
    
        metric = 'mean'
        
        problem_args = {'objectives':['final_cells','max_cells'],
                        'constraints':['total_dose','max_concentration','ld50_toxicity'],
                        'threshold':0.25, 'max_dose':2.0, 'weights':np.array([0.5,0.5]),
                        'metric':metric,'estimated':True,'true_schedule':False,
                        'partial':True,'norm':True,'cycles':True,'interval':2}
        
        problem = opt.problemSetup_cellMin(twin, **problem_args)
        start = time.time()
        output = twin.optimize_cellMin(problem, method = 1, initial_guess = 2, test_guess = True)
        opt_time = time.time() - start
        
        results = {}
        results['soc_simulation'] = twin.predict(threshold = thresh, plot = False, change_t = problem['t_pred_end'])
        
        results['t_optimize'] = opt_time
        
        results['optimal_simulation'] = twin.predict(treatment = output[1], threshold = thresh, plot = False, change_t = problem['t_pred_end'])
        
        results['output'] = output
        
        # results['problem'] = problem
        
        save_name = savepath + files[z].replace('.mat','')
        with open(save_name,'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
        print('Patient '+str(z)+' complete.')
        
        
        