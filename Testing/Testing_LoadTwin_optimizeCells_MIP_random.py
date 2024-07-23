

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
    
    problem_args = {'deliveries':3, 'objectives':['final_cells'],
                    'constraints':['doses','max_concentration','ld50_toxicity','cumulative_cells'],
                    'threshold':0.25, 'max_dose':2.0, 'weights':np.array([1]),
                    'metric':metric,'estimated':True,'true_schedule':False,
                    'partial':True,'norm':True, 'dt':0.5}

    problem = opt_mip.problemSetup_cellMin(twin, **problem_args)
    Model = opt_mip.CachedModel(twin, problem)
    
    samples = 100
    treatments = [None]*samples
    objectives = [None]*samples
    for i in range(samples):
        temp = opt_mip.randomizeInitialGuess(twin,problem)
        temp_objective = Model.objective(temp)
        
        objectives[i] = temp_objective
        treatments[i] = temp
    
    
    import matplotlib.pyplot as plt
    plt.histogram(objectives)
    
    
    # optimal_simulation = twin.predict(treatment = output[1], threshold = 0.25, plot = False, change_t = problem['t_pred_end'])  
    # dtwin.plotCI_comparison(soc_simulation, optimal_simulation)
    # opt_mip.plotObj_comparison(problem, soc_simulation, optimal_simulation)
    # cons = opt_mip.plotCon_comparison(problem, soc_simulation, optimal_simulation)