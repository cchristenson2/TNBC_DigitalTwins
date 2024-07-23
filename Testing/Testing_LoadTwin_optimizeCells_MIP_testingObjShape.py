

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

import pyoptsparse as pyo

import midaco

def printTreatment(doses, days, string):
    print(string)
    for i in range(doses.size):
        print(str(doses[i])+' on day '+str(days[i]))
        
def normalizeDoses(doses, days):
    return 0
        

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

    problem_args = {'deliveries':5, 
                    'objectives':['final_cells'],
                    'constraints':['doses','max_concentration','ld50_toxicity', 'cumulative_cells'],
                    'threshold':0.25, 'max_dose':2.0, 'weights':np.array([1]),
                    'metric':metric,'estimated':True,'true_schedule':False,
                    'partial':True,'norm':True,'cum_cells_tol':2e-1}
    

    problem = opt_mip.problemSetup_cellMin(twin, **problem_args)
    problem['initial_guess'] = opt_mip.randomizeInitialGuess(twin,problem)
    n = problem['deliveries']
    print('Guess days: '+str(problem['initial_guess'][n:]))
    print('Guess doses: '+str(problem['initial_guess'][:n]))
    
    Model = opt_mip.CachedModel(twin, problem)
    
    soc_con = Model.soc_constraints()
    problem['con_size'] = soc_con.size
    con_size = soc_con.size
    
    
    bounds = np.array(problem['bounds'])
    dose_vec = np.linspace(bounds[0,0], bounds[0,1], num = 20)
    days_vec = np.arange(bounds[n,0], bounds[n,1])
    
    obj_grid = np.zeros((days_vec.size,1))
    con_grid = np.zeros((days_vec.size,1))
    
    ind_con_grid = np.zeros((days_vec.size, con_size))
    
    # for i, dose in enumerate(dose_vec):
    for j, day in enumerate(days_vec):
        temp_dose = problem['initial_guess'][:n].copy()
        temp_days = problem['initial_guess'][n:].copy()
        
        # temp_dose[0] = dose
        temp_days[0] = day
        
        idx = np.argsort(temp_days)
        
        
        obj = Model.objective(np.append(temp_dose, temp_days[idx]))
        temp = Model.constraints(np.append(temp_dose, temp_days[idx]))
        con = temp[0]
        violate = con[np.argwhere(con < 0)]
        
        obj_grid[j] = obj
        con_grid[j] = violate.size
        
        ind_con_grid[j,:] = con
    
    import matplotlib.pyplot as plt
    
    original = problem['initial_guess'][n]
    
    fig, ax = plt.subplots(1,2,layout='constrained')
    ax[0].plot(days_vec, obj_grid)
    ax[0].set_xlabel('Day of delivery')
    ax[0].set_ylabel('Objective function')
    ax[0].scatter(original, obj_grid[np.argwhere(days_vec==original)])
    
    ax[1].plot(days_vec, con_grid)
    ax[1].set_xlabel('Day of delivery')
    ax[1].set_ylabel('Constraints violated')    
    ax[1].scatter(original, con_grid[np.argwhere(days_vec==original)])    
            

    fig, ax = plt.subplots(1,con_size,layout = 'constrained', figsize = (con_size*4, 4))
    names = ['Total dose', 'A - Max concentration', 'C - Max concentration', 'A Toxicity','C - Toxicity', 'Cumulative cells']
    for i in range(con_size):
        ax[i].plot(days_vec, ind_con_grid[:,i])
        ax[i].set_xlabel('Day of delivery')
        ax[i].set_ylabel('Violation magnitude')
        ax[i].set_title(names[i])
        
    