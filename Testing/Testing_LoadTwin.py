
import numpy as np
import os
import time
import pickle

import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings(action = 'ignore', category=LinAlgWarning)

import DigitalTwin as dtwin
import Optimize as opt

if __name__ == '__main__':
    #Set paths
    home = os.path.dirname(os.getcwd())
    datapath = home + '\Results\ABC_calibrated_twins\\'
    #Get results in folder
    files = os.listdir(datapath)
    
    twin = pickle.load(open(datapath + files[0],'rb'))
    
    start = time.time()
    twin.simulations = twin.predict(dt = 0.5, threshold = 0.25, plot = True, visualize = False, parallel = False)
    print('Prediction time = ' + str(time.time() - start))
    
    problem_args = {'objectives':['final_cells', 'max_cells'], 'threshold':0.25}
    problem = opt.problemSetup_cellMin(twin.tumor, twin.simulations, **problem_args)
    start = time.time()
    output = twin.optimize_cellMin(problem)
    print('Optimization time = ' + str(time.time() - start))
    optimal_simulation = twin.predict(treatment = output[1], threshold = 0.25, plot = True)
    
    dtwin.plotCI_comparison(twin.simulations, optimal_simulation)
    opt.plotObj_comparison(problem, twin.simulations, optimal_simulation)
    opt.plotCon_comparison(problem, twin.simulations, optimal_simulation)