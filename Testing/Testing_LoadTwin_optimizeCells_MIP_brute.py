

import numpy as np
import os
import time
import pickle

import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings(action = 'ignore', category=LinAlgWarning)

import DigitalTwin as dtwin
import Optimize_MIP as opt_mip

def arrayToLabels(array):
    labels = []
    for x in array:
        labels.append(np.array2string(x, precision=1))
    return labels

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
    
    # # problem_args = {'objectives':['final_cells', 'max_cells'], 'threshold':0.25, 'max_dose':2.0}
    # problem_args = {'objectives':['final_cells','midpoint_cells'], 'constraints':['cumulative_concentration', 'max_concentration'], 'threshold':0.25,
    #                 'max_dose':2.0, 'weights':np.array([0.5,0.5]), 'metric':metric,'estimated':True}
    # problem = opt.problemSetup_cellMin(twin, **problem_args)
    # start = time.time()
    # output = twin.optimize_cellMin(problem, method = 2)
    # print('Optimization time = ' + str(time.time() - start))
    # optimal_simulation = twin.predict(treatment = output[1], threshold = 0.25, plot = False)
    
    # dtwin.plotCI_comparison(twin.simulations, optimal_simulation)
    # opt.plotObj_comparison(problem, twin.simulations, optimal_simulation)
    # opt.plotCon_comparison(problem, twin.simulations, optimal_simulation)
    
    problem_args = {'deliveries':2, 'objectives':['final_cells'],
                    'constraints':['doses','max_concentration','ld50_toxicity','cumulative_cells'],
                    'threshold':0.25, 'max_dose':2.0, 'weights':np.array([1]),
                    'metric':metric,'estimated':True,'true_schedule':False,
                    'partial':True,'norm':True, 'dt':0.5}

    problem = opt_mip.problemSetup_cellMin(twin, **problem_args)
    start = time.time()
    output = twin.optimize_cellMin_mip(problem, method = 3)
    print('Optimization time = ' + str(time.time() - start))
    # soc_simulation = twin.predict(threshold = 0.25, plot = False, change_t = problem['t_pred_end'])
    # optimal_simulation = twin.predict(treatment = output[1], threshold = 0.25, plot = False, change_t = problem['t_pred_end'])
    
    # dtwin.plotCI_comparison(soc_simulation, optimal_simulation)
    # opt_mip.plotObj_comparison(problem, soc_simulation, optimal_simulation)
    # cons = opt_mip.plotCon_comparison(problem, soc_simulation, optimal_simulation)
    
    days_vec = np.linspace(32, 60, 20)
    results = {}
    for i, day1 in enumerate(days_vec):
        for j, day2 in enumerate(days_vec):
            test_trx = {}
            test_trx['t_trx'] = np.append(problem['t_trx_soc'], np.array([day1, day2]))
            test_trx['doses'] = np.ones(4)
            results[str(day1)+'_'+str(day2)] = twin.predict(treatment = test_trx,
                                                            threshold = 0.25,
                                                            plot = False,
                                                            change_t = problem['t_pred_end'])
    
    Model = opt_mip.CachedModel(twin, problem)
    obj_grid = np.zeros((20,20))
    con_grid = np.zeros((20,20))
    for i, day1 in enumerate(days_vec):
        for j, day2 in enumerate(days_vec):
            x = np.array([1.0,1.0,day1,day2])
            temp_obj = np.sqrt(Model.objective(x))
            obj_grid[i,j] = 100*(temp_obj * problem['soc_obj']['final_cells'] - problem['soc_obj']['final_cells'])/problem['soc_obj']['final_cells']
            temp_con = Model.constraints(x)
            max_con = temp_con[1]
            if max_con < 0:
                con_grid[i,j] = np.inf
            else:
                cum_cells = problem['soc_con']['cumulative_cells']+problem['tol_vec'][5] - temp_con[5]
                con_grid[i,j] = (cum_cells - problem['soc_con']['cumulative_cells'])/problem['soc_con']['cumulative_cells']
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2,figsize=(8,4),layout='constrained')
    p1 = ax[0].imshow(obj_grid, cmap=plt.cm.RdYlGn)
    x = ax[0].get_xticks()
    ax[0].set_xticks(x,labels = arrayToLabels(np.linspace(32, 60, x.size)))
    y = ax[0].get_yticks()
    ax[0].set_yticks(y,labels = arrayToLabels(np.linspace(32, 60, y.size)))
    ax[0].set_xlabel('Delivery 2')
    ax[0].set_ylabel('Delivery 1')
    ax[0].set_title('Final cells % Difference')
    ax[0].scatter(9,0,c = 'black', marker = 'o')
    plt.colorbar(p1,fraction=0.046, pad=0.04)
    p2 = ax[1].imshow(con_grid, vmin = 0, vmax = 1, cmap=plt.cm.RdYlGn)
    x = ax[1].get_xticks()
    ax[1].set_xticks(x,labels = arrayToLabels(np.linspace(32, 60, x.size)))
    y = ax[1].get_yticks()
    ax[1].set_yticks(y,labels = arrayToLabels(np.linspace(32, 60, y.size)))
    ax[1].set_xlabel('Delivery 2')
    ax[1].set_ylabel('Delivery 1')
    ax[1].set_title('Cumulative cells % Difference')
    ax[1].scatter(9,0,c = 'black', marker = 'o')
    cbar = plt.colorbar(p2,fraction=0.046, pad=0.04)
    temp = cbar.get_ticks()
    cbar.set_ticklabels(np.linspace(0,100,temp.size))         
            
            
            
            
    
    