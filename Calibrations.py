# Calibrations.py
""" Assimilate data in tumor based on the model chosen
*calibrateRXDIF_LM(tumor, params, dt = 0.5, options = {}, parallel = True, plot = True)
    - Calibrate requested parameters using FOM and LM
*calibrateRXDIF_LM_ROM(tumor, ROM, params, dt = 0.5, options = {}, plot = True)
    - Calibrate requested parameters using ROM and LM
*calibrateRXDIF_gwMCMC_ROM(tumor, ROM, params, dt = 0.5, options = {}, plot = True)
    - Calibrate requested parameters using ROM and ensemble MCMC
*calibrateRXDIF_ABC_ROM(tumor, ROM, params, dt = 0.5, options = {}, plot = True)
    - Calibrate requested parameters using ROM and ABC
    *Parameter options:
        D - global, fixed, or off
        k - global, local, fixed, or off
        alpha - single drug efficacy only; global, fixed, or off
        betas - Each drug individually; global, fixed, or off
    *All work with 2D or 3D
    
#Useful internal functions
*getOperators(curr_params, ROM):

Last updated: 4/30/2024
"""
import numpy as np
import scipy.ndimage as ndi
import scipy.linalg as la
import concurrent.futures
import matplotlib.pyplot as plt
import copy

import cvxpy as cp
import scipy.optimize as sciop

import ForwardModels as fwd
import Library as lib

import time

#Supress warnings for ill-conditioned matrices
import warnings
warnings.filterwarnings(action = 'ignore', module = 'la.LinAlgWarning')
warnings.filterwarnings(action = 'ignore', module = 'la')

###############################################################################
#Global variables
call_dict = {
    'fwd.RXDIF_2D_wAC': fwd.RXDIF_2D_wAC,
    'fwd.RXDIF_3D_wAC': fwd.RXDIF_3D_wAC,
    'fwd.OP_RXDIF_wAC': fwd.OP_RXDIF_wAC}
###############################################################################
# Internal use only
def atleast_4d(arr):
    if arr.nidm < 4:
        return np.expand_dims(np.atleast_3d(arr),axis=3)
    else:
        return arr
    
def update_trx_params(curr_params, trx_params):
    trx_params_new = copy.deepcopy(trx_params)
    trx_params_new['beta'] = np.array([curr_params['beta_a'], curr_params['beta_c']])
    return trx_params_new

#Required for jacobian building
def getJScenarios(p_locator, curr_params, trx_params, delta, default_scenario):
    scenarios = []
    for i, elem in enumerate(p_locator):
        temp = default_scenario.copy()
        temp['params'] = curr_params.copy()
        if elem[1] == 0:
            temp['params'][elem[0]] = temp['params'][elem[0]] * delta
            temp['dif'] = temp['params'][elem[0]] - curr_params[elem[0]]
        else:
            arr = temp['params'][elem[0]].copy()
            initial = arr[elem[1]]
            new = arr[elem[1]].copy() * delta
            arr[elem[1]] = new
            temp['params'][elem[0]] = arr
            temp['dif'] = new - initial
        temp['trx_params'] = update_trx_params(temp['params'], default_scenario['trx_params'])
        scenarios.append(temp.copy())
        del temp
    return scenarios

def getJColumn(scenario):
        N = call_dict[scenario['model']](scenario['N0'], scenario['params']['k'], 
                             scenario['params']['d'], scenario['params']['alpha'],
                             scenario['trx_params'], scenario['t_true'],
                             scenario['h'], scenario['dt'], scenario['bcs'])[0]
        
        return N
    
def checkBounds(new, old, bounds):
    if new < bounds[0]:
        new = old - (old - bounds[0])/2
    elif new > bounds[1]:
        new = old + (bounds[1] - old)/2
    return new

def updateParams(calibrated, params):
    for i, elem in enumerate(params):
        params[i].update(calibrated[params[i].name])
    return params

###############################################################################
#FOM LM calibration
def calibrateRXDIF_LM(tumor, params, dt = 0.5, options = {}, parallel = True, plot = True):
    """
    Calibrate parameters to the data in tumor based on assignments for parameters
    For LM, tumor must have at least 2 data points
    
    Parameters
    ----------
    tumor : dict
        Contains all tumor information.
    params : List
        List of parameter objects, contains d, k, alpha, beta_a, and beta_c. If
        any of these are missing, assumes that value is 0.
    dt : float; default = 0.5
        Euler time step for the simulations.
    options : dictionary; default = empty dictionary
        Contains LM calibration parameters, if fields are missing resorts to default.
            - e_tol: Error tolerance stopping point.
            - e_conv: Convergence tolerance.
            - max_it: Max number of iterations.
            - delta: Perturbation factor for jacobian build.
            - pass: Move towards gauss newton factor.
            - fail: Move towards gradient descent factor.
            - lambda: Starting LM lambda.
            - j_freq: Successful parameter updates before jacobian updates.
    parallel : boolean; default = true
        Should parallel processing be used to build the Jacobian, recommended 
        if parameters are local.

    Returns
    -------
    params
        Updated list of parameters with calibrated values.

    """
    #Define options based on inputs and defaults
    options_fields = ['e_tol','e_conv','max_it','delta','pass','fail','lambda','j_freq']
    default_options = [1e-5,   1e-6,    500,     1.001,  7,     9,     1,       1]
    for i, elem in enumerate(options_fields):
        if elem not in options:
            options[elem] = default_options[i]
    l = options['lambda']     
    
    #Determine model to use based off data dimensions and prep data for calibration
    dim = tumor['Mask'].ndim
    if dim == 2:
        model = 'fwd.RXDIF_2D_wAC'
        N0 = tumor['N'][:,:,0]
        N_true = tumor['N'][:,:,1:]
    else:
        model = 'fwd.RXDIF_3D_wAC'
        N0 = tumor['N'][:,:,:,0]
        N_true = tumor['N'][:,:,:,1:]
    t_true = tumor['t_scan'][1:]    
    trx_params = {}
    trx_params['t_trx'] = tumor['t_trx']  
    trx_params['AUC'] = tumor['AUC']
    
    #Generate mask for where local parameters are defined
    if dim == 2:
        morph = ndi.generate_binary_structure(2, 1)
        Mask = N0 + np.sum(np.atleast_3d(N_true),2)
    else:
        morph = ndi.generate_binary_structure(2, 1)
        Mask = N0 + np.sum(atleast_4d(N_true),3)
    for i in range(2):
        Mask = ndi.binary_dilation(Mask, morph)
    
    #Get starting values for k, d, alpha, beta_a and beta_c
    required_params = ['k','d','alpha','beta_a','beta_c']
    found_params = []
    curr = {}
    for elem in params:
        found_params.append(elem.name)
        if elem.value == None:
            curr[elem.name] = np.mean(elem.bounds)
        else:
            curr[elem.name] = elem.value
        if elem.assignment == 'l':
            curr[elem.name] = curr[elem.name] * Mask
    for elem in required_params:
        if elem not in found_params:
            curr[elem] = 0
        
    #Pull out calibrated parameters
    p_locator = []
    for elem in params:
        if elem.assignment == 'g':
            p_locator.append([elem.name,0,elem.getBounds()])
        elif elem.assignment == 'l':
            for i in range(Mask[Mask==1].size):
                p_locator.append([elem.name,tuple(np.argwhere(Mask==1)[i,:]),elem.getBounds()])
    
    #Jacobian size
    num_p = len(p_locator)
    num_v = N_true.size    
    
    trx_params = update_trx_params(curr, trx_params)
    N_guess = call_dict[model](N0, curr['k'], curr['d'], curr['alpha'],
                               trx_params, t_true, tumor['h'], dt, tumor['bcs'])[0]
    SSE = np.sum((N_guess - N_true)**2, axis = None) 
    
    
    #Final prepping for loop
    iteration = 1
    j_curr = options['j_freq']
    default_scenario = {}
    default_scenario['N0'] = N0
    default_scenario['trx_params'] = trx_params
    default_scenario['t_true'] = t_true
    default_scenario['dt'] = dt
    default_scenario['model'] = model
    default_scenario['bcs'] = tumor['bcs']
    default_scenario['h'] = tumor['h']
    
    stats = {'d_track': list([curr['d']]), 'alpha_track': list([curr['alpha']]),
             'k_track': list([curr['k']]),'Lambda_track': list([l]),
             'SSE_track': list([SSE])}
    
    while iteration <= options['max_it'] and SSE > options['e_tol']:    
        if iteration % 10 == 0:
            print('iteration '+str(iteration))
        if j_curr == options['j_freq']: #Build jacobian
            J = np.zeros([num_v, num_p])
            #Use parallel
            all_scenarios = getJScenarios(p_locator, curr, trx_params,
                                          options['delta'], default_scenario)
            if parallel:
                with concurrent.futures.ProcessPoolExecutor(max_workers = 12) as executor:
                    futures = executor.map(getJColumn, all_scenarios)
                    for i, output in enumerate(futures):
                        J[:,i] = np.reshape((output - N_guess)/all_scenarios[i]['dif'],(-1,))
            else:
                for i, scenario in enumerate(all_scenarios):
                    J[:,i] = np.reshape((getJColumn(scenario) - N_guess)/scenario['dif'],(-1,))
            j_curr = 0
            
        #Calculate update
        damped_hessian = J.T @ J + l * np.diag(np.diag(J.T @ J))
        error_gradient = J.T @ np.reshape(N_true - N_guess,(-1))
        update = la.solve(damped_hessian, error_gradient)
        
        #Create test parameters
        test = copy.deepcopy(curr)
        for i, elem in enumerate(p_locator):
            #Global param update
            if elem[1] == 0:
                test[elem[0]] = checkBounds(test[elem[0]] + update[i], curr[elem[0]], elem[2])
            else:
                test[elem[0]][elem[1]] = checkBounds(test[elem[0]][elem[1]] + update[i],
                                                     curr[elem[0]][elem[1]], elem[2]).copy()  
    
        
        #Run with test parameters
        trx_params_test = update_trx_params(test, trx_params)
        N_test = call_dict[model](N0, test['k'], test['d'], test['alpha'],
                                  trx_params_test, t_true, tumor['h'], dt, tumor['bcs'])[0]
        SSE_test = np.sum((N_test - N_true)**2, axis = None) 
        
        if SSE_test < SSE:
            curr = copy.deepcopy(test)
            trx_params = copy.deepcopy(trx_params_test)
            N_guess = N_test.copy()
            
            l = l / options['pass']
            
            #Check for tolerance met
            if np.abs(SSE - SSE_test) < options['e_conv']:
                print('Algorithm converged on iteration: '+str(iteration))
                break
            SSE = SSE_test
            if SSE < options['e_tol']:
                print('Tolerance met on iteration: '+str(iteration))
                break
            j_curr += 1
        else:
            l = l * options['fail']
            if l > 1e15:
                l = 1e-15
            
            
        iteration += 1
        stats['d_track'].append(curr['d'].copy())
        stats['k_track'].append(curr['k'].copy())
        stats['alpha_track'].append(curr['alpha'].copy())
        stats['Lambda_track'].append(l)
        stats['SSE_track'].append(SSE)
        
    if plot:
        figure, ax = plt.subplots(2,3, layout="constrained")
        ax[0,0].plot(range(iteration), stats['d_track'])
        ax[0,0].set_ylabel('Diffusivity')
        # ax[0,1].plot(range(iteration), stats['k_track'])
        # ax[0,1].set_ylabel('Proliferation')
        ax[0,2].plot(range(iteration), stats['alpha_track'])
        ax[0,2].set_ylabel('Alpha')
        
        ax[1,1].plot(range(iteration), stats['Lambda_track'])
        ax[1,1].set_ylabel('Lambda')
        ax[1,1].set_yscale('log')
        ax[1,2].plot(range(iteration), stats['SSE_track'])
        ax[1,2].set_ylabel('SSE')
        
    return updateParams(curr, params), stats

###############################################################################
# Internal for ROM updating
def getOperators(curr_params, ROM):
    operators = {}
    for elem in curr_params:
        if elem == 'd':
            operators['A'] = lib.interpolateGlobal(ROM['Library']['A'], curr_params[elem])
        elif elem == 'k':
            # print(curr_params[elem])
            if curr_params[elem].size == 1:
                operators['B'] = lib.interpolateGlobal(ROM['Library']['B'], curr_params[elem])
                operators['H'] = lib.interpolateGlobal(ROM['Library']['H'], curr_params[elem])
            else:
                operators['B'] = lib.interpolateLocal(ROM['Library']['B'], curr_params[elem])
                operators['H'] = lib.interpolateLocal(ROM['Library']['H'], curr_params[elem])
        elif elem == 'alpha':
            operators['T'] = lib.interpolateGlobal(ROM['Library']['T'], curr_params[elem])
    return operators

def getJColumnROM(scenario):        
    N = call_dict[scenario['model']](scenario['N0_r'], 
                                     getOperators(scenario['params'], scenario['ROM']), 
                                     scenario['trx_params'], 
                                     scenario['t_true'], scenario['dt'])[0]
    
    return N

def forceBounds(curr_param, V, bounds, coeff_bounds, x0 = None):
    start = time.time()
    test = V @ curr_param
    if (len(np.nonzero((test < bounds[0]) & (abs(test) > 1e-6))) != 0 or len(np.nonzero(test > bounds[1])) != 0):
        
        indices = np.nonzero(abs(test)>1e-6)[0]
        A = np.concatenate((V[np.s_[indices,:]], -1*V[np.s_[indices,:]]), axis = 0)
        B = np.squeeze(np.concatenate((bounds[1]*np.ones([len(indices),1]), -1*bounds[0]*np.ones([len(indices),1])),axis = 0))
        
        ### CVXPY optimization test
        # x = cp.Variable(V.shape[1])
        # if x0 is None:
        #     x.value = curr_param
        # else:
        #     x.value = x0    
        # objective = cp.Minimize(cp.sum_squares(V@x - test))

        # constraints = [A@x <= B, coeff_bounds[:,0] <= x, x <= coeff_bounds[:,1]]
        # prob = cp.Problem(objective, constraints)
        # prob.solve()
        # curr_param = x.value.copy()
        
        ### scipy minimize
        if x0 is None:
            x0 = curr_param.copy()
        else:
            x0 =  np.mean(coeff_bounds, axis = 1)
        lincon = sciop.LinearConstraint(A, -np.inf, B)
        result = sciop.minimize(lambda x: np.linalg.norm(V@x - test, ord = 2), x0, constraints = lincon, method = 'COBYLA')
        curr_param = result.x.copy()
        
        print('Linear solve time = ' + str(time.time() - start))
        
    return curr_param

###############################################################################
#ROM LM calibration
def calibrateRXDIF_LM_ROM(tumor, ROM, params, dt = 0.5, options = {}, plot = True):
    """
    Calibrate parameters to the data in tumor based on assignments for parameters
    For LM, tumor must have at least 2 data points
    
    Parameters
    ----------
    tumor : dict
        Contains all tumor information.
    ROM : dict
        Contains basis, library for interpoaltion, and reduced tumor states for calibration
    params : List
        List of parameter objects, contains d, k, alpha, beta_a, and beta_c. If
        any of these are missing, assumes that value is 0.
    dt : float; default = 0.5
        Euler time step for the simulations.
    options : dictionary; default = empty dictionary
        Contains LM calibration parameters, if fields are missing resorts to default.
            - e_tol: Error tolerance stopping point.
            - e_conv: Convergence tolerance.
            - max_it: Max number of iterations.
            - delta: Perturbation factor for jacobian build.
            - pass: Move towards gauss newton factor.
            - fail: Move towards gradient descent factor.
            - lambda: Starting LM lambda.
            - j_freq: Successful parameter updates before jacobian updates.
    plot : boolean; default = true
        Should tracking variables be plot

    Returns
    -------
    params
        Updated list of parameters with calibrated values.

    """
    #Define options based on inputs and defaults
    options_fields = ['e_tol','e_conv','max_it','delta','pass','fail','lambda','j_freq']
    default_options = [1e-5,   1e-6,    500,     1.001,  7,     9,     1,       1]
    for i, elem in enumerate(options_fields):
        if elem not in options:
            options[elem] = default_options[i]
    l = options['lambda']
    
    #Determine model to use based off data dimensions and prep data for calibration
    model = 'fwd.OP_RXDIF_wAC'
    N0_r = ROM['ReducedTumor']['N_r'][:,0]
    N_true_r = ROM['ReducedTumor']['N_r'][:,1:]
    t_true = tumor['t_scan'][1:]    
    trx_params = {}
    trx_params['t_trx'] = tumor['t_trx']
    
    #Get size of ROM for reference
    n, r = ROM['V'].shape
    
    #Get initial guesses
    required_params = ['d','alpha','beta_a','beta_c']
    found_params = []
    full_bounds = {}
    initial_guess = {}
    coeff_bounds = {}
    curr = {}
    for elem in params:
        found_params.append(elem.name)
        if elem.value == None:
            if elem.assignment == 'r':
                #For R we build a guess proliferation map, then find the corresponding reduction
                k_target = np.mean(elem.getFullBounds())
                if k_target == 0:
                    k_target = 1e-3
                k_test = np.zeros([n,1])
                k_test[np.nonzero(np.abs(np.sum(ROM['V'],axis=1)) > 1e-6)] = k_target
                k_r_test = np.squeeze(ROM['V'].T @ k_test)

                k_r_test = forceBounds(k_r_test, ROM['V'], elem.getFullBounds(), elem.getBounds()).copy()

                curr[elem.name] = k_r_test.copy()
                full_bounds[elem.name] = elem.getFullBounds()
                coeff_bounds[elem.name] = elem.getBounds()
                initial_guess[elem.name] = k_r_test
            else:
                curr[elem.name] = np.mean(elem.getBounds())
                # curr[elem.name] = elem.getBounds()[0]*options['delta']
        else:
            curr[elem.name] = elem.value
    for elem in required_params: #Cannot turn off proliferation right now
        if elem not in found_params:
            curr[elem] = 0
    
    #Pull out calibrated parameters
    p_locator = []
    for elem in params:
        if elem.assignment == 'g':
            p_locator.append([elem.name,0,elem.getBounds()])
        elif elem.assignment == 'r':
            for i in range(r):
                p_locator.append([elem.name,(i,),elem.getBounds()[i,:]])
                
    #Jacobian size
    num_p = len(p_locator)
    num_v = N_true_r.size
    
    trx_params = update_trx_params(curr, trx_params)
    ops = getOperators(curr, ROM)
    N_guess_r = call_dict[model](N0_r, ops, trx_params, t_true, dt)[0]
    SSE = np.sum((N_guess_r - N_true_r)**2, axis = None) 
    
    #Final prepping for loop
    iteration = 1
    j_curr = options['j_freq']
    default_scenario = {}
    default_scenario['N0_r'] = N0_r
    default_scenario['trx_params'] = trx_params
    default_scenario['t_true'] = t_true
    default_scenario['dt'] = dt
    default_scenario['model'] = model
    default_scenario['ROM'] = ROM
    
    stats = {'d_track': list([curr['d']]), 'alpha_track': list([curr['alpha']]),
             'k_track': list([curr['k'].copy()]),'Lambda_track': list([l]),
             'SSE_track': list([SSE])}
    
    while iteration <= options['max_it'] and SSE > options['e_tol']:    
        if iteration % 10 == 0:
            print('iteration '+str(iteration))
        if j_curr == options['j_freq']: #Build jacobian
            J = np.zeros([num_v, num_p])
            #Use parallel
            all_scenarios = getJScenarios(p_locator, curr, trx_params, options['delta'], default_scenario)
            for i, scenario in enumerate(all_scenarios):
                J[:,i] = np.reshape((getJColumnROM(scenario) - N_guess_r)/scenario['dif'],(-1,))
            j_curr = 0
            
        #Calculate update
        damped_hessian = J.T @ J + l * np.diag(np.diag(J.T @ J))
        error_gradient = J.T @ np.reshape(N_true_r - N_guess_r,(-1))
        update = la.solve(damped_hessian, error_gradient)
        
        #Create test parameters
        test = copy.deepcopy(curr)
        for i, elem in enumerate(p_locator):
            #Global param update
            if type(elem[1]) == int:
                test[elem[0]] = checkBounds(test[elem[0]] + update[i], curr[elem[0]], elem[2])
            else:
                new = test[elem[0]][elem[1]].copy() + update[i]
                test[elem[0]][elem[1]] = checkBounds(new, curr[elem[0]][elem[1]].copy(), elem[2])
        
        #Force all bounds for reduced parameters
        for elem in full_bounds:
            test[elem] = forceBounds(test[elem], ROM['V'], full_bounds[elem], coeff_bounds[elem], initial_guess[elem])
        
        #Run with test parameters
        trx_params_test = update_trx_params(test, trx_params)
        ops_test = getOperators(test, ROM)
        N_test_r = call_dict[model](N0_r, ops_test, trx_params, t_true, dt)[0]
        SSE_test = np.sum((N_test_r - N_true_r)**2, axis = None) 
        
        if SSE_test < SSE:
            curr = copy.deepcopy(test)
            trx_params = copy.deepcopy(trx_params_test)
            ops = copy.deepcopy(ops_test)
            N_guess_r = N_test_r.copy()
            
            l = l / options['pass']
            
            #Check for tolerance met
            if np.abs(SSE - SSE_test) < options['e_conv']:
                print('Algorithm converged on iteration: '+str(iteration))
                break
            SSE = SSE_test
            if SSE < options['e_tol']:
                print('Tolerance met on iteration: '+str(iteration))
                break
            j_curr += 1
        else:
            l = l * options['fail']
            if l > 1e15:
                l = 1e-15 
        iteration += 1
        stats['d_track'].append(curr['d'].copy())
        stats['k_track'].append(curr['k'].copy())
        stats['alpha_track'].append(curr['alpha'].copy())
        stats['Lambda_track'].append(l)
        stats['SSE_track'].append(SSE)
        
    if plot:
        figure, ax = plt.subplots(2,3, layout="constrained")
        ax[0,0].plot(range(iteration), stats['d_track'])
        ax[0,0].set_ylabel('Diffusivity')
        ax[0,1].plot(range(iteration), np.vstack(stats['k_track']))
        ax[0,1].set_ylabel('Proliferation')
        ax[0,2].plot(range(iteration), stats['alpha_track'])
        ax[0,2].set_ylabel('Alpha')
        
        ax[1,1].plot(range(iteration), stats['Lambda_track'])
        ax[1,1].set_ylabel('Lambda')
        ax[1,1].set_yscale('log')
        ax[1,2].plot(range(iteration), stats['SSE_track'])
        ax[1,2].set_ylabel('SSE')
            
    return updateParams(curr, params), ops, stats




