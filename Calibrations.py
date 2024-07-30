# Calibrations.py
""" Calibrate data in tumor based on the method chosen
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
*generatePriors(params)
    - Generates priors based on parameters being calibrated. Lots of assumptions
      hard coded into this.

Last updated: 5/23/2024
"""
import numpy as np
import scipy.ndimage as ndi
import scipy.linalg as la
import scipy.stats as stats
import scipy.optimize as sciop
import concurrent.futures
import matplotlib.pyplot as plt
import copy
import emcee
import multiprocessing as mp
import pyabc as pyabc
from operator import itemgetter

import os
import tempfile

import ForwardModels as fwd
import Library as lib



#Supress warnings for ill-conditioned matrices
#Doesn't always work and I don't know why
import warnings
warnings.filterwarnings(action = 'ignore', module = 'la.LinAlgWarning')
warnings.filterwarnings(action = 'ignore', module = 'la')

############################# Global variables ################################
call_dict = {
    'fwd.RXDIF_2D_wAC': fwd.RXDIF_2D_wAC,
    'fwd.RXDIF_3D_wAC': fwd.RXDIF_3D_wAC,
    'fwd.OP_RXDIF_wAC': fwd.OP_RXDIF_wAC}

########################### FOM LM calibration ################################
def calibrateRXDIF_LM(tumor, params, dt = 0.5, options = {}, parallel = False,
                      plot = False, output = False):
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
    stats
        Tracking variables from calibration

    """
    #Define options based on inputs and defaults
    options_fields = ['e_tol','e_conv','max_it','delta','pass','fail',
                      'lambda','j_freq']
    default_options = [1e-5,   1e-6,    500,     1.001,  7,     9,
                       1,       1]
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
        Mask = N0 + np.sum(_atleast_4d(N_true),3)
    for i in range(2):
        Mask = ndi.binary_dilation(Mask, morph)
    
    #Get starting values for d, k, alpha, beta_a and beta_c
    required_params = ['d','k','alpha','beta_a','beta_c']
    found_params = []
    curr = {}
    for elem in params:
        found_params.append(elem)
        if params[elem].value == None:
            curr[elem] = np.mean(params[elem].bounds)
        else:
            curr[elem] = params[elem].value
        if params[elem].assignment == 'l':
            curr[elem] = curr[elem] * Mask
    for elem in required_params:
        if elem not in found_params:
            if elem == 'k':
                curr[elem] = np.zeros(tumor['Mask'].shape)
            else:
                curr[elem] = 0
        
    #Pull out calibrated parameters
    p_locator = []
    for elem in params:
        if params[elem].assignment == 'g':
            p_locator.append([elem,0,params[elem].getBounds()])
        elif params[elem].assignment == 'l':
            for i in range(Mask[Mask==1].size):
                p_locator.append([elem,tuple(np.argwhere(Mask==1)[i,:]),
                                  params[elem].getBounds()])
    
    #Jacobian size
    num_p = len(p_locator)
    num_v = N_true.size    
    
    trx_params = _update_trx_params(curr, trx_params)
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
        if output == True:
            if iteration % 10 == 0:
                print('iteration '+str(iteration))
        if j_curr == options['j_freq']: #Build jacobian
            J = np.zeros([num_v, num_p])
            #Use parallel
            all_scenarios = _getJScenarios(p_locator, curr, trx_params,
                                          options['delta'], default_scenario)
            if parallel:
                with (concurrent.futures.ProcessPoolExecutor()
                      as executor):
                    futures = executor.map(_getJColumn, all_scenarios)
                    for i, output in enumerate(futures):
                        J[:,i] = np.reshape((output - N_guess)
                                            /all_scenarios[i]['dif'],(-1,))
            else:
                for i, scenario in enumerate(all_scenarios):
                    J[:,i] = np.reshape((_getJColumn(scenario) - N_guess)
                                        /scenario['dif'],(-1,))
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
                test[elem[0]] = _checkBounds(test[elem[0]] + update[i],
                                            curr[elem[0]], elem[2])
            else:
                test[elem[0]][elem[1]] = _checkBounds(test[elem[0]][elem[1]]
                                                     + update[i],
                                                     curr[elem[0]][elem[1]],
                                                     elem[2]).copy()  
    
        
        #Run with test parameters
        trx_params_test = _update_trx_params(test, trx_params)
        N_test = call_dict[model](N0, test['k'], test['d'], test['alpha'],
                                  trx_params_test, t_true, tumor['h'], dt,
                                  tumor['bcs'])[0]
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
        
    return _updateParams(curr, params), stats, model, N_guess

########################### ROM LM calibration ################################
def calibrateRXDIF_LM_ROM(tumor, ROM, params, dt = 0.5, options = {}, plot = False, output = False):
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
    stats
        Tracking variables from calibration
    """
    #Define options based on inputs and defaults
    options_fields = ['e_tol','e_conv','max_it','delta','pass','fail',
                      'lambda','j_freq']
    default_options = [1e-5,   1e-6,    500,     1.001,  7,     9,
                       1,       1]
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
    required_params = ['d','k','alpha','beta_a','beta_c']
    found_params = []
    full_bounds = {}
    initial_guess = {}
    coeff_bounds = {}
    curr = {}
    for elem in params:
        found_params.append(elem)
        if params[elem].value == None:
            if params[elem].assignment == 'r':
                #For R we build a guess proliferation map, then find the corresponding reduction
                k_target = np.mean(params[elem].getBounds())
                if k_target == 0:
                    k_target = 1e-3
                k_test = np.zeros([n,1])
                k_test[np.nonzero(np.abs(np.sum(ROM['V'],axis=1)) > 1e-6)] = k_target
                k_r_test = np.squeeze(ROM['V'].T @ k_test)
                k_r_test = _forceBounds(k_r_test, ROM['V'], params[elem].getBounds(),
                                       params[elem].getCoeffBounds()).copy()
                curr[elem] = k_r_test.copy()
                full_bounds[elem] = params[elem].getBounds()
                coeff_bounds[elem] = params[elem].getCoeffBounds()
                initial_guess[elem] = k_r_test
            else:
                curr[elem] = np.mean(params[elem].getBounds())
        else:
            curr[elem] = params[elem].value
    for elem in required_params: #Cannot turn off proliferation right now
        if elem not in found_params:
            curr[elem] = 0
    
    #Pull out calibrated parameters
    p_locator = []
    for elem in params:
        if params[elem].assignment == 'g':
            p_locator.append([elem,0,params[elem].getBounds()])
        elif params[elem].assignment == 'r':
            for i in range(r):
                p_locator.append([elem,(i,),params[elem].getCoeffBounds()[i,:]])
                
    #Jacobian size
    num_p = len(p_locator)
    num_v = N_true_r.size
    
    trx_params = _update_trx_params(curr, trx_params)
    ops = lib.getOperators(curr, ROM)
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
        if output == True:
            if iteration % 10 == 0:
                print('iteration '+str(iteration))
        if j_curr == options['j_freq']: #Build jacobian
            J = np.zeros([num_v, num_p])
            #Use parallel
            all_scenarios = _getJScenarios(p_locator, curr, trx_params,
                                          options['delta'], default_scenario)
            for i, scenario in enumerate(all_scenarios):
                J[:,i] = np.reshape((_getJColumnROM(scenario) - N_guess_r)
                                    /scenario['dif'],(-1,))
            j_curr = 0
            
        #Calculate update
        damped_hessian = J.T @ J + l * np.diag(np.diag(J.T @ J))
        error_gradient = J.T @ np.reshape(N_true_r - N_guess_r,(-1))
        try:
            update = la.solve(damped_hessian, error_gradient)
        except:
            print(l)
            print(damped_hessian)
            print(error_gradient)
        
        #Create test parameters
        test = copy.deepcopy(curr)
        for i, elem in enumerate(p_locator):
            #Global param update
            if type(elem[1]) == int:
                test[elem[0]] = _checkBounds(test[elem[0]] + update[i],
                                            curr[elem[0]], elem[2])
            else:
                new = test[elem[0]][elem[1]].copy() + update[i]
                test[elem[0]][elem[1]] = _checkBounds(new,
                                                     curr[elem[0]][elem[1]].copy(),
                                                     elem[2])
        
        #Force all bounds for reduced parameters
        for elem in full_bounds:
            test[elem] = _forceBounds(test[elem], ROM['V'], full_bounds[elem],
                                     coeff_bounds[elem], initial_guess[elem])
        
        #Run with test parameters
        trx_params_test = _update_trx_params(test, trx_params)
        ops_test = lib.getOperators(test, ROM)
        N_test_r = call_dict[model](N0_r, ops_test, trx_params, t_true, dt)[0]
        SSE_test = np.sum((N_test_r - N_true_r)**2, axis = None) 
        
        if SSE_test < SSE:
            curr = copy.deepcopy(test)
            trx_params = copy.deepcopy(trx_params_test)
            ops = copy.deepcopy(ops_test)
            N_guess_r = N_test_r.copy()
            
            if l > 1e-15:
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
            
    return _updateParams(curr, params), stats, model, N_guess_r

########################## ROM gwMCMC calibration #############################
def calibrateRXDIF_gwMCMC_ROM(tumor, ROM, params, priors, dt = 0.5,
                              options = {}, parallel = False, plot = False):
    """
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
            - thin: save every 'thin' iteration from final chains
            - progress: show progress bar from emcee
            - samples: number of proposals to save
            - burnin: number of starting samples to discard, in fraction from 0 to 1
            - step_size: stretch step size
            - walker_factor: number of walkers = problem dimension * walker_factor
            
    parallel : boolean; default = False
        Should multicore processing be used.
    plot : boolean; default = False
        Should tracking variables be plot.
        
    Returns
    -------
    params - chain format
        Updated list of parameters with calibrated values. Samples for each parameter
    sampler : emcee sampler object
        Used to check statistics after the calibration
    """
    options_fields = ['thin','progress','samples','burnin','step_size','walker_factor']
    default_options = [1,    False,       1000,     0.1    , 1.6       , 4]
    for i, elem in enumerate(options_fields):
        if elem not in options:
            options[elem] = default_options[i]
            
    #Determine model to use based off data dimensions and prep data for calibration
    data = {}
    data['model'] = 'fwd.OP_RXDIF_wAC'
    data['N0_r'] = ROM['ReducedTumor']['N_r'][:,0]
    data['N_true_r'] = ROM['ReducedTumor']['N_r'][:,1:]
    data['t_true'] = tumor['t_scan'][1:]
    data['trx_params'] = {}
    data['trx_params']['t_trx'] = tumor['t_trx']
    data['ROM'] = ROM
    data['priors'] = priors
    data['dt'] = dt
    
    #Get size of ROM for reference
    n, r = ROM['V'].shape
    
    #Get calibrated params for MCMC, bounds for reduced parameters, fixed parameters
    required_params = ['d','k','alpha','beta_a','beta_c','sigma']
    found_params = []
    p_locator = {}
    full_bounds = {}
    coeff_bounds = {}
    fixed_params = {}
    initial_guess = {}
    names = []
    
    ndim = 0
    curr = 0
    for elem in params:
        found_params.append(elem)
        if params[elem].assignment == 'g':
            ndim += 1
            # p_locator.append({elem.name : 0})
            p_locator[elem] = list([curr])
            curr += 1
            names.append(elem)
        elif params[elem].assignment == 'r':
            ndim += r
            # p_locator.append({elem.name : list(range(r))})
            p_locator[elem] = list(range(curr, curr+r))
            curr += r
            full_bounds[elem] = params[elem].getBounds()
            coeff_bounds[elem] = params[elem].getCoeffBounds()
            #Set good initial guess for bound forcing
            k_target = np.mean(params[elem].getBounds())
            if k_target == 0:
                k_target = 1e-3
            k_test = np.zeros([n,1])
            k_test[np.nonzero(np.abs(np.sum(ROM['V'],axis=1)) > 1e-6)] = k_target
            k_r_test = np.squeeze(ROM['V'].T @ k_test)
            initial_guess[elem] = k_r_test
            for i in range(r):
                names.append(elem+str(i))
        elif params[elem].assignment == 'f':
            fixed_params[elem] = params[elem].get()
    for elem in required_params: #Cannot turn off proliferation right now
        if elem not in found_params:
            if elem == 'sigma':
                raise ValueError('Sigma must be fixed (non-zero) or calibrated for MCMC')
            else:
                fixed_params[elem] = 0      
                
    #Set data for passing into the log function. Stuff related to fixed parameters should be defined here
    #Data contains model, tumor info, timing, treatment details, default params dictionary with all fixed parameters
    data['default_params'] = {}
    for elem in fixed_params:
        data['default_params'][elem] = fixed_params[elem]
    data['full_bounds'] = full_bounds
    
    #Set up walker start points
    nwalkers = ndim*options['walker_factor']
    init = np.zeros((nwalkers, ndim))
    for i in range(nwalkers):
        for elem in p_locator:
            if len(p_locator[elem]) == 1:
                #Get single random variable from prior
                init[i,p_locator[elem]] = priors[elem].rvs()
            else:
                #Get random variable from each prior
                temp = np.zeros(r)
                for j in range(r):
                    temp[j] = priors[elem+str(j)].rvs()
                #force initial guess to be within bounds
                temp = _forceBounds(temp, ROM['V'], full_bounds[elem],
                                    coeff_bounds[elem], x0 = None)
                init[i, p_locator[elem]] = temp
    
    #Initialize and run sampler
    nsteps = int(np.ceil(options['samples']/nwalkers))
    s = options['step_size']
    moves = [(emcee.moves.StretchMove(s), 0.8), (emcee.moves.WalkMove(), 0.2)]
    if parallel == True:
        with mp.Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_posterior, 
                                            args = (data,), 
                                            parameter_names = p_locator,  
                                            pool = pool, moves = moves)
            sampler.run_mcmc(init, nsteps, progress = options['progress'])
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_posterior, 
                                        args = (data,), 
                                        parameter_names = p_locator, 
                                        moves = moves)
        sampler.run_mcmc(init, nsteps, progress = options['progress'])
         
    if plot:
        fig, ax = plt.subplots(ndim, 1, figsize = (8, ndim), layout = 'constrained')
        full_chains = sampler.get_chain()
        for i in range(ndim):
            for j in range(nwalkers):
                ax[i].plot(np.squeeze(full_chains[:,j,i]),color = (0,0,0,0.2))
                ax[i].set_ylabel(names[i])
                ax[i].set_xlabel('Step')
        
    chains = sampler.get_chain(discard = int(np.ceil(options['burnin']*nsteps)) ,
                               thin = options['thin'], flat = True)
    
    return _unpackChains(params, chains, p_locator), sampler, data['model']

########################### ROM pyABC calibration #############################
def calibrateRXDIF_ABC_ROM(tumor, ROM, params, priors, dt = 0.5,
                              options = {}, plot = False):    
    #Prep measured data and get simulation details
    options_fields = ['n_pops','pop_size','thin','burnin','epsilon','distance']
    default_options = [5, 1000, 1, 0.1, 0.2,'SSE']
    for i, elem in enumerate(options_fields):
        if elem not in options:
            options[elem] = default_options[i]
            
    distance = _distance(options['distance'])
            
    #Determine model to use based off data dimensions and prep data for calibration
    data = {}
    data['model'] = 'fwd.OP_RXDIF_wAC'
    data['N0_r'] = ROM['ReducedTumor']['N_r'][:,0]
    data['N_true_r'] = ROM['ReducedTumor']['N_r'][:,1:]
    data['t_true'] = tumor['t_scan'][1:]
    data['trx_params'] = {}
    data['trx_params']['t_trx'] = tumor['t_trx']
    data['ROM'] = ROM
    data['dt'] = dt
    
    #Get size of ROM for reference
    n, r = ROM['V'].shape
    
    #Get what parameters are calibrated and how they are calibrated
    #Currently only works with global d, alpha, beta and reduced k
    required_params = ['d','k','alpha','beta_a','beta_c','sigma']
    found_params = []
    fixed_params = {}
    p_locator = {}
    bounds = {}
    for elem in params:
        found_params.append(elem) 
        if params[elem].assignment == 'f':
            fixed_params[elem] = params[elem].get()
        elif params[elem].assignment == 'g':
            p_locator[elem] = [elem]
        elif params[elem].assignment == 'r':
            #store variable name with mode number attached for each
            temp = [elem+str(x) for x in range(r)]
            p_locator[elem] = temp
            bounds[elem] = params[elem].getBounds()
            
    if options['epsilon'] == 'calibrated':
        #Only used for distance epsilon
        Model_r = calibrateRXDIF_LM_ROM(tumor, ROM, params)[3]
        options['epsilon'] = distance({'data':Model_r}, {'data':data['N_true_r']})
        print('Epsilon = '+str(options['epsilon']))
    for elem in required_params: #Cannot turn off proliferation right now
        if elem not in found_params:
            fixed_params[elem] = 0
                
    data['default_params'] = {}
    for elem in fixed_params:
        data['default_params'][elem] = fixed_params[elem]
                
    constrainedPriors = _ConstrainedPrior(priors, ROM['V'], bounds, p_locator)  
    
    sampler = pyabc.ABCSMC(_model(data, p_locator), constrainedPriors, distance,
                            population_size = options['pop_size'], sampler = pyabc.sampler.SingleCoreSampler())
    
    db_path = os.path.join(tempfile.gettempdir(), "ABC_test.db")
    sampler.new("sqlite:///" + db_path, observed_sum_stat = {"data": data['N_true_r']})
    history = sampler.run(minimum_epsilon=options['epsilon'], max_nr_populations=options['n_pops'])
    
    return _unpackSamplesABC(params, history.get_distribution()[0], p_locator), data['model']

######################## Priors for MCMC calibration ##########################
def generatePriors(params):
    priors = {}
    for elem in params:
        if params[elem].assignment != 'f':
            if elem == 'd':
                priors['d'] = stats.truncnorm((params[elem].getBounds()[0] - 5e-4)/2.5e-4,
                                              (params[elem].getBounds()[1] - 5e-4)/2.5e-4,
                                              loc = 5e-4, scale = 2.5e-4)
            if elem == 'alpha':
                priors['alpha'] = stats.uniform(params[elem].getBounds()[0],
                                                params[elem].getBounds()[1] 
                                                - params[elem].getBounds()[0])
            if elem == 'beta_a':
                priors['beta_a'] = stats.truncnorm((params[elem].getBounds()[0] - 0.60)/0.0625,
                                                   (params[elem].getBounds()[1] - 0.60)/0.0625,
                                                   loc = 0.60, scale = 0.0625)
            if elem == 'beta_c':
                priors['beta_c'] = stats.truncnorm((params[elem].getBounds()[0] - 3.25)/0.5625,
                                                   (params[elem].getBounds()[1] - 3.25)/0.5625,
                                                   loc = 3.25, scale = 0.5625)
            if elem == 'k':
                if params[elem].assignment == 'g':
                    priors['k'] = stats.uniform(params[elem].getBounds()[0],
                                                params[elem].getBounds()[1] - params[elem].getBounds()[0])
                elif params[elem].assignment == 'r':
                    bounds = params[elem].getCoeffBounds()
                    for j in range(bounds.shape[0]):
                        priors['k'+str(j)] = stats.uniform(bounds[j,0],
                                                           bounds[j,1] - bounds[j,0])
            if elem == 'sigma':
                if params[elem].assignment == 'g':
                    priors['sigma'] = stats.uniform(params[elem].getBounds()[0],
                                                    params[elem].getBounds()[1] 
                                                    - params[elem].getBounds()[0])
    return priors

################################ Internal use #################################
def _atleast_4d(arr):
    if arr.nidm < 4:
        return np.expand_dims(np.atleast_3d(arr),axis=3)
    else:
        return arr

def _update_trx_params(curr_params, trx_params):
    trx_params_new = copy.deepcopy(trx_params)
    trx_params_new['beta'] = np.array([curr_params['beta_a'], 
                                       curr_params['beta_c']])
    return trx_params_new

#Required for jacobian building
def _getJScenarios(p_locator, curr_params, trx_params, delta, default_scenario):
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
        temp['trx_params'] = _update_trx_params(temp['params'], 
                                               default_scenario['trx_params'])
        scenarios.append(temp.copy())
        del temp
    return scenarios

def _getJColumn(scenario):
        N = call_dict[scenario['model']](scenario['N0'], scenario['params']['k'], 
                             scenario['params']['d'], scenario['params']['alpha'],
                             scenario['trx_params'], scenario['t_true'],
                             scenario['h'], scenario['dt'], scenario['bcs'])[0]
        return N
    
def _checkBounds(new, old, bounds):
    if new < bounds[0]:
        new = old - (old - bounds[0])/2
    elif new > bounds[1]:
        new = old + (bounds[1] - old)/2
    return new

def _updateParams(calibrated, params):
    new_params = copy.deepcopy(params)
    for elem in new_params:
        try:
            new_params[elem].update(np.reshape(calibrated[elem].copy(),(-1)))
        except:
            new_params[elem].update(np.reshape(calibrated[elem],(-1)))
    return new_params

######################### Internal use for ROM ################################
def _getJColumnROM(scenario):        
    N = call_dict[scenario['model']](scenario['N0_r'], 
                                     lib.getOperators(scenario['params'], scenario['ROM']), 
                                     scenario['trx_params'], 
                                     scenario['t_true'], scenario['dt'])[0]
    
    return N

def _forceBounds(curr_param, V, bounds, coeff_bounds, x0 = None):
    test = V @ curr_param
    if (len(np.nonzero((test < bounds[0]) & (abs(test) > 1e-6))) != 0 
        or len(np.nonzero(test > bounds[1])) != 0):
        
        indices = np.nonzero(abs(test)>1e-6)[0]
        A = np.concatenate((V[np.s_[indices,:]], -1*V[np.s_[indices,:]]),
                           axis = 0)
        B = np.squeeze(np.concatenate((bounds[1]*np.ones([len(indices),1]),
                                       -1*bounds[0]*np.ones([len(indices),1])),
                                      axis = 0))
        if x0 is None:
            x0 =  np.mean(coeff_bounds, axis = 1)
            
        lincon = sciop.LinearConstraint(A, -np.inf, B)
        result = sciop.minimize(lambda x: np.linalg.norm(V@x - test, ord = 2),
                                x0, constraints = lincon, method = 'COBYLA')
        curr_param = result.x.copy()
        
    return curr_param

######################### Internal use for MCMC ###############################
def _log_prior(p, priors, V, bounds, tol = 1+1e-2):
    prior = 1
    names = []
    for elem in p:
        if len(p[elem]) == 1:
        # if p[elem].size == 1:
            prior *= priors[elem].pdf(p[elem])
        else:
            names.append(elem)
            for i in range(len(p[elem])):
            # for i in range(p[elem].size):
                prior *= priors[elem+str(i)].pdf(p[elem][i])           
    # ensure reconstructed parameters are in bounds
    for elem in names:
        recon = V@p[elem]
        indices = np.nonzero(abs(recon)>1e-3)[0]
        if (np.nonzero((recon[indices] < bounds[elem][0]/tol) 
                       | (recon[indices] > bounds[elem][1]*tol))[0].size != 0):
            return -np.inf
    if prior == 0:
        return -np.inf
    else:
        return np.log(prior)

def _log_posterior(p, data):
    prior = _log_prior(p, data['priors'], data['ROM']['V'], data['full_bounds'])
    if not np.isfinite(prior):
        return -np.inf
    #Get operators from current p
    curr_params = copy.deepcopy(data['default_params'])
    for elem in p:
        curr_params[elem] = p[elem].copy()
    ops = lib.getOperators(curr_params, data['ROM'])
    trx_params = copy.deepcopy(data['trx_params'])
    trx_params = _update_trx_params(curr_params, trx_params)
    #Solve model
    sim = call_dict[data['model']](data['N0_r'], ops, trx_params, 
                                   data['t_true'], data['dt'])[0]
    #Calculate likelihood
    residuals = data['N_true_r'] - sim
    ll = -0.5 * np.sum((residuals/curr_params['sigma'])**2 
                        + np.log(2*np.pi) 
                        + np.log(curr_params['sigma']**2))
    return prior + ll

def _unpackChains(params, chains, p_locator):
    new_params = copy.deepcopy(params)
    for elem in new_params:
        for i, loc_elem in enumerate(p_locator):
            if elem == loc_elem:
                indices = p_locator[loc_elem]
                new_params[elem].update(chains[:,indices].T.copy())
                
    return new_params

######################### Internal use for pyABC ##############################
class _ConstrainedPrior(pyabc.DistributionBase):
    def __init__(self,priors,V,bounds,p_locator):
        self.priors = {}
        for elem in priors:
            self.priors[elem] = priors[elem]
        self.V = V
        self.bounds = bounds
        self.p_locator = p_locator
    
    def rvs(self, *args, **kwargs):
        while True:
            params = {}
            local = {}
            for elem in self.p_locator:
                if len(self.p_locator[elem]) == 1:
                    params[elem] = self.priors[elem].rvs()
                else:
                    temp = []
                    for curr_prior in self.p_locator[elem]:
                        params[curr_prior] = self.priors[curr_prior].rvs()
                        temp.append(params[curr_prior])
                    local[elem] = np.array(temp).copy()
            good_set = 1
            
            for elem in local:
                recon = self.V@local[elem]
                indices = np.nonzero(abs(recon)>1e-3)[0]
                tol = 1+1e-2
                if (np.nonzero((recon[indices] < self.bounds[elem][0]/tol) 
                               | (recon[indices] > self.bounds[elem][1]*tol))[0].size != 0):
                    good_set = 0
            if good_set == 1:
                return pyabc.Parameter(params)
    
    def pdf(self, x):
        prior = 1
        for elem in self.priors:
            prior *= self.priors[elem].pdf(x[elem])
        return prior 
        
class _model:
    def __init__(self, data, p_locator):
        self.curr_params = copy.deepcopy(data['default_params'])
        self.data = data
        self.p_locator = p_locator        
        
    def __call__(self, p):
        k_names = [];
        for elem in p:
            if 'k' in elem:
                k_names.append(elem)
            else:
                self.curr_params[elem] = p[elem]
        self.curr_params['k'] = np.array(itemgetter(*k_names)(p))
        
        for elem in self.p_locator:
            if len(self.p_locator[elem]) == 1:
                self.curr_params[elem] = p[elem]
            else:
                self.curr_params[elem] = np.array(itemgetter(*self.p_locator[elem])(p))
        
        ops = lib.getOperators(self.curr_params, self.data['ROM'])
        trx_params = copy.deepcopy(self.data['trx_params'])
        trx_params = _update_trx_params(self.curr_params, trx_params)
        sim = call_dict[self.data['model']](self.data['N0_r'], ops, trx_params, 
                                       self.data['t_true'], self.data['dt'])[0]
        return {'data':sim} 
        
class _distance:
    def __init__(self, option):
        self.option = option
        
    def __call__(self, x, y):
        if self.option == 'SSE':
            return np.sum((x['data'] - y['data'])**2)
        elif self.option == 'MSE':
            return np.mean((x['data'] - y['data'])**2)
        elif self.option == 'SAE':
            return np.sum(np.abs(x['data'] - y['data']))
        elif self.option == 'MAE':
            return np.mean(np.abs(x['data'] - y['data']))

def _unpackSamplesABC(params, datastore, p_assignment):
    new_params = copy.deepcopy(params)
    for elem in p_assignment:
        new_params[elem].update(datastore.loc[:,p_assignment[elem]]\
                                .to_numpy().T.copy())
    return new_params


