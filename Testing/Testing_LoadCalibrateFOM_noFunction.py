import numpy as np
import os

import LoadData as ld
import ForwardModels as fwd
# import Calibrations as cal
import scipy.ndimage as ndi
import scipy.sparse.linalg as sla
import scipy.linalg as la
import concurrent.futures

call_dict = {
    'fwd.RXDIF_2D_wAC': fwd.RXDIF_2D_wAC,
    'fwd.RXDIF_3D_wAC': fwd.RXDIF_3D_wAC,
    'fwd.OP_RXDIF_wAC': fwd.OP_RXDIF_wAC}

# Internal use only
def atleast_4d(arr):
    if arr.nidm < 4:
        return np.expand_dims(np.atleast_3d(arr),axis=3)
    else:
        return arr
    
def update_trx_params(curr_params, trx_params):
    trx_params_new = trx_params.copy()
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
            # if i == 1:
            #     print(temp['params'][elem[0]][13,12])
            arr = temp['params'][elem[0]].copy()
            initial = arr[elem[1]]
            new = arr[elem[1]].copy() * delta
            arr[elem[1]] = new
            temp['params'][elem[0]] = arr
            temp['dif'] = new - initial
            # if i == 1:
            #     print(temp['params'][elem[0]][13,12])
        temp['trx_params'] = update_trx_params(temp['params'], default_scenario['trx_params'])
        scenarios.append(temp.copy())
        del temp
        
    return scenarios

def getJColumn(scenario):
        N0 = scenario['N0']
        curr = scenario['params']
        trx_params = scenario['trx_params']
        t_true = scenario['t_true']
        dt = scenario['dt']
        model = scenario['model']
        bcs = scenario['bcs']
        h = scenario['h']
        
        N = call_dict[model](N0, curr['k'], curr['d'], curr['alpha'], trx_params, t_true, h, dt, bcs)[0]
        
        return N
    
def checkBounds(new, old, bounds):
    if new < bounds[0]:
        new = old - (old - bounds[0])/2
    elif new > bounds[1]:
        new = old + (bounds[1] - old)/2
    return new
            
            

if __name__ == '__main__':
    #Set paths
    home = os.path.dirname(os.getcwd())
    datapath = home + '\Data\PatientData_ungrouped\\'
    #Get tumor information in folder
    files = os.listdir(datapath)
    
    #Load the first patient
    tumor = ld.LoadTumor_mat(datapath + files[0], crop2D = 'true')
    tumor['N'] = tumor['N'][:,:,(0,1)]
    tumor['t_scan'] = tumor['t_scan'][:-1]
    
    #Set up call to calibrate with FOM, 2D since 3D takes a long time
    params = ([fwd.Parameter('d','g'), fwd.Parameter('k','l'), fwd.Parameter('alpha','g'),
               fwd.Parameter('beta_a','f'), fwd.Parameter('beta_c','f')])
    
    params[0].setBounds(np.array([1e-6,1e-3]))
    params[1].setBounds(np.array([1e-6,0.1]))
    params[2].setBounds(np.array([1e-6,0.8]))
    
    
    #Call calibration
    # params = cal.calibrateRXDIF_LM(tumor, params)
    
    
    dt = 0.5
    options = {'max_it':20}
    parallel = 'true'
    
    options_fields = ['e_tol','e_conv','max_it','delta','pass','fail','lambda','j_freq']
    default_options = [1e-6,   1e-7,    500,     1.001,  7,     9,     1,       1]
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
    N_guess = call_dict[model](N0, curr['k'], curr['d'], curr['alpha'], trx_params, t_true, tumor['h'], dt, tumor['bcs'])[0]
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
    
    stats = {'d_track': [curr['d']], 'alpha_track': [curr['alpha']], 'kp_track': [curr['k']],'Lambda_track': [l]}
    
    while iteration <= options['max_it'] and SSE > options['e_tol']:    
        if iteration % 10 == 0:
            print(iteration)
        if j_curr == options['j_freq']: #Build jacobian
            print('building jacobian')
            J = np.zeros([num_v, num_p])
            #Use parallel
            all_scenarios = getJScenarios(p_locator, curr, trx_params, options['delta'], default_scenario)
            if parallel == 'true':
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
        # update, _ = sla.bicgstab(damped_hessian, error_gradient, tol = 1e-10, maxiter = 50)
        update, _ = la.solve(damped_hessian, error_gradient)
        
        #Create test parameters
        test = curr.copy()
        for i, elem in enumerate(p_locator):
            #Global param update
            if elem[1] == 0:
                test[elem[0]] = checkBounds(test[elem[0]] + update[i], curr[elem[0]], elem[2])
            else:
                test[elem[0]][elem[1]] = checkBounds(test[elem[0]][elem[1]] + update[i], curr[elem[0]][elem[1]], elem[2])  

        
        #Run with test parameters
        trx_params_test = update_trx_params(test, trx_params)
        N_test = call_dict[model](N0, test['k'], test['d'], test['alpha'], trx_params_test, t_true, tumor['h'], dt, tumor['bcs'])[0]
        SSE_test = np.sum((N_test - N_true)**2, axis = None) 
        
        if SSE_test < SSE:
            curr = test.copy()
            trx_params = trx_params_test.copy()
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
            
            
        iteration += 1
        stats['d_track'].append(curr['d'].copy())
        stats['kp_track'].append(curr['k'].copy())
        stats['alpha_track'].append(curr['alpha'].copy())
        stats['Lambda_track'].append(l)