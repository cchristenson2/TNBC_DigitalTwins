import numpy as np
import os
import scipy.linalg as la

import LoadData as ld
import ForwardModels as fwd
import ROM as ROM
import Library as lib

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

###############################################################################
# Internal for ROM updating
def getOperators(curr_params, ROM):
    operators = {}
    for elem in curr_params:
        if elem == 'd':
            operators['A'] = lib.interpolateGlobal(ROM['Library']['A'], curr_params[elem])
        elif elem == 'k':
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

if __name__ == '__main__':
    #Set paths
    home = os.path.dirname(os.getcwd())
    datapath = home + '\Data\PatientData_ungrouped\\'
    #Get tumor information in folder
    files = os.listdir(datapath)
    
    #Load the first patient
    print(files[0])
    tumor = ld.LoadTumor_mat(datapath + files[0], crop2D = 'false', split = 2)
    
    bounds = {}
    bounds['d'] = np.array([1e-6, 1e-3])
    bounds['k'] = np.array([1e-6, 0.1])
    bounds['alpha'] = np.array([1e-6, 0.8])
    
    ROM = ROM.constructROM_RXDIF(tumor, bounds)
    
    params = ([fwd.Parameter('d','g'), fwd.ReducedParameter('k','r',ROM['V']), fwd.Parameter('alpha','g'),
               fwd.Parameter('beta_a','f'), fwd.Parameter('beta_c','f')])
    
    params[0].setBounds(np.array([1e-6,1e-3]))
    params[1].setBounds(ROM['Library']['B']['coeff_bounds'])
    params[2].setBounds(np.array([1e-6,0.8]))
    
    
    
    dt = 0.5
    options = {'max_it':20}
    parallel = 'true'
     
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
    
    required_params = ['d','alpha','beta_a','beta_c']
    found_params = []
    curr = {}
    for elem in params:
        found_params.append(elem.name)
        if elem.value == None:
            if elem.assignment == 'r':
                curr[elem.name] = np.mean(elem.getBounds(),axis=1)
            else:
                curr[elem.name] = np.mean(elem.getBounds())
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
             'k_track': list([curr['k']]),'Lambda_track': list([l]),
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
        test = curr.copy()
        for i, elem in enumerate(p_locator):
            #Global param update
            if type(elem[1]) == int:
                test[elem[0]] = checkBounds(test[elem[0]] + update[i], curr[elem[0]], elem[2])
            else:
                test[elem[0]][elem[1]] = checkBounds(test[elem[0]][elem[1]] + update[i],
                                                     curr[elem[0]][elem[1]], elem[2]).copy()
                test[elem[0]][elem[1]] = forceBounds(test[elem[0]][elem[1]], 0)
    
        
        #Run with test parameters
        trx_params_test = update_trx_params(test, trx_params)
        ops_test = getOperators(test, ROM)
        N_test_r = call_dict[model](N0_r, ops_test, trx_params, t_true, dt)[0]
        SSE_test = np.sum((N_test_r - N_true_r)**2, axis = None) 
        
        if SSE_test < SSE:
            curr = test.copy()
            ops = ops_test.copy()
            trx_params = trx_params_test.copy()
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