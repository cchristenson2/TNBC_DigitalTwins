# Optimize.py
"""
Created on Fri May 24 10:42:40 2024

@author: Chase Christenson
"""

import numpy as np
import scipy.optimize as sciop
from scipy import integrate

################################## Setup ######################################
def problemSetup_cellMin(tumor, simulations, objectives = [], constraints = [],
                         interval = 1, separate_doses = False, metric = 'mean',
                         weights = None, max_dose = None, threshold = 0.0,
                         norm = True, tol = 1e-3, dt = 0.5):
    #Get last measured time
    t_sim_end = tumor['t_scan'][-1]
    #Get last prediction time
    t_pred_end = tumor['Future t_scan'][-1]
    
    t_trx_soc = tumor['t_trx'][tumor['t_trx']<t_sim_end]
    t_trx_future = tumor['t_trx'][tumor['t_trx']>=t_sim_end]
    
    # stop_days = np.concatenate((t_trx_future, [t_pred_end]), axis = 0)
    # potential_days = np.array([])
    # for i, elem in enumerate(t_trx_future):
    #     potential_days = np.concatenate(
    #         (potential_days,np.arange(elem+1, np.min([elem+13, stop_days[i+1]])+1,
    #                                   step = interval)), axis = 0)
    potential_days = np.arange(t_trx_future[0], t_pred_end, step = interval)
    
    if max_dose == None:
        max_dose = t_trx_future.size
        
    #Set up constraint and objective functions
    valid_objectives = {'final_cells':{'func':qoi_TotalCells,'args':['cell_tc'],'kwargs':{},'name':'final_cells'},
                        'max_cells':{'func':qoi_MaxCells,'args':['cell_tc',int(t_sim_end/dt)],'kwargs':{},'name':'final_cells'},
                        #Add more objective calls here
                        }
    call_objectives = []
    for elem in objectives:
        call_objectives.append(valid_objectives[elem])
        #Overwrite default kwargs based on inputs
    if not call_objectives:
        for elem in valid_objectives:
            call_objectives.append(valid_objectives[elem])
            
    valid_constraints = {'cummulative_concentration':{'func':con_CummulativeConcentration,'args':'concentrations','kwargs':{'dt':dt},'name':'cummulative_concentration'},
                         'max_concentration':{'func':con_MaxConcentration, 'args':'concentrations','kwargs':{},'name':'max_concentration'},
                         'ld50_toxicity':{'func':con_Toxicity_ld50,'args':'concentrations','kwargs':{'dt':dt},'name':'ld50_toxicity'},
                         #Add more nonlinear constraint calls here
                         }
    call_constraints = []
    lin_constraints = []
    for elem in constraints:
        if elem != 'total_dose':
            call_constraints.append(valid_constraints[elem])
        else:
            lin_constraints.append({'type':'eq', 'fun':lambda x: max_dose - sum(x)})
    if not call_constraints:
        for elem in valid_constraints:
            if elem != 'total_dose':
                call_constraints.append(valid_constraints[elem])
            else:
                lin_constraints.append({'type':'eq', 'fun':lambda x: max_dose - sum(x)})
        
    #Evaluate objectives for SOC protocol
    soc_obj = {}
    for i in range(len(call_objectives)):
        soc_obj[call_objectives[i]['name']] = _call_objectiveFunc(call_objectives[i], simulations, metric)
        
    #Evaluate constraints for SOC protocol    
    soc_con = {}
    for i in range(len(call_constraints)):
        soc_con[call_constraints[i]['name']] = _call_constraintFunc(call_constraints[i], simulations, metric)
    
    if weights == None:
        w = np.ones(len(call_objectives))
    else:
        w = weights
        
    #Generate initial guess that satisifies constraints
    if separate_doses == True:
        doses_soc = np.ones((t_trx_soc.size, 2))
    else:
        doses_soc = np.ones((t_trx_soc.size))
        
    problem = {'t_trx_soc':t_trx_soc, 'doses_soc': doses_soc, 
               'potential_days':potential_days,
               'objectives':call_objectives,
               'lin-constraints':lin_constraints, 'nonlin-constraints':call_constraints,
               'soc_obj':soc_obj,'soc_con':soc_con,'metric':metric,'weights':w,
               'threshold':threshold, 
               'max_dose':max_dose, 'separate_doses':separate_doses,
               'norm':norm,'contol':tol,'dt':dt,
               't_sim_end':t_sim_end,'t_pred_end':t_pred_end}
    
    return problem

def randomizeInitialGuess(twin, problem):
    run = 1
    while run == 1:
        doses_guess = generateGuess(problem['potential_days'].size, problem['max_dose'], problem['separate_doses'])
        test = constrainedObjective(doses_guess, problem, twin)
        if test != np.inf:
            run = 0
    return doses_guess
            
def generateGuess(n, max_dose, separate_doses):
    if separate_doses == True:
        x = np.random.rand(n,2)
    else:
        x = np.random.rand(n)
    return x * (max_dose / np.sum(x, axis = 0))

############################### Constraints ###################################
#Plan based constraints
#These can be solved with a linear inequality
def con_TotalDose(trx_params):
    if 'doses' not in trx_params.keys():
        return trx_params['t_trx'].size
    else:
        return sum(trx_params['doses'])

#Concentration based constraints
def con_CummulativeConcentration(concentrations, dt = 0.5):
    return integrate.cumtrapz(concentrations, dx = dt, axis = 0)[-1,:]
    
def con_MaxConcentration(concentrations):
    return np.amax(concentrations, axis = 0)

def con_Toxicity_ld50(concentrations, dt = 0.5):
    #Get list of ld50s to check
    check = np.linspace(0, np.amax(concentrations, axis = 0), 10)
    
    toxic = np.array([])
    for i in range(check.shape[0]):
        a = np.argwhere(concentrations[:,0] >= check[i,0]).size * dt
        b = np.argwhere(concentrations[:,1] >= check[i,1]).size * dt
        if i == 0:
            toxic = np.hstack((toxic, np.array([a,b])))
        else:
            toxic = np.vstack((toxic, np.array([a,b])))
        
    return integrate.cumtrapz(toxic, x = check, axis = 0)[-1,:]

############################### Objectives ####################################
def qoi_TotalCells(timecourse):
    return timecourse[-1]

def qoi_MaxCells(timecourse, index):
    return max(timecourse[index:])

############################### Optimizer #####################################
def constrainedObjective(x, problem, twin, tol = 1e-3, norm = True):
    """
    x = current test dosage. Either n x 2 or n x 1 if the drugs are optimized together
    problem = tells which functions to run
    twin = holds parameter details from calibration and prediction functions
    """
    new_days = np.concatenate((problem['t_trx_soc'], problem['potential_days']),axis = 0)
    new_doses = np.concatenate((problem['doses_soc'], x), axis = 0)
    trx_params = {'t_trx': new_days, 'doses': new_doses}
    simulations = twin.predict(treatment = trx_params, threshold = problem['threshold'])
    
    # test_con = {}
    for i in range(len(problem['nonlin-constraints'])):
        # test_con[problem['nonlin-constraints'][i]['name']] = _call_constraintFunc(problem['nonlin-constraints'][i], simulations, problem['metric'])
        test = _call_constraintFunc(problem['nonlin-constraints'][i], simulations, problem['metric'])
        #If worse break and return infinity
        check = problem['soc_con'][problem['nonlin-constraints'][i]['name']]
        # print(problem['nonlin-constraints'][i]['name'])
        # print(check)
        if (test >= (check + tol)).any():
            # print(problem['nonlin-constraints'][i]['name'] + ' violated')
            return np.inf
        
    obj = 0
    for i in range(len(problem['objectives'])):
        if norm == False:
            n = 1
        else:
            n = problem['soc_obj'][problem['objectives'][i]['name']]
        obj += (_call_objectiveFunc(problem['objectives'][i], simulations, problem['metric']) / n)**2 * problem['weights'][i]
    return obj

def objective(x, problem, twin, norm = True):
    new_days = np.concatenate((problem['t_trx_soc'], problem['potential_days']),axis = 0)
    new_doses = np.concatenate((problem['doses_soc'], x), axis = 0)
    trx_params = {'t_trx': new_days, 'doses': new_doses}
    simulations = twin.predict(treatment = trx_params, threshold = problem['threshold'])
    obj = 0
    for i in range(len(problem['objectives'])):
        if norm == False:
            n = 1
        else:
            n = problem['soc_obj'][problem['objectives'][i]['name']]
        obj += (_call_objectiveFunc(problem['objectives'][i], simulations, problem['metric']) / n)**2 * problem['weights'][i]
    return obj
    

def constraints(x, problem, twin, tol = 1e-3):
    new_days = np.concatenate((problem['t_trx_soc'], problem['potential_days']),axis = 0)
    new_doses = np.concatenate((problem['doses_soc'], x), axis = 0)
    trx_params = {'t_trx': new_days, 'doses': new_doses}
    simulations = twin.predict(treatment = trx_params, threshold = problem['threshold'])
    A = []
    B = []
    for i in range(len(problem['nonlin-constraints'])):
        test = _call_constraintFunc(problem['nonlin-constraints'][i], simulations, problem['metric'])
        check = problem['soc_con'][problem['nonlin-constraints'][i]['name']]
        try:
            A = np.concatenate((A, test), axis = 0)
            B = np.concatenate((B, check), axis = 0)
        except:
            A = test.copy()
            B = check.copy()
            
    return (B+tol) - A
 
################################ Cache ########################################
class CachedModel:
    def __init__(self, twin, problem):
        self.cache = {}
        self.twin = twin
        self.problem = problem
        
    def in_cache(self, x):
        return str(x) in self.cache
    
    def updateSimulation(self, x):
        if len(self.cache.keys()) > 10:
            self.cache.clear()
        
        new_days = np.concatenate((self.problem['t_trx_soc'], self.problem['potential_days']),axis = 0)
        new_doses = np.concatenate((self.problem['doses_soc'], x), axis = 0)
        trx_params = {'t_trx': new_days, 'doses': new_doses}
        self.cache[str(x)] = self.twin.predict(treatment = trx_params, threshold = self.problem['threshold'])
        
    def objective(self, x):
        if not self.in_cache(x):
            self.updateSimulation(x)
        simulations = self.cache[str(x)]       
        obj = 0
        for i in range(len(self.problem['objectives'])):
            if self.problem['norm'] == False:
                n = 1
            else:
                n = self.problem['soc_obj'][self.problem['objectives'][i]['name']]
            obj += (_call_objectiveFunc(self.problem['objectives'][i], simulations, self.problem['metric']) / n)**2 * self.problem['weights'][i]
        return obj
    
    def constraints(self, x):
        if not self.in_cache(x):
            self.updateSimulation(x)
        simulations = self.cache[str(x)]  
        A = []
        B = []
        for i in range(len(self.problem['nonlin-constraints'])):
            test = _call_constraintFunc(self.problem['nonlin-constraints'][i], simulations, self.problem['metric'])
            check = self.problem['soc_con'][self.problem['nonlin-constraints'][i]['name']]
            try:
                A = np.concatenate((A, test), axis = 0)
                B = np.concatenate((B, check), axis = 0)
            except:
                A = test.copy()
                B = check.copy()
                
        return (B+self.problem['contol']) - A
    
############################### Internal ######################################
def _call_objectiveFunc(function, simulations, metric):
    string = function['args'][0]
    if string == 'cell_tc':
        data = _evalMetric(simulations['cell_tc'], metric, 'timecourse')
    elif string == 'volume_tc':
        data = _evalMetric(simulations['volume_tc'], metric, 'timecourse')
    if len(function['args']) > 1:
        return function['func'](data, *function['args'][1:], **function['kwargs'])
    else:
        return function['func'](data, **function['kwargs'])
    
def _call_constraintFunc(function, simulations, metric):
    string = function['args']
    if string == 'trx_params':
        return function['func'](simulations['trx_params'], **function['kwargs'])
    elif string == 'concentrations':
        data = _evalMetric(simulations['drug_tc'], metric, 'concentration')
        return function['func'](data, **function['kwargs'])
    
def _evalMetric(data, metric, datatype):
    if datatype == 'timecourse':
        get_axis = 1
        if data.shape[1]>1:
            samples = True
        else:
            samples = False
    elif datatype == 'concentration':
        get_axis = 0
        if len(data) > 1:
            samples = True
        else:
            samples = False
        data = np.squeeze(np.array(data))
    if samples == True:
        if metric == 'mean':
            return np.mean(data, axis = get_axis)
        #Add more metrics here (e.g., max, median, quantiles, super quantiles)
    else:
        return data
    
def getDrugConcentrations(trx_params, dt = 0.5, tol = 0.01):
    """
    Run until all deliveries have passed and dosage is approximately zero
    0.01 default tolerance
    """
    beta, nt_trx, delivs, drugs, doses = _setupTRX(trx_params, dt)
    run = 1
    step = 1
    while run == 1:
        delivs  = _updateDosing(step, nt_trx, delivs, dt)
        #Get current dosage
        curr_drugs = np.zeros((1,2))
        for n in range(delivs.size):
            curr_drugs[0,0] = curr_drugs[0,0] + doses[n,0]*np.exp(-1*beta[0]*delivs[n])
            curr_drugs[0,1] = curr_drugs[0,1] + doses[n,1]*np.exp(-1*beta[1]*delivs[n])
        # print(drugs.shape)
        # print(curr_drugs.shape)
        drugs = np.append(drugs, curr_drugs, axis = 0)
        
        if (np.max(curr_drugs) <= tol) and (delivs.size >= nt_trx.size):
            run = 0
        step += 1      
    return drugs        
        
def _updateDosing(step, nt_trx, delivs, dt):
    #Increment all current treatments by dt
    if delivs.size > 0:
        delivs = delivs + dt
    if delivs.size < nt_trx.size:
        if step - 1 >= nt_trx[delivs.size]: #Delivery occurred at previous step
            delivs = np.append(delivs,0)  
    return delivs
  
def _setupTRX(trx_params, dt):
    if np.isscalar(trx_params.get('beta')):
        beta = np.array([trx_params.get('beta'), trx_params.get('beta')])
    else:
        beta = trx_params.get('beta')  
        
    #Setup treatment matrices
    nt_trx = trx_params.get('t_trx') / dt #Indices of treatment times
    delivs = np.array([]) #Storage for time since deliveries that have passed
    drugs = np.zeros((1,2))
    
    #Check if doses are specified
    if 'doses' in trx_params:
        #Check if each drug gets a different dosage at each time
        doses = trx_params.get('doses')
        if doses.ndim == 1:
            doses = np.expand_dims(doses,1)
            doses = np.append(doses,doses,1)
    else: #All treatments get normalized dose of 1
        doses = np.ones([nt_trx.size,2])  

    return beta, nt_trx, delivs, drugs, doses  

