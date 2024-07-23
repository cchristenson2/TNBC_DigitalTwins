# Optimize.py
"""
Created on Fri May 24 10:42:40 2024

@author: Chase Christenson
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

################################## Setup ######################################
def problemSetup_cellMin(twin, objectives = [], constraints = [],
                         interval = 1, separate_doses = False, metric = 'mean',
                         weights = None, max_dose = None, threshold = 0.0,
                         norm = True, tol = 1e-5, dt = 0.5, estimated = True,
                         true_schedule = True, partial = False, cycles = False,
                         MIP_problem = None, cum_cells_tol = 1e-1):
    #Get last measured time
    t_sim_end = twin.tumor['t_scan'][-1]
    #Get last prediction time
    t_pred_end = twin.tumor['Future t_scan'][-1]
    
    t_trx_soc = twin.tumor['t_trx'][twin.tumor['t_trx']<t_sim_end]
    t_trx_future = twin.tumor['t_trx'][twin.tumor['t_trx']>=t_sim_end]
    
    if true_schedule == True:
        potential_days = np.arange(t_trx_future[0], t_pred_end, step = interval)
    else:
        dif = np.diff(t_trx_future)
        t_pred_end = t_trx_future[-1]+dif
        potential_days = np.arange(t_trx_future[0], t_pred_end, step = interval)
    
    if max_dose is None:
        max_dose = t_trx_future.size
        
    #Set up constraint and objective functions
    valid_objectives = {'final_cells':{'func':qoi_TotalCells,'args':['cell_tc'],'kwargs':{},'name':'final_cells','title':'Final cell count'},
                        'max_cells':{'func':qoi_MaxCells,'args':['cell_tc','index'],'kwargs':{},'name':'max_cells','title':'Max cell count'},
                        'cumulative_cells':{'func':qoi_CumulativeCells,'args':['cell_tc',dt],'kwargs':{},'name':'cumulative_cells','title':'Cumulative cell count'},
                        'midpoint_cells':{'func':qoi_MidCells,'args':['cell_tc','mid_index'],'kwargs':{},'name':'midpoint_cells','title':'Midpoint cell count'},
                        'third_cells':{'func':qoi_MidCells,'args':['cell_tc','third_index'],'kwargs':{},'name':'third_cells','title':'33% cell count'},
                        'twothirds_cells':{'func':qoi_MidCells,'args':['cell_tc','twothird_index'],'kwargs':{},'name':'twothirds_cells','title':'66% cell count'},
                        'time_to_prog':{'func':qoi_TTP,'args':['cell_tc',dt],'kwargs':{},'name':'time_to_prog','title':'Time to progression'}
                        #Add more objective calls here
                        }
    call_objectives = []
    for elem in objectives:
        call_objectives.append(valid_objectives[elem])
        if elem == 'time_to_prog':
            t_pred_end = t_pred_end + (15*12) #Add 12 weeks to final measurement
        #Overwrite default kwargs based on inputs
    if not call_objectives:
        for elem in valid_objectives:
            call_objectives.append(valid_objectives[elem])
            
    valid_constraints = {'cumulative_concentration':{'func':con_CumulativeConcentration,
                                                      'args':['concentrations'],
                                                      'kwargs':{'dt':dt},
                                                      'name':'cumulative_concentration',
                                                      'title':'Cumulative Concentration',
                                                      'ylabel':'Concentration * days',
                                                      'tol':1e-3},
                         'max_concentration':{'func':con_MaxConcentration, 
                                              'args':['concentrations'],
                                              'kwargs':{},
                                              'name':'max_concentration',
                                              'title':'Max Concentration',
                                              'ylabel':'Concentration',
                                              'tol':1e-3},
                         'ld50_toxicity':{'func':con_Toxicity_ld50,
                                          'args':['concentrations'],
                                          'kwargs':{'dt':dt},
                                          'name':'ld50_toxicity',
                                          'title':'LD50 Toxicity',
                                          'ylabel':'Toxicity estimate',
                                          'tol':1e-3},
                         'gradient':{'func':con_Gradient,'args':['cells'],
                                     'kwargs':{'dt':dt}, 'name':'gradient',
                                     'title':'Gradient','ylabel':'cells/day',
                                     'tol':1e-3},
                         'doses':{'func':None,'args':['doses'],'kwargs':{},
                                  'name':'doses','title':'Total dose',
                                  'ylabel':'Dose',
                                  'tol':0},
                         'cumulative_cells':{'func':con_CumulativeCells,
                                             'args':['cells',dt],
                                             'kwargs':{},
                                             'name':'cumulative_cells',
                                             'title':'Cumulative cells',
                                             'ylabel':'cells * days',
                                             'tol':cum_cells_tol},
                         'midpoint_cells':{'func':qoi_MidCells,
                                           'args':['cells','mid_index'],
                                           'kwargs':{},
                                           'name':'midpoint_cells',
                                           'title':'Midpoint cell count',
                                           'ylabel':'cells',
                                           'tol':1e-3},
                         'third_cells':{'func':qoi_MidCells,
                                           'args':['cells','third_index'],
                                           'kwargs':{},
                                           'name':'third_cells',
                                           'title':'1/3 cell count',
                                           'ylabel':'cells',
                                           'tol':1e-3},
                         'twothirds_cells':{'func':qoi_MidCells,
                                           'args':['cells','twothirds_index'],
                                           'kwargs':{},
                                           'name':'twothirds_cells',
                                           'title':'2/3 cell count',
                                           'ylabel':'cells',
                                           'tol':1e-3},
                         'V3_cells':{'func':qoi_MidCells,
                                           'args':['cells','V3_index'],
                                           'kwargs':{},
                                           'name':'V3_cells',
                                           'title':'Visit 3 cell count',
                                           'ylabel':'cells',
                                           'tol':1e-3},
                         #Add more nonlinear constraint calls here
                         }
    call_constraints = []
    lin_constraints = []
    for elem in constraints:
        if elem != 'total_dose':
            call_constraints.append(valid_constraints[elem])
        else:
            if cycles == False:
                lin_constraints.append({'type':'ineq', 'fun':lambda x: max_dose - sum(x)})
            else:
                lin_constraints.append({'type':'ineq', 'fun':lambda x: max_dose/2 - sum(x)})
    
    simulations = twin.predict(dt = dt, threshold = threshold, estimated = estimated, partial = partial, change_t = t_pred_end)
        
    #Evaluate objectives for SOC protocol
    soc_obj = {}
    for i in range(len(call_objectives)):
        soc_obj[call_objectives[i]['name']] = _call_objectiveFunc(call_objectives[i], simulations, metric)
        
    #Evaluate constraints for SOC protocol
    soc_con = {}
    tol_vec = np.empty(0)
    for i in range(len(call_constraints)):
        if call_constraints[i]['name'] == 'gradient':
            temp = np.zeros(2)
        else:
            temp = _call_constraintFunc(call_constraints[i], simulations, metric)
        soc_con[call_constraints[i]['name']] = temp
        tol_vec = np.append(tol_vec, temp * call_constraints[i]['tol'])
    
    soc_con = {}
    for i in range(len(call_constraints)):
        if call_constraints[i]['name'] == 'gradient':
            soc_con[call_constraints[i]['name']] = np.zeros(2)
        else:
            soc_con[call_constraints[i]['name']] = _call_constraintFunc(call_constraints[i], simulations, metric)
    
    if weights is None:
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
               't_sim_end':t_sim_end,'t_pred_end':t_pred_end,'tc_est':estimated,
               'true_schedule':true_schedule,'partial':partial,'cycles':cycles,
               'tol_vec':tol_vec,'t_trx_future_soc':t_trx_future}
    
    if MIP_problem:
        problem['MIP_problem'] = MIP_problem
    
    return problem

def randomizeInitialGuess(twin, problem):
    run = 1
    while run == 1:
        doses_guess = generateGuess(problem['potential_days'].size, problem['max_dose'],
                                    problem['separate_doses'], problem['cycles'])
        test = constrainedObjective(doses_guess, problem, twin)[0]
        if test != np.inf:
            run = 0
    return doses_guess
            
def generateGuess(n, max_dose, separate_doses, cycles):
    if cycles == True:
        max_dose = max_dose/2
        n = int(n/2) 
    x = np.random.rand(n)
    return x * (max_dose / np.sum(x, axis = 0))

def reorganize_doses(x, problem):
    if problem['separate_doses'] == True:
        x = np.reshape(x,(-1,2))
    if problem['cycles'] == True:
        x = np.concatenate((x, x), axis = 0)
    return x

def socInitialGuess(twin, problem):
    if problem['cycles'] == True:
        x = np.zeros(int(problem['potential_days'].size/2))
        x[0] = 1.0
    else:
        x = np.zeros(problem['potential_days'].size)
        x[problem['potential_days'] == problem['t_trx_future_soc'][0]] = 1.0
        x[problem['potential_days'] == problem['t_trx_future_soc'][1]] = 1.0
    return x
  
def singleInitialGuess(x, problem):
    doses_guess = generateGuess(problem['potential_days'].size, problem['max_dose'], problem['separate_doses'], problem['cycles'])
    return doses_guess

def mipInitialGuess(doses, days, problem):    
    x = np.zeros(problem['potential_days'].size)
    x[np.where(np.in1d(problem['potential_days'], days))[0]] = doses
    
    if problem['cycles'] == True:
        n = x.size
        x = x[:n]
        
    return x

############################### Constraints ###################################
#Plan based constraints
#These can be solved with a linear inequality
def con_TotalDose(trx_params):
    if 'doses' not in trx_params.keys():
        return trx_params['t_trx'].size
    else:
        return sum(trx_params['doses'])

#Concentration based constraints
def con_CumulativeConcentration(concentrations, dt = 0.5):
    return integrate.cumulative_trapezoid(concentrations, dx = dt, axis = 0)[-1,:]
    
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
    return integrate.cumulative_trapezoid(toxic, x = check, axis = 0)[-1,:]

#Cell based constraints
def con_CumulativeCells(timecourse, index, dt):
    return integrate.cumulative_trapezoid(timecourse[index:], dx = dt)[-1]

def con_Gradient(timecourse, indices, dt = 0.5):
    grad1 = (timecourse[indices[1]] - timecourse[indices[0]])/((indices[1] - indices[0])*dt)
    grad2 = (timecourse[indices[2]] - timecourse[indices[1]])/((indices[2] - indices[1])*dt)
    return np.array([grad1, grad2])

############################### Objectives ####################################
def qoi_TotalCells(timecourse):
    return timecourse[-1]

def qoi_MaxCells(timecourse, index):
    return max(timecourse[index:])

def qoi_CumulativeCells(timecourse, index, dt):
    return integrate.cumulative_trapezoid(timecourse[index:], dx = dt)[-1]

def qoi_MidCells(timecourse, index):
    return timecourse[index]

def qoi_TTP(timecourse, dt, index):
    tc = timecourse[index:]
    
    temp = timecourse[index]
    idx = np.where(tc > temp)
    # print(idx)

    
    try:
        return -tc[:idx[0][0]].size * dt
    except:
        return -tc.size * dt

############################### Optimizer #####################################
def constrainedObjective(x, problem, twin, tol = 1e-15, norm = False):
    """
    x = current test dosage. Either n x 2 or n x 1 if the drugs are optimized together
    problem = tells which functions to run
    twin = holds parameter details from calibration and prediction functions
    """
    x = reorganize_doses(x, problem)
    new_days = np.concatenate((problem['t_trx_soc'], problem['potential_days']),axis = 0)
    new_doses = np.concatenate((problem['doses_soc'], x), axis = 0)
    trx_params = {'t_trx': new_days, 'doses': new_doses}
    simulations = _call_predict(twin, problem, treatment = trx_params)
    
    A = np.empty(0)
    B = np.empty(0)
    for i in range(len(problem['nonlin-constraints'])):
        test = _call_constraintFunc(problem['nonlin-constraints'][i], simulations, problem['metric'])
        check = problem['soc_con'][problem['nonlin-constraints'][i]['name']]
        A = np.append(A, test)
        B = np.append(B, check) 
        
    if np.any(A > B+problem['tol_vec']):
        return np.inf, B+problem['tol_vec'] - A
    
    obj = 0
    for i in range(len(problem['objectives'])):
        if norm == False:
            n = 1
        else:
            n = problem['soc_obj'][problem['objectives'][i]['name']]
        obj += (_call_objectiveFunc(problem['objectives'][i], simulations, problem['metric']) / n) * problem['weights'][i]

    return obj, B+problem['tol_vec'] - A
 
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
        x_save = x.copy()
        x = reorganize_doses(x, self.problem)
        new_days = np.concatenate((self.problem['t_trx_soc'], self.problem['potential_days']),axis = 0)
        new_doses = np.concatenate((self.problem['doses_soc'], x), axis = 0)
        trx_params = {'t_trx': new_days, 'doses': new_doses}
        self.cache[str(x_save)] = _call_predict(self.twin, self.problem, treatment = trx_params)
        
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
            obj += (_call_objectiveFunc(self.problem['objectives'][i], simulations, self.problem['metric']) / n) * self.problem['weights'][i]
        return obj
    
    def constraints(self, x):
        if not self.in_cache(x):
            self.updateSimulation(x)
        simulations = self.cache[str(x)]  
        A = np.empty(0)
        B = np.empty(0)
        for i in range(len(self.problem['nonlin-constraints'])):
            test = _call_constraintFunc(self.problem['nonlin-constraints'][i], simulations, self.problem['metric'])
            check = self.problem['soc_con'][self.problem['nonlin-constraints'][i]['name']]
            A = np.append(A, test)
            B = np.append(B, check)

        return B+self.problem['tol_vec'] - A
    
############################### Internal ######################################
def _call_predict(twin, problem, treatment = None):
    return twin.predict(dt = problem['dt'], threshold = problem['threshold'], 
                 estimated = problem['tc_est'], partial = problem['partial'], 
                 change_t = problem['t_pred_end'], treatment = treatment)
    
def _call_objectiveFunc(function, simulations, metric):
    string = function['args'][0]
    if string == 'cell_tc':
        data = _evalMetric(simulations['cell_tc'], metric, 'timecourse')
    elif string == 'volume_tc':
        data = _evalMetric(simulations['volume_tc'], metric, 'timecourse')
        
    if len(function['args']) > 1:
        #Currently assumes max time course constraint
        if function['name'] == 'max_cells':
            return function['func'](data, simulations['t_pred_index'], **function['kwargs'])
        elif function['name'] == 'midpoint_cells':
            return function['func'](data, simulations['t_mid_index'], **function['kwargs'])
        elif function['name'] == 'third_cells':
            return function['func'](data, simulations['t_third_index'], **function['kwargs'])
        elif function['name'] == 'twothirds_cells':
            return function['func'](data, simulations['t_twothirds_index'], **function['kwargs'])
        elif function['name'] == 'time_to_prog':
            return function['func'](data, function['args'][1], simulations['ttp_index'], **function['kwargs'])
        else:
            return function['func'](data, simulations['t_pred_index'], *function['args'][1:], **function['kwargs'])
    else:
        return function['func'](data, **function['kwargs'])
    
def _call_constraintFunc(function, simulations, metric):      
    string = function['args'][0]
    if string == 'trx_params':
        return function['func'](simulations['trx_params'], **function['kwargs'])
    elif string == 'concentrations':
        data = _evalMetric(simulations['drug_tc'], metric, 'concentration')
        return function['func'](data, **function['kwargs'])
    elif string == 'doses':
        return sum(simulations['doses'])
    elif string == 'cells':
        data = _evalMetric(simulations['cell_tc'], metric, 'timecourse')
        if function['name'] == 'cumulative_cells':
            index = simulations['t_pred_index']
            return function['func'](data, index, *function['args'][1:], **function['kwargs'])
        elif function['name'] == 'midpoint_cells':
            index = simulations['t_mid_index']
            return function['func'](data, index, **function['kwargs'])
        elif function['name'] == 'third_cells':
            index = simulations['t_third_index']
            return function['func'](data, index, **function['kwargs'])
        elif function['name'] == 'twothirds_cells':
            index = simulations['t_twothirds_index']
            return function['func'](data, index, **function['kwargs'])
        elif function['name'] == 'V3_cells':
            index = simulations['t_V3_index']
            return function['func'](data, index, **function['kwargs'])
        else:
            indices = [simulations['t_pred_index'], simulations['t_third_index'], simulations['t_twothirds_index']]
            return function['func'](data, indices, **function['kwargs'])
    
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
        elif metric == 'percentile':
            return np.percentile(data, 0.95, axis = get_axis)
        elif metric == 'median':
            return np.median(data, axis = get_axis)
        elif metric == 'max':
            return np.max(data, axis = get_axis)
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

############################### Plotting ######################################
def plotObj_comparison(problem, simulation1, simulation2, color1 = [0,0,1], color2 = [0.92,0.56,0.02]):
    num = len(problem['objectives'])
    fig, ax = plt.subplots(1,num,layout = 'constrained',figsize = (3*num,3))
    obj1 = []
    obj2 = []
    for i in range(num):
        function = problem['objectives'][i]
        data1 = simulation1[function['args'][0]]
        data2 = simulation2[function['args'][0]]
        temp1 = []
        temp2 = []
        for j in range(data1.shape[1]):
            if len(function['args']) > 1:
                #Currently assumes max time course constraint
                if function['name'] == 'max_cells':
                    temp1.append(function['func'](data1[:,j],simulation1['t_pred_index'], **function['kwargs']))
                    temp2.append(function['func'](data2[:,j],simulation2['t_pred_index'], **function['kwargs']))
                elif function['name'] == 'midpoint_cells':
                    temp1.append(function['func'](data1[:,j],simulation1['t_mid_index'], **function['kwargs']))
                    temp2.append(function['func'](data2[:,j],simulation2['t_mid_index'], **function['kwargs']))
                elif function['name'] == 'third_cells':
                    temp1.append(function['func'](data1[:,j],simulation1['t_third_index'], **function['kwargs']))
                    temp2.append(function['func'](data2[:,j],simulation2['t_third_index'], **function['kwargs']))
                elif function['name'] == 'twothirds_cells':
                    temp1.append(function['func'](data1[:,j],simulation1['t_twothirds_index'], **function['kwargs']))
                    temp2.append(function['func'](data2[:,j],simulation2['t_twothirds_index'], **function['kwargs']))
                elif function['name'] == 'time_to_prog':
                    temp1.append(function['func'](data1[:,j],*function['args'][1:],simulation1['ttp_index'], **function['kwargs']))
                    temp2.append(function['func'](data2[:,j],*function['args'][1:],simulation2['ttp_index'], **function['kwargs']))
                else:
                    temp1.append(function['func'](data1[:,j],simulation1['t_pred_index'], *function['args'][1:], **function['kwargs']))
                    temp2.append(function['func'](data2[:,j],simulation2['t_pred_index'], *function['args'][1:], **function['kwargs']))
            else:
                temp1.append(function['func'](data1[:,j],**function['kwargs']))
                temp2.append(function['func'](data2[:,j],**function['kwargs']))
        try:
            ax[i].hist(temp1,bins=15,color = color1,label='SoC',alpha=0.6)[1] 
            ax[i].hist(temp2,bins=15,color = color2,label='Opt.',alpha=0.6)
            ax[i].legend(fontsize='xx-small')
            ax[i].set_xlabel(function['title']+'; w = '+str(problem['weights'][i]), fontsize='small')
            ax[i].set_ylabel('Counts')
        except:
            ax.hist(temp1,bins=15,color = color1,label='SoC',alpha=0.6)[1] 
            ax.hist(temp2,bins=15,color = color2,label='Opt.',alpha=0.6)
            ax.legend(fontsize='xx-small')
            ax.set_xlabel(function['title'])
            ax.set_ylabel('Counts')    
        obj1.append(temp1)
        obj2.append(temp2)
    return obj1, obj2
def plotCon_comparison(problem, simulation1, simulation2, color1 = [0,0,1], color2 = [0.92,0.56,0.02]):
    num_l = len(problem['lin-constraints'])
    num_nl = len(problem['nonlin-constraints'])
    num = num_l + num_nl
    fig, ax = plt.subplots(1,num,layout = 'constrained',figsize = (3*num,3))
    nl_cnt = 0
    
    con1 = []
    con2 = []
    for i in range(num):
        if i+1 <= num_l:
            function = problem['lin-constraints'][i]
            val1 = con_TotalDose(simulation1['trx_params'])
            val2 = con_TotalDose(simulation2['trx_params'])
            ax[i].bar(['SoC','Opt.'],[val1,val2],color=[color1,color2])
            ax[i].set_ylabel('Total Dose (over 4 cycles)')
            ax[i].set_xlabel('Schedule')
            ax[i].set_title('Total Dose')
            
            con1.append(val1)
            con2.append(val2)
        else:
            function = problem['nonlin-constraints'][nl_cnt]
            nl_cnt += 1
            data1 = np.array(simulation1['drug_tc'])
            data2 = np.array(simulation2['drug_tc'])
            
            cells1 = np.array(simulation1['cell_tc'])
            cells2 = np.array(simulation2['cell_tc']) 
            
            #Plotting
            if function['name'] == 'gradient':
                temp1 = np.empty((cells1.shape[1],2))
                temp2 = np.empty((cells1.shape[1],2))
                for j in range(cells1.shape[1]):
                    indices1 = [simulation1['t_pred_index'], simulation1['t_third_index'], simulation1['t_twothirds_index']]
                    indices2 = [simulation2['t_pred_index'], simulation2['t_third_index'], simulation2['t_twothirds_index']]
                    temp1[j,:] = function['func'](cells1[:,j],indices1, **function['kwargs'])
                    temp2[j,:] = function['func'](cells2[:,j],indices2, **function['kwargs'])
                ind = np.arange(2)
                width = 0.35
                ax[i].bar(ind, np.mean(temp1, axis = 0), width, yerr = np.std(temp1, axis = 0), color = color1, label = 'SoC')
                ax[i].bar(ind + width, np.mean(temp2, axis = 0), width, yerr = np.std(temp2,axis = 0), color = color2, label = 'Opt.')
            elif function['name'] == 'cumulative_cells':
                temp1 = np.empty((cells1.shape[1],1))
                temp2 = np.empty((cells1.shape[1],1))
                for j in range(cells1.shape[1]):
                    index1 = simulation1['t_pred_index']
                    index2 = simulation2['t_pred_index']
                    temp1[j] = function['func'](cells1[:,j],index1, *function['args'][1:], **function['kwargs'])
                    temp2[j] = function['func'](cells2[:,j],index2, *function['args'][1:], **function['kwargs'])
                ax[i].bar(['SoC','Opt.'],[np.mean(temp1),np.mean(temp2)], yerr = [np.std(temp1),np.std(temp2)],color=[color1,color2])
            elif function['name'] == 'midpoint_cells':
                temp1 = np.empty((cells1.shape[1],1))
                temp2 = np.empty((cells1.shape[1],1))
                for j in range(cells1.shape[1]):
                    index1 = simulation1['t_mid_index']
                    index2 = simulation2['t_mid_index']
                    temp1[j] = function['func'](cells1[:,j],index1, **function['kwargs'])
                    temp2[j] = function['func'](cells2[:,j],index2, **function['kwargs'])
                ax[i].bar(['SoC','Opt.'],[np.mean(temp1),np.mean(temp2)], yerr = [np.std(temp1),np.std(temp2)],color=[color1,color2])
                ax[i].set_ylim(bottom = 0)
            elif function['name'] == 'third_cells':
                temp1 = np.empty((cells1.shape[1],1))
                temp2 = np.empty((cells1.shape[1],1))
                for j in range(cells1.shape[1]):
                    index1 = simulation1['t_third_index']
                    index2 = simulation2['t_third_index']
                    temp1[j] = function['func'](cells1[:,j],index1, **function['kwargs'])
                    temp2[j] = function['func'](cells2[:,j],index2, **function['kwargs'])
                ax[i].bar(['SoC','Opt.'],[np.mean(temp1),np.mean(temp2)], yerr = [np.std(temp1),np.std(temp2)],color=[color1,color2])
                ax[i].set_ylim(bottom = 0)
            elif function['name'] == 'twothirds_cells':
                temp1 = np.empty((cells1.shape[1],1))
                temp2 = np.empty((cells1.shape[1],1))
                for j in range(cells1.shape[1]):
                    index1 = simulation1['t_twothirds_index']
                    index2 = simulation2['t_twothirds_index']
                    temp1[j] = function['func'](cells1[:,j],index1, **function['kwargs'])
                    temp2[j] = function['func'](cells2[:,j],index2, **function['kwargs'])
                ax[i].bar(['SoC','Opt.'],[np.mean(temp1),np.mean(temp2)], yerr = [np.std(temp1),np.std(temp2)],color=[color1,color2])
                ax[i].set_ylim(bottom = 0)
            elif function['name'] == 'V3_cells':
                temp1 = np.empty((cells1.shape[1],1))
                temp2 = np.empty((cells1.shape[1],1))
                for j in range(cells1.shape[1]):
                    index1 = simulation1['t_V3_index']
                    index2 = simulation2['t_V3_index']
                    temp1[j] = function['func'](cells1[:,j],index1, **function['kwargs'])
                    temp2[j] = function['func'](cells2[:,j],index2, **function['kwargs'])
                ax[i].bar(['SoC','Opt.'],[np.mean(temp1),np.mean(temp2)], yerr = [np.std(temp1),np.std(temp2)],color=[color1,color2])
                ax[i].set_ylim(bottom = 0)
            else:
                temp1 = np.empty((data1.shape[0],2))
                temp2 = np.empty((data1.shape[0],2))
                for j in range(data1.shape[0]):
                    temp1[j,:] = function['func'](data1[j,:,:],**function['kwargs'])
                    temp2[j,:] = function['func'](data2[j,:,:],**function['kwargs'])
                ind = np.arange(2)
                width = 0.35
                ax[i].bar(ind, np.mean(temp1, axis = 0), width, yerr = np.std(temp1, axis = 0), color = color1, label = 'SoC')
                ax[i].bar(ind + width, np.mean(temp2, axis = 0), width, yerr = np.std(temp2,axis = 0), color = color2, label = 'Opt.')
                
            #Plot details
            if function['name'] == 'gradient':
                ax[i].set_xticks(ind + width/2, labels = ['33%','66%'])
                ax[i].set_xlabel('Time to end')
                ax[i].legend(fontsize='x-small',loc='lower right')
            elif function['name'] == 'cumulative_cells' or 'midpoint_cells' or 'third_cells' or 'twothirds_cells':
                ax[i].set_xlabel('Schedule')
            else:
                ax[i].set_xticks(ind + width/2, labels = ['A.','C.'])
                ax[i].set_xlabel('Drug type')
                ax[i].legend(fontsize='x-small',loc='lower right')
            ax[i].set_title(function['title'])
            ax[i].set_ylabel(function['ylabel'])
            
            con1.append(temp1)
            con2.append(temp2)
    return con1, con2