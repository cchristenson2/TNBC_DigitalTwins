# ForwardModels.py
""" Forward models used in calibration and prediction
*Example RXDIF_2D(N0, kp, d, t, h, dt, bcs)
*RXDIF_3D_wAC(N0, kp, d, alpha, trx_params, t, h, dt, bcs)
*OP_RXDIF_wAC(N0, ops, trx_params, t, dt)

Defines parameter object
Constists of:
    - parameter name; d, alpha, k, beta1, beta2
    - assignment; f, l, or g for fixed, local, or global
    - value; default = None, replaced by either fixed value or calibration

Internal:
    *updateDosing(k, nt_trx, delivs, dt)

Last updated: 5/1/2024
"""

import numpy as np

###############################################################################
## Example base reaction diffusion model
def RXDIF_2D(N0, k, d, t, h, dt, bcs):
    """
    Parameters
    ----------
    N0 : (sy, sx) ndarray
        Initial condition matrix
    kp : (N0 shape) ndarray
        Proliferation matrix, either local values or constant
    d : float
        Scalar diffusivity value
    t : (num simulation outputs) nparray
        Time points to output simulation
    h : float
        Grid spacing for FDM
    dt : float
        Time spacing for euler step
    bcs : (N0 Shape, 2) ndarray
        Boundary condition mask, as [y,x,[x boundary type, y boundary type]]

    Returns
    -------
    Simulation
        (N0 Shape, number of points in t) ndarray

    """
    #Setup matrices and indexing
    t_ = (t / dt).astype(int)
    [sy,sx] = N0.shape
    nt = np.arange(0,t[-1] + dt,dt).size
    Sim = np.zeros([sy,sx,nt])
    Sim[:,:,0] = N0
    
    #Time stepping
    for step in range(1,nt):
        #Space stepping
        for y in range(sy):
            for x in range(sx):
                boundary = bcs[y,x,:]
                
                #FDM in X direction
                if boundary[0] == 0.:
                    lap_x = d*(Sim[y,x-1,step-1] - 2*Sim[y,x,step-1]
                               + Sim[y,x+1,step-1])/h**2
                elif boundary[0] == 1.:
                    lap_x = d*(2*Sim[y,x-1,step-1] - 2*Sim[y,x,step-1])/h**2
                elif boundary[0] == -1.:
                    lap_x = d*(-2*Sim[y,x,step-1] + 2*Sim[y,x+1,step-1])/h**2
                else:
                    lap_x = 0           
                #FDM in Y direction
                if boundary[1] == 0.:
                    lap_y = d*(Sim[y-1,x,step-1] - 2*Sim[y,x,step-1] 
                               + Sim[y+1,x,step-1])/h**2
                elif boundary[1] == 1.:
                    lap_y = d*(2*Sim[x,y-1,step-1] - 2*Sim[y,x,step-1])/h**2
                elif boundary[1] == -1.:
                    lap_y = d*(-2*Sim[y,x,step-1] + 2*Sim[y+1,x,step-1])/h**2
                else:
                    lap_y = 0
                
                #Add together
                diffusion = lap_y + lap_x
                proliferation = Sim[y,x,step-1]*k[y,x]*(1 - Sim[y,x,step-1])
                #Apply time stepping
                Sim[y,x,step] = Sim[y,x,step-1] + dt*(diffusion + proliferation)
                
    return Sim[:,:,t_]

###############################################################################
# Internal functions for treatment solves
def updateDosing(step, nt_trx, delivs, dt):
    """
    Parameters
    ----------
    step : TYPE
        Current time index
    nt_trx : ndarray
        Indices of treatment times
    delivs : ndarray
        Contains time since deliveries that have occurred
    dt : float
        Euler time stepping

    Returns
    -------
    Updated delivs

    """
    #Increment all current treatments by dt
    if delivs.size > 0:
        delivs = delivs + dt
    if delivs.size < nt_trx.size:
        if step - 1 >= nt_trx[delivs.size]: #Delivery occurred at previous step
            delivs = np.append(delivs,0)  
    return delivs
  
def setupTRX(trx_params, nt, dt):
    """
    Decrease repeated definitions in each individual forward models
    """
    if np.isscalar(trx_params.get('beta')):
        beta = np.array([trx_params.get('beta'), trx_params.get('beta')])
    else:
        beta = trx_params.get('beta')  
        
    #Setup treatment matrices
    nt_trx = trx_params.get('t_trx') / dt #Indices of treatment times
    delivs = np.array([]) #Storage for time since deliveries that have passed
    drugs = np.zeros([2,nt])
    
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
            
###############################################################################
## FDM based simulations
# Treatment included
def RXDIF_3D_wAC(N0, k, d, alpha, trx_params, t, h, dt, bcs):
    """
    Parameters
    ----------
    N0 : (sy, sx, sz) ndarray
        Initial condition matrix
    k : (N0 shape) ndarray
        Proliferation matrix, either local values or constant
    d : float
        Scalar diffusivity value
    alpha: (number of calibrated treatment efficacies) ndarray
        Treatment efficacy term, if alpha = 0, treatment does nothing except slow
        down evaluation
    trx_params: dictionary
        Contains treatment information, drug decays, schedule, and dosages
    t : (num simulation outputs) nparray
        Time points to output simulation
    h : (dimensions) ndarray
        Grid spacing for FDM, as [x, y, z]
    dt : float
        Time spacing for euler step
    bcs : (N0 Shape, 2) ndarray
        Boundary condition mask, as [y,x,z,[x boundary type, y boundary type, z type]]

    Returns
    -------
    Simulation
        (N0 Shape, number of points in t) ndarray

    """
    #Setup matrices and indexing
    t_ = (t / dt).astype(int)
    [sy,sx,sz] = N0.shape
    nt = np.arange(0,t[-1] + dt,dt).size
    Sim = np.zeros([sy,sx,sz,nt])
    Sim[:,:,:,0] = N0
    
    if np.isscalar(alpha): #If using one alpha for both drugs
        alpha = np.array([alpha, alpha]) 
        
    if np.isscalar(k):
        k = k*np.ones([sy,sx,sz])
        
    beta, nt_trx, delivs, drugs, doses = setupTRX(trx_params, nt, dt)

    #Time stepping
    for step in range(1,nt):
        
        delivs  = updateDosing(step, nt_trx, delivs, dt)
        #Get current dosage
        for n in range(delivs.size):
            drugs[0,step] = drugs[0,step] + doses[n,0]*np.exp(-1*beta[0]*delivs[n])
            drugs[1,step] = drugs[1,step] + doses[n,1]*np.exp(-1*beta[1]*delivs[n])
            
        #Space stepping
        for z in range(sz):
            for y in range(sy):
                for x in range(sx):
                    boundary = bcs[y,x,z,:]
                    
                    #FDM in X direction
                    if boundary[0] == 0.:
                        lap_x = d*(Sim[y,x-1,z,step-1] - 2*Sim[y,x,z,step-1] 
                                   + Sim[y,x+1,z,step-1])/h[0]**2
                    elif boundary[0] == 1.:
                        lap_x = d*(2*Sim[y,x-1,z,step-1] 
                                   - 2*Sim[y,x,z,step-1])/h[0]**2
                    elif boundary[0] == -1.:
                        lap_x = d*(-2*Sim[y,x,z,step-1] 
                                   + 2*Sim[y,x+1,z,step-1])/h[0]**2
                    else:
                        lap_x = 0
                    #FDM in Y direction
                    if boundary[1] == 0.:
                        lap_y = d*(Sim[y-1,x,z,step-1] - 2*Sim[y,x,z,step-1]
                                   + Sim[y+1,x,z,step-1])/h[1]**2
                    elif boundary[1] == 1.:
                        lap_y = d*(2*Sim[y-1,x,z,step-1] 
                                   - 2*Sim[y,x,z,step-1])/h[1]**2
                    elif boundary[1] == -1.:
                        lap_y = d*(-2*Sim[y,x,z,step-1] 
                                   + 2*Sim[y+1,x,z,step-1])/h[1]**2
                    else:
                        lap_y = 0
                    #FDM in Z direction
                    if boundary[2] == 0.:
                        lap_z = d*(Sim[y,x,z-1,step-1] - 2*Sim[y,x,z,step-1] 
                                   + Sim[y,x,z+1,step-1])/h[2]**2
                    elif boundary[2] == 1.:
                        lap_z = d*(2*Sim[y,x,z-1,step-1] 
                                   - 2*Sim[y,x,z,step-1])/h[2]**2
                    elif boundary[2] == -1.:
                        lap_z = d*(-2*Sim[y,x,z,step-1] 
                                   + 2*Sim[y,x,z+1,step-1])/h[2]**2
                    else:
                        lap_z = 0
                    
                    #Add together
                    diffusion = lap_y + lap_x + lap_z
                    proliferation = (Sim[y,x,z,step-1]*k[y,x,z]
                                     *(1 - Sim[y,x,z,step-1]))
                    treat = (trx_params.get('AUC')[y,x,z] * Sim[y,x,z,step-1] 
                             * (drugs[0,step]*alpha[0] + drugs[1,step]*alpha[1]))
                    
                    #Apply time stepping
                    Sim[y,x,z,step] = Sim[y,x,z,step-1] + dt*(diffusion 
                                                              + proliferation 
                                                              - treat)
                
    return Sim[:,:,:,t_], drugs

def RXDIF_2D_wAC(N0, k, d, alpha, trx_params, t, h, dt, bcs):
    """
    Parameters
    ----------
    N0 : (sx, sy) ndarray
        Initial condition matrix
    p : (N0 shape) ndarray
        Proliferation matrix, either local values or constant
    d : float
        Scalar diffusivity value
    alpha: (number of calibrated treatment efficacies) ndarray
        Treatment efficacy term, if alpha = 0, treatment does nothing except slow
        down evaluation
    trx_params: dictionary
        Contains treatment information, drug decays, schedule, and dosages
    t : (num simulation outputs) nparray
        Time points to output simulation
    h : (dimensions) ndarray
        Grid spacing for FDM, as [x, y]
    dt : float
        Time spacing for euler step
    bcs : (N0 Shape, 2) ndarray
        Boundary condition mask, as [x,y,[x boundary type, y boundary type]]

    Returns
    -------
    Simulation
        (N0 Shape, number of points in t) ndarray

    """
    #Setup matrices and indexing
    t_ = (t / dt).astype(int)
    [sy,sx] = N0.shape
    nt = np.arange(0,t[-1] + dt,dt).size
    Sim = np.zeros([sy,sx,nt])
    Sim[:,:,0] = N0
    
    if np.isscalar(alpha): #If using one alpha for both drugs
        alpha = np.array([alpha, alpha]) 
        
    if np.isscalar(k):
        k = k*np.ones([sy,sx])
        
    beta, nt_trx, delivs, drugs, doses = setupTRX(trx_params, nt, dt)

    #Time stepping
    for step in range(1,nt):
        
        delivs  = updateDosing(step, nt_trx, delivs, dt)
        #Get current dosage
        for n in range(delivs.size):
            drugs[0,step] = drugs[0,step] + doses[n,0]*np.exp(-1*beta[0]*delivs[n])
            drugs[1,step] = drugs[1,step] + doses[n,1]*np.exp(-1*beta[1]*delivs[n])
            
        #Space stepping
        for y in range(sy):
            for x in range(sx):
                boundary = bcs[y,x,:]
                
                #FDM in X direction
                if boundary[0] == 0.:
                    lap_x = d*(Sim[y,x-1,step-1] - 2*Sim[y,x,step-1] 
                               + Sim[y,x+1,step-1])/h[0]**2
                elif boundary[0] == 1.:
                    lap_x = d*(2*Sim[y,x-1,step-1] - 2*Sim[y,x,step-1])/h[0]**2
                elif boundary[0] == -1.:
                    lap_x = d*(-2*Sim[y,x,step-1] + 2*Sim[y,x+1,step-1])/h[0]**2
                else:
                    lap_x = 0
                #FDM in Y direction
                if boundary[1] == 0.:
                    lap_y = d*(Sim[y-1,x,step-1] - 2*Sim[y,x,step-1] 
                               + Sim[y+1,x,step-1])/h[1]**2
                elif boundary[1] == 1.:
                    lap_y = d*(2*Sim[y-1,x,step-1] - 2*Sim[y,x,step-1])/h[1]**2
                elif boundary[1] == -1.:
                    lap_y = d*(-2*Sim[y,x,step-1] + 2*Sim[y+1,x,step-1])/h[1]**2
                else:
                    lap_y = 0
                
                #Add together
                diffusion = lap_y + lap_x
                proliferation = Sim[y,x,step-1]*k[y,x]*(1 - Sim[y,x,step-1])
                
                treat = (trx_params.get('AUC')[y,x] * Sim[y,x,step-1] 
                         * (drugs[0,step]*alpha[0] + drugs[1,step]*alpha[1]))
                
                #Apply time stepping
                Sim[y,x,step] = Sim[y,x,step-1] + dt*(diffusion + proliferation - treat)
                
    return Sim[:,:,t_], drugs

###############################################################################
## Operator based simulations (works wth 2D or 3D depending on input operators and states)
# Treatment included
def OP_RXDIF_wAC(N0, ops, trx_params, t, dt):
    """
    Parameters
    ----------
    N0 : array
        Initial condition vector
    ops : (4x1) dictionary
        contains operators for diffusivity, proliferation, and treatment
    trx_params: dictionary
        Contains treatment information, drug decays, schedule, and dosages
    t : (num simulation outputs) nparray
        Time points to output simulation
    dt : float
        Time spacing for euler step

    Returns
    -------
    Simulation
        (N0 Shape, number of points in t) ndarray

    """
    #Setup matrices and indexing
    t_ = (t / dt).astype(int)
    n = N0.size
    nt = np.arange(0,t[-1] + dt,dt).size
    Sim = np.zeros([n,nt])
    Sim[:,0] = N0
    
    A = ops.get('A')
    B = ops.get('B')
    H = ops.get('H')
    T = ops.get('T')
    #If each drug does not have its own T operator, duplicate the first
    if np.atleast_3d(T).shape[2] == 1:
        T = np.append(np.atleast_3d(T),np.atleast_3d(T),2)
        
    beta, nt_trx, delivs, drugs, doses = setupTRX(trx_params, nt, dt)
    
    #Time stepping
    for step in range(1,nt):
        
        delivs  = updateDosing(step, nt_trx, delivs, dt)
        #Get current dosage
        for n in range(delivs.size):
            drugs[0,step] = drugs[0,step] + doses[n,0]*np.exp(-1*beta[0]*delivs[n])
            drugs[1,step] = drugs[1,step] + doses[n,1]*np.exp(-1*beta[1]*delivs[n])
        
        #solve for next time point
        Sim[:,step] = Sim[:,step-1] + dt * (A@Sim[:,step-1] + B@Sim[:,step-1] 
                                            - H@np.kron(Sim[:,step-1],Sim[:,step-1]) 
                                            - (T[:,:,0]*drugs[0,step])@Sim[:,step-1] 
                                            - (T[:,:,1]*drugs[1,step])@Sim[:,step-1])
            
    return Sim[:,t_], drugs       
            
###############################################################################
# Parameter class object and functions
class Parameter:
    def __init__(self, name, assignment, value = None):
        valid_names = ['d','k','alpha','beta_a','beta_c','sigma']
        valid_assignment = ['f','l','g']
        default_values = [5e-3, 0.05, 0.4, 0.60, 3.25, 0.0]
        
        #Store name of parameter
        if name.lower() in valid_names:
            self.name = name.lower()
        else:
            raise ValueError('Valid parameter names are currently; "d","k",'
                             '"alpha","beta_a","beta_c","sigma"')
        
        #Store assignment type for parameter
        if assignment.lower() in valid_assignment:
            self.assignment = assignment.lower()
        else:
            raise ValueError('Valid assignment types are currently; "f","l","g"')
        self.value = value
        
        #If value is fixed and nothing is provided, use the default parameter value
        if self.assignment == 'f':
            for i, elem in enumerate(valid_names):
                if self.name == elem and value == None:
                    self.value = default_values[i]
   
    def __str__(self):
        if self.assignment == 'f':
            string1 = 'fixed'
        elif self.assignment == 'l':
            string1 = 'local'
        else:
            string1 = 'global'     
        
        if self.value is None:
            string2 = 'not assigned'
        else:
            string2 = 'assigned'
        
        return 'Parameter '+self.name+' is '+string1+'\nValue is currently '+string2
    
    def setBounds(self, bounds):
        self.bounds = bounds
    
    def update(self, value):
        self.value = value
        
    def get(self):
        return self.value
    
    def getBounds(self):
        return self.bounds
    
    def delete(self):
        self.value = None
    
class ReducedParameter:
    def __init__(self, name, assignment, basis, value = None):
        valid_names = ['k']
        valid_assignment = ['r']
        
        #Store name of parameter
        if name.lower() in valid_names:
            self.name = name.lower()
        else:
            raise ValueError('Valid parameter names are currently; "k"')
        
        #Store assignment type for parameter
        if assignment.lower() in valid_assignment:
            self.assignment = assignment.lower()
        else:
            raise ValueError('Valid assignment types are currently; "r"')
        self.value = value
        self.basis = basis
        
    def __str__(self):
        if self.assignment == 'r':
            string1 = 'reduced'   
        
        if self.value is None:
            string2 = 'not assigned'
        else:
            string2 = 'assigned'
        
        return 'Parameter '+self.name+' is '+string1+'\nValue is currently '+string2
    
    def setBounds(self, bounds):
        self.bounds = bounds
        
    def setFullBounds(self, bounds):
        self.full_bounds = bounds
    
    def update(self, value):
        self.value = value
        
    def get(self):
        return self.value
    
    def getBounds(self):
        return self.bounds  
    
    def getFullBounds(self):
        return self.full_bounds 
    
    def delete(self):
        self.value = None