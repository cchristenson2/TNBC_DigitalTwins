# Library.py
""" Construct complete libraries for a given model and interpolate
*getROMLibrary(tumor, V, bounds, num = 2, zipped = None)
    - Construct library:
        - RXDIF w AC model (diffusivity, proliferation, and treatment operators)
*interpolateGlobal(Lib, val)
    - Interpolate global parameters from library
*interpolateLocal(Libs, vals)
    - Interpolate local parameters from library
*getOperators(curr, ROM)
    - Gets operators for all parameters in a dictionary

#Internal only
*constructGlobalLibrary(var, tumor, V, bounds, num)
*constructLocalLibrary(var, tumor, V, bounds, num)

Last updated: 5/2/2024
"""

import numpy as np

import Operators as op

########################## Internal constructions #############################
def _constructGlobalLibrary(var, tumor, V, bounds, num):
    """
    Used to build an operator library with num components for operator 
    label 'var'.
    Works for all operators based on global parameters used in the RXDIF model.
    """
    Lib = {}
    vec = np.linspace(bounds[0], bounds[1], num)
    for i in range(vec.size):
        if var == 'A':
            Lib[str(i)] = op.buildReduced_A(V, tumor, vec[i])
        elif var == 'B':
            Lib[str(i)] = op.buildReduced_B(V, tumor, vec[i])
        elif var == 'H':
            Lib[str(i)] = op.buildReduced_H(V, tumor, vec[i])
        elif var == 'T':
            Lib[str(i)] = op.buildReduced_T(V, tumor, vec[i])
    Lib['vec'] = vec
    
    return Lib

def _constructLocalLibrary(var, tumor, V, bounds, num):
    """
    Used to build an operator library with num components for operator 
    label 'var'.
    Only works for local proliferation based operators.
    """
    Lib = {}
    n,r = V.shape
    coeff_bounds = np.zeros([r,2])
    for i in range(r):
        temp = np.zeros([1,2])
        #Get coefficient boundaries
        test_vec = np.zeros([n,1])
        test_vec[V[:,i] >= 0] = bounds[1]
        test_vec[V[:,i] < 0] = bounds[0]
        temp[0,0] = V[:,i].T @ test_vec
        test_vec = np.zeros([n,1])
        test_vec[V[:,i] >= 0] = bounds[0]
        test_vec[V[:,i] < 0] = bounds[1]
        temp[0,1] = V[:,i].T @ test_vec
        temp = np.sort(temp)
        coeff_bounds[i,:] = temp
        
        Lib["Mode"+str(i)] = {}
        vec = np.linspace(temp[0,0],temp[0,1],num)
        for j in range(vec.size):
            if var == 'B':
                Lib["Mode"+str(i)][str(j)] = op.buildReduced_B(V, tumor, V[:,i]
                                                               * vec[j])
            elif var == 'H':
                Lib["Mode"+str(i)][str(j)] = op.buildReduced_H(V, tumor, V[:,i]
                                                               * vec[j])
        Lib["Mode"+str(i)]['vec'] = vec
    Lib['coeff_bounds'] = coeff_bounds
    return Lib

############################# Construct Library ###############################
def getROMLibrary(tumor, V, bounds, num = 2, zipped = None):
    """
    Builds the operator library needed for model
    Default to the full RXDIF w AC chemo model where global diffusivity, local
    prolfieration, and global treatment operators are needed; include model
    requirements in a zip if changing

    Parameters
    ----------
    tumor : dict
        Contains tumor data needed for snapshot prep.
    V : TYPE
        Projection basis.
    bounds : dict
        Parameter bounds for each operator based parameter.
    num : integer; default = 2
        Size of library to make, cannot be less than 2 or interpolation is invalid.
    zipped : zipped tuple; default = None
        (Libraries needed, corresponding key in bounds, local or global).

    Returns
    -------
    Library
        Nested dictionary containing all operators in reduced space

    """
    if zipped == None:
        required_ops = ('A','B','H','T')
        params_ops = ('d','k','k','alpha')
        type_ops = ('g','l','l','g')
        zipped = zip(required_ops, params_ops, type_ops)
    
    Library = {}
    for name, param, kind in zipped:
        if kind.lower() == 'g':
            Library[name] = _constructGlobalLibrary(name, tumor, V,
                                                   bounds[param], num)
        elif kind.lower() == 'l':
            Library[name] = _constructLocalLibrary(name, tumor, V,
                                                  bounds[param], num)

    return Library


########################### Library interpolation #############################
def interpolateGlobal(Lib, val):
    if len(Lib['vec']) == 2:
        dif = abs(val - Lib['vec'])
        dist = Lib['vec'][1] - Lib['vec'][0]
        return Lib['0']*(1 - (dif[0]/dist)) + Lib['1']*(1 - (dif[1]/dist))
    else:
        ind  = Lib['vec'][Lib['vec'] <= val].size - 1
        dif = np.zeros((2))
        dif[0] = abs(val - Lib['vec'][ind])
        dif[1] = abs(val - Lib['vec'][ind+1])
        dist = Lib['vec'][ind] - Lib['vec'][ind+1]
        return Lib['0']*(1 - (dif[0]/dist)) + Lib['1']*(1 - (dif[1]/dist))

def interpolateLocal(Libs, vals):
    ops = np.zeros(Libs['Mode0']['0'].shape)
    for i in range(len(Libs)-1):
        Lib_curr = Libs['Mode'+str(i)]
        if len(Lib_curr['vec']) == 2:
            dif = abs(vals[i] - Lib_curr['vec'])
            dist = Lib_curr['vec'][1] - Lib_curr['vec'][0]
            ops += (Lib_curr['0']*(1 - (dif[0]/dist)) 
                    + Lib_curr['1']*(1 - (dif[1]/dist)))
        else:
            ind  = Lib_curr['vec'][Lib_curr['vec'] <= vals[i]].size - 1
            dif = np.zeros((2))
            dif[0] = abs(vals[i] - Lib_curr['vec'][ind])
            dif[1] = abs(vals[i] - Lib_curr['vec'][ind+1])
            dist = Lib_curr['vec'][ind] - Lib_curr['vec'][ind+1]
            ops += np.array(Lib_curr[str(ind)]*(1 - (dif[0]/dist)) 
                            + Lib_curr[str(ind+1)]*(1 - (dif[1]/dist)))
            
    return ops

def getOperators(curr, ROM):
    """
    Parameters
    ----------
    curr : dict
        Keys are parameter names, with value attached
    ROM : dict
        ROM library

    Returns
    -------
    operators : dict
        Operators for all parameters

    """
    operators = {}
    for elem in curr:
        if elem == 'd':
            operators['A'] = interpolateGlobal(ROM['Library']['A'],curr[elem])
        elif elem == 'k':
            if curr[elem].size == 1:
                operators['B'] = interpolateGlobal(ROM['Library']['B'],curr[elem])
                operators['H'] = interpolateGlobal(ROM['Library']['H'],curr[elem])
            else:
                operators['B'] = interpolateLocal(ROM['Library']['B'],curr[elem])
                operators['H'] = interpolateLocal(ROM['Library']['H'],curr[elem])
        elif elem == 'alpha':
            operators['T'] = interpolateGlobal(ROM['Library']['T'],curr[elem])
    return operators