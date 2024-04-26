# Library.py
""" Construct complete libraries for a given model and interpolate
    - Construct library:
            - RXDIF w AC model (diffusivity, proliferation, and treatment operators)
    - Interpolate from library:
            - Globally controlled parameters
            - Locally controlled parameters

Last updated: 4/23/2024
"""

import numpy as np

import Operators as op

###############################################################################
# Internal constructions
def constructGlobalLibrary(var, tumor, V, bounds, num):
    """
    Used to build an operator library with num components for operator label 'var'
    Works for all operators based on global parameters used in the RXDIF model
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

def constructLocalLibrary(var, tumor, V, bounds, num):
    """
    Used to build an operator library with num components for operator label 'var'
    Only works for local proliferation based operators
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
        temp[0,0] = V[:,i] @ test_vec
        test_vec = np.zeros([n,1])
        test_vec[V[:,i] >= 0] = bounds[0]
        test_vec[V[:,i] < 0] = bounds[1]
        temp[0,1] = V[:,i] @ test_vec
        temp = np.sort(temp)
        coeff_bounds[i,:] = temp
        
        Lib["Mode"+str(i)] = {}
        vec = np.linspace(temp[0,0],temp[0,1],num)
        for j in range(vec.size):
            if var == 'B':
                Lib["Mode"+str(i)][str(j)] = op.buildReduced_B(V, tumor, V * vec[j])
            elif var == 'H':
                Lib["Mode"+str(i)][str(j)] = op.buildReduced_H(V, tumor, V * vec[j])
        Lib["Mode"+str(i)]['vec'] = vec
    Lib['coeff_bounds'] = coeff_bounds
    return Lib

###############################################################################
# Construct library
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
        type_ops = ('G','L','L','G')
        zipped = zip(required_ops, params_ops, type_ops)
    
    Library = {}
    for name, param, kind in zipped:
        if kind == 'G':
            Library[name] = constructGlobalLibrary(name, tumor, V, bounds[param], num)
        elif kind == 'L':
            Library[name] = constructLocalLibrary(name, tumor, V, bounds[param], num)

    return Library


###############################################################################
# Library interpolation
def interpolateGlobal(Lib, val):
    return 0

def interpolateLocal(Lib, vals):
    return 0



