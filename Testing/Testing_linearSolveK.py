import numpy as np
import os

import LoadData as ld
import ForwardModels as fwd
import Calibrations as cal
import ReducedModel as rm

import matplotlib.pyplot as plt

import cvxpy as cp

def forceBounds(curr_param, V, bounds, coeff_bounds = None, x0 = None):
    test = V @ curr_param
    if (len(np.nonzero((test < bounds[0]) & (abs(test) > 1e-6))) != 0 or len(np.nonzero(test > bounds[1])) != 0):
        
        indices = np.nonzero(abs(test)>1e-6)[0]
        A = np.concatenate((V[np.s_[indices,:]], -1*V[np.s_[indices,:]]), axis = 0)
        B = np.squeeze(np.concatenate((bounds[1]*np.ones([len(indices),1]), -1*bounds[0]*np.ones([len(indices),1])),axis = 0))
        
        print(A.shape)
        print(B.shape)
        
        ### CVXPY optimization test
        x = cp.Variable(V.shape[1])
        if x0 is None:
            x.value = curr_param
        else:
            x.value = x0    
        objective = cp.Minimize(cp.sum_squares(V@x - test))

        # constraints = cp.Constraint([A@x <= B])
        constraints = [A@x <= B]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=True, eps_abs = 1e-10)
        curr_param = x.value
        
        
        
    return curr_param

#Set paths
home = os.path.dirname(os.getcwd())
datapath = home + '\Data\PatientData_ungrouped\\'
#Get tumor information in folder
files = os.listdir(datapath)

#Load the first patient
print(files[0])
tumor = ld.LoadTumor_mat(datapath + files[0], crop2D = True, split = 2)

bounds = {}
bounds['d'] = np.array([1e-6, 1e-3])
bounds['k'] = np.array([1e-6, 0.1])
bounds['alpha'] = np.array([1e-6, 0.8])

ROM = rm.constructROM_RXDIF(tumor, bounds)

#Params needed for solve
V = ROM['V']
bounds = np.array([1e-6, 0.1])
coeff_bounds = ROM['Library']['B']['coeff_bounds']

np.random.seed(1)
#Build bad k map
sy,sx,nt = tumor['N'].shape
temp_k = np.random.uniform(-0.5, 0.5, (sy,sx))

temp_k = np.reshape(V @ (V.T @ np.reshape(temp_k, (-1))), (sy,sx))

plt.figure()
plt.imshow(temp_k)
plt.colorbar()
plt.title('Initial k map')


#Force bounds on k
# forced_k_r = forceBounds(V.T @ np.reshape(temp_k, (-1)), V, bounds, coeff_bounds)
test = V @ (V.T @ np.reshape(temp_k, (-1)))
if (len(np.nonzero((test < bounds[0]) & (abs(test) > 1e-6))) != 0 or len(np.nonzero(test > bounds[1])) != 0):
    
    indices = np.nonzero(abs(test)>1e-6)[0]
    A = np.concatenate((V[np.s_[indices,:]], -1*V[np.s_[indices,:]]), axis = 0)
    B = np.squeeze(np.concatenate((bounds[1]*np.ones([len(indices),1]), -1*bounds[0]*np.ones([len(indices),1])),axis = 0))
    
    print(A.shape)
    print(B.shape)
    
    ### CVXPY optimization test
    x = cp.Variable(V.shape[1])
    objective = cp.Minimize(cp.sum_squares(V@x - test))

    # constraints = cp.Constraint([A@x <= B])
    constraints = [A@x <= B]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=True, eps_abs = 1e-10)
    forced_k_r = x.value

forced_k = np.reshape(V @ (forced_k_r), (sy,sx))

plt.figure()
plt.imshow(forced_k)
plt.colorbar()
plt.title('Bound focrced k map')






