# ReducedModel.py
""" Building ROM for reaction diffusion model through projections
*getProjectionBasis(snapshots, r = 0)
    - Uses scipy's SVD to build the basis starting from snapshots
*constructROM_RXDIF(tumor, bounds, augmentation = 'average', depth = 8, samples = None, r = 0, zipped = None, num = 2)
    - Constructs ROM from the data in zipped, starting from tumor NTC maps
*augmentAverage(tumor, depth)
    - Augments data through sequential averages and smoothing directly on measured data
*augmentSample(tumor, samples)
    - Augments data through simulations with the parameters contained in samples  
*visualizeBasis(ROM, shape)
    - plots the central slice (if 3D) of each mode for visualization

Last updated: 4/30/2024
"""

import numpy as np
# import scipy.sparse.linalg as la
import scipy.linalg as la
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

import Library as lib

###############################################################################
# Augmentation functions
def augmentAverage(tumor, depth):
    """
    Augments data by averaging the tumor NTC maps and applying a filter
    Performs 'depth' averages on the dataset
    """
    N_aug = tumor['N']
    if tumor['Mask'].ndim == 2:
        mask = np.sum(N_aug,axis=2)
        mask[mask > 0] = 1
        for j in range(depth):
            nt = N_aug.shape[2]
            curr = 0;
            for i in range(nt-1):
                N_mid = (N_aug[:,:,i+curr] + N_aug[:,:,i+1+curr])/2
                N_mid = np.squeeze(ndi.gaussian_filter(N_mid, 0.5))
                # if j == 0:
                #     N_mid = np.squeeze(ndi.gaussian_filter(N_mid, 0.5))
                N_aug = np.insert(N_aug,i+1+curr,N_mid,axis=2)
                curr += 1
        for i in range(N_aug.shape[2]):
            temp = N_aug[:,:,i]
            temp[mask!=1] = 0
            N_aug[:,:,i] = temp
        N_aug = np.reshape(N_aug, (-1,N_aug.shape[2]))
        
    else:
        mask = np.sum(N_aug,axis=3)
        mask[mask > 0] = 1
        for j in range(depth):
            nt = N_aug.shape[3]
            curr = 0;
            for i in range(nt-1):
                N_mid = (N_aug[:,:,:,i+curr] + N_aug[:,:,:,i+1+curr])/2
                N_mid = np.squeeze(ndi.gaussian_filter(N_mid, 0.5))
                N_aug = np.insert(N_aug,i+1+curr,N_mid,axis=3)
        for i in range(N_aug.shape[3]):
            temp = N_aug[:,:,:,i]
            temp[mask!=1] = 0
            N_aug[:,:,:,i] = temp
        N_aug = np.reshape(N_aug, (-1,N_aug.shape[3]))
            
    return N_aug
    
def augmentSample(tumor, samples):
    """
    Augments data by simulating to final time of tumor['t_scan'] using parameters
    contained in samples
    
    Need a way to pass in model and parameter formatting? We will probably need this anyway
    """
    N_aug = tumor['N']
    
    return N_aug

###############################################################################
# Construct POD basis
def getProjectionBasis(snapshots, r = 0):
    """
    Get orthogonal projecion matrix from snapshot data using SVD
    
    Parameters
    ----------
    snapshots : ndarray
        Sampled data representations
    r : TYPE, optional; default = 0
        Number of modes to retain, if 0 selects r based on cummulative energy decay

    Returns
    -------
    V : ndarray
        Projection matrix
    """
    # if r == 0:
    #     full_r = 20
    # else:
    #     full_r = r
    # U, s, _ = la.svds(snapshots, full_r)
    # U = np.flip(U, axis = 1)
    # s = np.flip(s, axis = 0)
    
    U, s, _ = la.svd(snapshots)
    
    if r == 0:
        cummulative_e = np.cumsum(s)
        max_e = np.max(cummulative_e)
        r = cummulative_e[cummulative_e < max_e * 0.995].size
        
    V = U[:,0:r+1]
    return V

###############################################################################
# Construct ROM models
def constructROM_RXDIF(tumor, bounds, augmentation = 'average', depth = 8, 
                       samples = None, r = 0, zipped = None, num = 2):
    """
    Build the reduced order model from data in tumor. Defaults to the complete
    RXDIF w AC chemo model, pass zipped to try different model constraints
    
    Parameters
    ----------
    tumor : dictionary
        Contains tumor data needed for snapshot prep.
    bounds : dictionary
        Contains bounds for each calibrated parameter in the RXDIF w/ AC model.
    augmentation : string, optional; default = 'average'
        Method for augmenting snapshots.
        Valid: average, sample (not written yet).
        *sample requires parameter sets provided from MCMC calibration.
    depth : integer, optional; default = None
    samples : ndarray, optional; default = None
        parameter samples.
    r : integer, optional; default = 0
        Desired rank of projection matrix. If r = 0 the rank is based on 
        cummulative energy.
    zipped : zipped tuple; default = None
        (Libraries needed, corresponding key in bounds, local or global).

    Returns
    -------
    ROM : dictionary
        Library, Projection basis, reduced tumor
    """
    #Augment the data
    ROM = {}
    if augmentation == 'average':
        snapshots = augmentAverage(tumor, depth)
    elif augmentation == 'sample':
        if samples == None:
            raise ValueError('augmentation specified as ''sample'' but samples were not provided')
        else:
            snapshots = augmentSample(tumor, samples)

    #Build projecton basis
    V = getProjectionBasis(snapshots, r)
    
    #Build library for operators required for RXDIF w AC model
    Library = lib.getROMLibrary(tumor, V, bounds, num = num, zipped = zipped)
    
    #Reduce tumor inputs
    ReducedTumor = {}
    if tumor['Mask'].ndim == 2:
        ReducedTumor['N_r'] = V.T @ np.reshape(tumor['N'],(-1,tumor['N'].shape[2]))
        if 'Future N' in tumor.keys():
            ReducedTumor['Future N_r'] = V.T @ np.reshape(tumor['Future N'],
                                                          (-1,tumor['Future N'].shape[2]))
    else:
        ReducedTumor['N_r'] = V.T @ np.reshape(tumor['N'],(-1,tumor['N'].shape[3]))
        if 'Future N' in tumor.keys():
            ReducedTumor['Future N_r'] = V.T @ np.reshape(tumor['Future N'],
                                                          (-1,tumor['Future N'].shape[3]))
    
    #Reduce measured tumor data
    ROM['V'] = V
    ROM['Library'] = Library
    ROM['ReducedTumor'] = ReducedTumor
    
    return ROM

###############################################################################
#Functions for ROM usage
def visualizeBasis(ROM, shape):
    n,k = ROM['V'].shape
    figure, ax = plt.subplots(2,int(np.ceil(k/2)), layout="constrained")
    ax = np.reshape(ax,(-1))
    for i in range(k):
        mode = np.reshape(ROM['V'][:,i], shape)
        if mode.ndim == 3:
            s = round(mode.shape[2]/2)-1
            mode = mode[:,:,s]
        p = ax[i].imshow(mode)
        ax[i].set_title('Mode '+str(i+1))
        plt.colorbar(p,fraction=0.046, pad=0.04)
    