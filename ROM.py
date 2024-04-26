# ROM.py
""" Building ROM for reaction diffusion model through projections
    - Augment data
    - Construct basis and operators
    

Last updated: 4/23/2024
"""

import numpy as np
# import scipy.sparse.linalg as la
import scipy.linalg as la
import scipy.ndimage as ndi

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
            for i in range(nt-1):
                N_mid = (N_aug[:,:,i] + N_aug[:,:,i])/2
                N_mid = np.squeeze(ndi.gaussian_filter(N_mid, 0.5))
                N_aug = np.insert(N_aug,i+1,N_mid,axis=2)
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
            for i in range(nt-1):
                N_mid = (N_aug[:,:,:,i] + N_aug[:,:,:,i])/2
                N_mid = np.squeeze(ndi.gaussian_filter(N_mid, 0.5))
                N_aug = np.insert(N_aug,i+1,N_mid,axis=3)
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

    # U, s = la.svds(snapshots, full_r)[0,1]
    U, s, _ = la.svd(snapshots)
    
    if r == 0:
        cummulative_e = np.cumsum(s)
        max_e = np.max(cummulative_e)
        r = cummulative_e[cummulative_e < max_e * 0.995].size + 1
        
    V = U[:,1:r+1]
    return V

###############################################################################
# Construct ROM models
def constructROM_RXDIF(tumor, bounds, augmentation = 'average', depth = 8, samples = None, r = 0, zipped = None):
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
    Library = lib.getROMLibrary(tumor, V, bounds)
    
    #Reduce measured tumor data
    ROM['V'] = V
    ROM['Library'] = Library
    
    return ROM

