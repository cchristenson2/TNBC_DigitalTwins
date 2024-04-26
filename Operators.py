# Operators.py
""" Required functions for operator building and reducing
    - Build operators required for RXDIF w AC model (either full (not written right now) or reduced)
        - A: diffusivity operator, scalar or local field
        - B: linear proliferation operator, scalar or local field
        - H: quadratic proliferation operator, scalar or local field
        - T: treatment operator, scalar

Last updated: 4/23/2024
"""

import numpy as np

###############################################################################
# Full sized operators - not needed for ROM, reduced memory requirements using
# approximate method



###############################################################################
# Reduce full sized operators


###############################################################################
# Internal for building reduced operators
def getARow(ind, d, bcs, h):
    if bcs.ndim == 3: #2D data input
        sy,sx = bcs.shape[0:2]
        n = sy*sx
        bcs = np.reshape(bcs, (-1,2))
    else: #3D data input
        sy,sx,sz = bcs.shape[0:3]
        n = sy*sx*sz 
        bcs = np.reshape(bcs, (-1,3))
        row_z = np.zeros([1,n])
        
    row = np.zeros([1,n])
    row_x = np.zeros([1,n])
    row_y = np.zeros([1,n])
    #X - Direction boundaries
    if bcs[ind,0] == 0:
        row_x[[0,0,0],[ind-1, ind, ind+1]] = d * np.array([1, -2, 1]) / h[0]**2
    elif bcs[ind,0] == -1:
        row_x[[0,0],[ind, ind+1]] = d * np.array([-2, 2]) / h[0]**2
    elif bcs[ind,0] == 1:
        row_x[[0,0],[ind-1, ind]] = d * np.array([2, -2]) / h[0]**2
    #Y - Direction boundaries
    if bcs[ind,1] == 0:
        row_y[[0,0,0],[ind-sx, ind, ind+sx]] = d * np.array([1, -2, 1]) / h[1]**2
    elif bcs[ind,1] == -1:
        row_y[[0,0],[ind, ind+sx]] = d * np.array([-2, 2]) / h[1]**2
    elif bcs[ind,1] == 1:
        row_y[[0,0],[ind-sx, ind]] = d * np.array([2, -2]) / h[1]**2
    #Z - Direction boundaries - if applicable
    if bcs.ndim == 4:
        if bcs[ind,2] == 0:
            row_z[[0,0,0],[ind-sx*sy, ind, ind+sx*sy]] = d * np.array([1, -2, 1]) / h[2]**2
        elif bcs[ind,2] == -1:
            row_z[[0,0],[ind, ind+sx*sy]] = d * np.array([-2, 2]) / h[2]**2
        elif bcs[ind,2] == 1:
            row_z[[0,0],[ind-sx*sy, ind]] = d * np.array([2, -2]) / h[2]**2
        row += row_z
    row += row_x + row_y
        
    return row

###############################################################################
# Build reduced operators directly
def buildReduced_A(V, tumor, d):
    """
    Build reduced diffusivity operator for diffusivity d, based on tumor size
    with boundary conditions in tumor['bcs']
    
    Parameters
    ----------
    V : ndarray
        Projection basis.
    tumor : dictionary
        Contains all tumor information.
    d : ndarray or float
        Diffusivity value (scalar or local (not written yet)).

    Returns
    -------
    A_r.
        Reduced A operator
    """
    n, r = V.shape
    A_r = np.zeros([r,r])
    for i in range(n):
        row = getARow(i, d, tumor['bcs'], tumor['h'])
        # print((row @ V).shape)
        # print(V[np.newaxis,i,:].T.shape)
        A_r += V[np.newaxis,i,:].T @ (row @ V)
        
    return A_r

def buildReduced_B(V, tumor, k):
    """
    Build reduced linear proliferation operator for proliferation k, based on tumor size
    
    Parameters
    ----------
    V : ndarray
        Projection basis.
    tumor : dictionary
        Contains all tumor information.
    k : ndarray or float
        Proliferation value (scalar or local).

    Returns
    -------
    B_r.
        Reduced B operator
    """
    n, r = V.shape
    B_r = np.zeros([r,r])
    if k.size == 1:
        k = k * np.ones([tumor['AUC'].size])
    for i in range(n):
        B_r += (V[np.newaxis,i,:] * k[i]).T @ V[np.newaxis,i,:]
    return B_r
    
def buildReduced_H(V, tumor, k):
     """
     Build reduced quadratic proliferation operator for proliferation k, based on tumor size
     
     Parameters
     ----------
     V : ndarray
         Projection basis.
     tumor : dictionary
         Contains all tumor information.
     k : ndarray or float
         Proliferation value (scalar or local).

     Returns
     -------
     H_r.
         Reduced H operator
     """
     n, r = V.shape
     H_r = np.zeros([r,r**2])
     if k.size == 1:
         k = k * np.ones([tumor['AUC'].size])
     for i in range(n):
         H_r += (V[np.newaxis,i,:] * k[i]).T @ np.kron(V[np.newaxis,i,:],V[np.newaxis,i,:])
     
     return H_r
    
def buildReduced_T(V, tumor, alpha):
      """
      Build reduced treatment operator for alpha, based on tumor size and drug 
      concentration map, tumor['AUC']
      
      Parameters
      ----------
      V : ndarray
          Projection basis.
      tumor : dictionary
          Contains all tumor information.
      alpha : float
          Drug Efficacy value.

      Returns
      -------
      T_r.
          Reduced T operator
      """      
      n, r = V.shape
      T_r = np.zeros([r,r])
      AUC = np.reshape(tumor['AUC'],(-1,1))
      for i in range(n):
          T_r += V[np.newaxis,i,:].T * (alpha * AUC[i]) @ V[np.newaxis,i,:]
      
      return T_r
