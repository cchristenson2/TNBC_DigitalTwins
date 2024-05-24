# LoadData.py
""" Loading .mat files and processing for digital twins
*LoadData_mat(location, downsample='true', inplane=4, inslice=1, crop2D = 'false')

Extra:
*downSample_2D(N, AUC, Mask, Tissues, h, inplane, inslice)
*downSample_3D(N, AUC, Mask, Tissues, h, inplane, inslice)
*buildBoundaries_2D(Mask)
*buildBoundaries_3D(Mask)

Last updated: 5/2/2024
"""

import numpy as np

from scipy import io as io

from scipy.interpolate import RegularGridInterpolator

pack_dens = 0.7405
cell_size = 4.189e-6
    
############################ Load .mat files ##################################
# Load full resolution and downsample if requested
def LoadTumor_mat(location, downsample = True, inplane = 4, inslice = 1, 
                  crop2D = False, split = None):
    """
    Parameters
    ----------
    location : string
        Data location in folder
    downsample : boolean; default = True
        Should the data be downsampled
    inplane : float, optional; default = 4
        Factor to reduce the in-plane resolution by
    inslice : float, optional; default = 1
        Factor to reduce the slice resolution by
    crop2D : boolean, optional; default = False
        Should the outputs be cropped to 2D at the center slice
    split : integer, optional; default = None
        Should NTC maps and scan info be split based on what has been "acquired"
        If integer input is provided, the data after visit "split" is added to
        seperate category in the tumor dictionary

    Returns
    -------
    tumor
        dictionary containing all scan info needed for digital twins
    """
    data = _loadmat(location)
    tumor = {}
    
    #Pull out treatment regimen info
    sch = data['schedule_info']
    times = sch['times']
    times = np.cumsum(times)
    times_labels = sch['schedule']
    scan_times = times[times_labels=='S']
    trx_times = times[times_labels==sch['drugtype']]
    dimensions = sch['imagedims']
    pcr = data['pcr']
    
    #Pull out and prep scan data
    N = _stackNTC(data['full_res_dat'])
    AUC = data['full_res_dat']['AUC']
    Mask = data['full_res_dat']['BreastMask']
    Tissues = data['full_res_dat']['Tissues']
    
    #Downsample
    if downsample:
        if N.ndim == 3:
            N, AUC, Mask, Tissues, dimensions = downSample_2D(N, AUC, Mask, 
                                                              Tissues, dimensions,
                                                              inplane, inslice)
        else:
            N, AUC, Mask, Tissues, dimensions = downSample_3D(N, AUC, Mask, 
                                                              Tissues, dimensions,
                                                              inplane, inslice)
    
    #Convert to cell fraction
    theta = pack_dens * np.prod(dimensions) / cell_size
    N = N / theta
    
    #2D Crop
    if crop2D:
        s = int(round(N.shape[2]/2))-1
        N = np.squeeze(N[:,:,s,:])
        AUC = np.squeeze(AUC[:,:,s])
        Mask = np.squeeze(Mask[:,:,s])
        Tissues = np.squeeze(Tissues[:,:,s])
        
    #Build boundaries
    if Mask.ndim == 2:
        bcs = buildBoundaries_2D(Mask)
    else:
        bcs = buildBoundaries_3D(Mask)
        
    #Split data
    if split != None:
        if Mask.ndim == 2:
            saved_N = N[:,:,split:]
            N = np.delete(N,np.s_[split:],axis=2)
        else:
            saved_N = N[:,:,:,split:]
            N = np.delete(N,np.s_[split:],axis=3)
        saved_times = scan_times[split:]
        scan_times = np.delete(scan_times,np.s_[split:],axis=0)
        tumor['Future N'] = saved_N
        tumor['Future t_scan'] = saved_times
        
    #Save in tumor
    tumor['N'] = N
    tumor['AUC'] = AUC
    tumor['Mask'] = Mask
    tumor['Tissues'] = Tissues
    
    tumor['bcs'] = bcs
    
    tumor['t_scan'] = scan_times
    tumor['t_trx'] = trx_times
    tumor['h'] = dimensions
    tumor['pcr_status'] = pcr
    
    tumor['theta'] = theta
    
    return tumor
 
def LoadInsilicoTumor_mat(location, split = None):
   tumor = _loadmat(location)['tumor']
   tumor['h'] = np.array([tumor['h'],tumor['h'],tumor['dz']])
   if tumor['Mask'].ndim == 2:
       tumor['bcs'] = buildBoundaries_2D(tumor['Mask'])
   else:
       tumor['bcs'] = buildBoundaries_3D(tumor['Mask'])
   if split != None:
       if tumor['Mask'].ndim == 2:
           saved_N = tumor['N'][:,:,split:]
           tumor['N'] = np.delete(tumor['N'],np.s_[split:],axis=2)
       else:
           saved_N = tumor['N'][:,:,:,split:]
           tumor['N'] = np.delete(tumor['N'],np.s_[split:],axis=3)
       saved_times = tumor['t_scan'][split:]
       tumor['t_scan'] = np.delete(tumor['t_scan'],np.s_[split:],axis=0)
       tumor['Future N'] = saved_N
       tumor['Future t_scan'] = saved_times
   return tumor    
    
########################## Processing functions ###############################
def downSample_2D(N, AUC, Mask, Tissues, h, inplane, inslice):
    """
    Downsamples all scan derived inputs based inplane and inslice factor
    """
    #Define original grid
    sy,sx,nt = N.shape
    x = np.linspace(h[0],sx*h[0],sx)
    y = np.linspace(h[1],sy*h[1],sy)
    
    #Build new spatial grid
    h_down = h.copy()
    h_down[0] = sx*h[0] / round(sx/inplane)
    h_down[1] = sy*h[1] / round(sy/inplane)
    X_coarse,Y_coarse = np.meshgrid(np.linspace(h_down[0],sx*h[0],
                                                round(sx/inplane)),
                                    np.linspace(h_down[1],sy*h[1],
                                                round(sy/inplane)),
                                    indexing='ij')
    ## NTC downsampling
    N_norm = N / np.prod(h) #Normalize by voxel volume
    interp = RegularGridInterpolator((x,y), N_norm) #Construct interpolator
    N_norm_down = interp((X_coarse, Y_coarse)) #Interpolate
    N_down = N_norm_down * np.prod(h_down) #Unnormalize
    
    ## AUC downsampling
    interp = RegularGridInterpolator((x,y), AUC)
    AUC_down = interp((X_coarse, Y_coarse))
    
    ## Mask downsapling
    interp = RegularGridInterpolator((x,y), Mask)
    Mask_down = interp((X_coarse, Y_coarse))
    Mask_down[Mask_down>0] = 1 #Convert back to mask of 1s and 0s
    
    ## Tissues downsapling
    interp = RegularGridInterpolator((x,y), Tissues)
    Tissues_down = interp((X_coarse, Y_coarse))
    Tissues_down = np.round(Tissues_down) #Ensure values are 0, 1, or 2
    Tissues_down[Tissues_down>2] = 2
    Tissues_down[Tissues_down<0] = 0
        
    return N_down, AUC_down, Mask_down, Tissues_down, h_down
        
    return N, AUC, Mask, Tissues

def downSample_3D(N, AUC, Mask, Tissues, h, inplane, inslice):
    """
    Downsamples all scan derived inputs based inplane and inslice factor
    """
    #Define original grid
    sy,sx,sz,nt = N.shape
    x = np.linspace(h[0],sx*h[0],sx)
    y = np.linspace(h[1],sy*h[1],sy)
    z = np.linspace(h[2],sz*h[2],sz)
    
    #Build new spatial grid
    h_down = h.copy()
    h_down[0] = sx*h[0] / round(sx/inplane)
    h_down[1] = sy*h[1] / round(sy/inplane)
    h_down[2] = sz*h[2] / round(sz/inslice)
    X_coarse,Y_coarse,Z_coarse = np.meshgrid(np.linspace(h_down[0],sx*h[0],
                                                         round(sx/inplane)),
                                             np.linspace(h_down[1],sy*h[1],
                                                         round(sy/inplane)),
                                             np.linspace(h_down[2],sz*h[2],
                                                         round(sz/inslice)), 
                                             indexing='ij')
    ## NTC downsampling
    N_norm = N / np.prod(h) #Normalize by voxel volume
    interp = RegularGridInterpolator((x,y,z), N_norm) #Construct interpolator
    N_norm_down = interp((X_coarse, Y_coarse, Z_coarse)) #Interpolate
    N_down = N_norm_down * np.prod(h_down) #Unnormalize
    
    ## AUC downsampling
    interp = RegularGridInterpolator((x,y,z), AUC)
    AUC_down = interp((X_coarse, Y_coarse, Z_coarse))
    
    ## Mask downsapling
    interp = RegularGridInterpolator((x,y,z), Mask)
    Mask_down = interp((X_coarse, Y_coarse, Z_coarse))
    Mask_down[Mask_down>0] = 1 #Convert back to mask of 1s and 0s
    
    ## Tissues downsapling
    interp = RegularGridInterpolator((x,y,z), Tissues)
    Tissues_down = interp((X_coarse, Y_coarse, Z_coarse))
    Tissues_down = np.round(Tissues_down) #Ensure values are 0, 1, or 2
    Tissues_down[Tissues_down>2] = 2
    Tissues_down[Tissues_down<0] = 0
        
    return N_down, AUC_down, Mask_down, Tissues_down, h_down

def buildBoundaries_2D(Mask):
    """
    Zero flux boundary conditions on all borders of mask
    
    Returns bcs
        ndarray with size [mask.shape, mask.ndims]
            Each location has vector [x-type, y-type]
    """
    bcs = np.zeros((Mask.shape+(2,)))
    for y in range(Mask.shape[0]):
        for x in range(Mask.shape[1]):
            if Mask[y,x] == 0:
                bcs[y,x,:] = np.array([2,2])
            else:
                boundary = np.array([0,0])
                #check X
                if x == 0:
                    boundary[0] = -1
                elif x == Mask.shape[1]-1:
                    boundary[0] = 1
                else:
                    if Mask[y,x-1] == 0:
                        boundary[0] = -1
                    elif Mask[y,x+1] == 0:
                        boundary[0] = 1            
                #check Y
                if y == 0:
                    boundary[1] = -1
                elif y == Mask.shape[0]-1:
                    boundary[1] = 1
                else:
                    if Mask[y-1,x] == 0:
                        boundary[1] = -1
                    elif Mask[y+1,x] == 0:
                        boundary[1] = 1
                bcs[y,x,:] = boundary
    return bcs
    
def buildBoundaries_3D(Mask):
    """
    Zero flux boundary conditions on all borders of mask
    
    Returns bcs
        ndarray with size [mask.shape, mask.ndims]
            Each location has vector [x-type, y-type, z-type]
    """
    bcs = np.zeros((Mask.shape+(3,)))
    for z in range(Mask.shape[2]):
        for y in range(Mask.shape[0]):
            for x in range(Mask.shape[1]):
                if Mask[y,x,z] == 0:
                    bcs[y,x,z,:] = np.array([2,2,2])
                else:
                    boundary = np.array([0,0,0])
                    #check X
                    if x == 0:
                        boundary[0] = -1
                    elif x == Mask.shape[1]-1:
                        boundary[0] = 1
                    else:
                        if Mask[y,x-1,z] == 0:
                            boundary[0] = -1
                        elif Mask[y,x+1,z] == 0:
                            boundary[0] = 1            
                    #check Y
                    if y == 0:
                        boundary[1] = -1
                    elif y == Mask.shape[0]-1:
                        boundary[1] = 1
                    else:
                        if Mask[y-1,x,z] == 0:
                            boundary[1] = -1
                        elif Mask[y+1,x,z] == 0:
                            boundary[1] = 1
                    #check Z
                    if z == 0:
                        boundary[2] = -1
                    elif z == Mask.shape[2]-1:
                        boundary[2] = 1
                    else:
                        if Mask[y,x,z-1] == 0:
                            boundary[2] = -1
                        elif Mask[y,x,z+1] == 0:
                            boundary[2] = 1
                    bcs[y,x,z,:] = boundary
    return bcs
   
########### Internal load for restructuring data from MATLAB ##################
def _loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], io.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, io.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def _stackNTC(dat):
    """
    Get all NTC maps and stack in order
    Returns 3D array if 2D NTC maps or 4D array if 3D NTC maps
    """
    names = []
    for key in dat.keys():
        if 'NTC' in key:
            names.append(key)
         
    names = sorted(names)
    for key in names:
        if 'NTCs' not in locals():
            NTCs = dat[key]
            NTCs = np.expand_dims(NTCs,NTCs.ndim)
        else:
            NTCs = np.append(NTCs, np.expand_dims(dat[key],NTCs.ndim - 1), 
                             NTCs.ndim - 1)
    return NTCs