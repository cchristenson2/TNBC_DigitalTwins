import numpy as np
import os

import LoadData as ld
import ForwardModels as fwd
import Calibrations as cal
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #Set paths
    home = os.path.dirname(os.path.dirname(os.getcwd()))
    datapath = home + '\Data\PatientData_ungrouped\\'
    #Get tumor information in folder
    files = os.listdir(datapath)
    
    #Load the first patient
    tumor = ld.LoadTumor_mat(datapath + files[0], crop2D = True)
    
    #Set up call to calibrate with FOM, 2D since 3D takes a long time
    params = ([fwd.Parameter('d','g'), fwd.Parameter('k','l'), fwd.Parameter('alpha','g'),
               fwd.Parameter('beta_a','f'), fwd.Parameter('beta_c','f')])
    
    params[0].setBounds(np.array([1e-6,1e-3]))
    params[1].setBounds(np.array([1e-6,0.1]))
    params[2].setBounds(np.array([1e-6,0.8]))
    
    tumor['N'] = tumor['N'][:,:,(0,1)]
    tumor['t_scan'] = tumor['t_scan'][:-1]
    
    #Call calibration
    params, stats = cal.calibrateRXDIF_LM(tumor, params, parallel = True, options = {'max_it': 100,'j_freq': 3})

    # for elem in params:
    #     print(elem.get())
    
    plt.figure()    
    plt.imshow(params[1].get())
    
    dt = 0.5
    N0 = tumor['N'][:,:,0]
    
    trx_params = {'t_trx': tumor['t_trx'], 'AUC': tumor['AUC']}
    trx_params['beta'] = np.array([params[3].get(), params[4].get()])
    
    N_sim, _ = fwd.RXDIF_2D_wAC(N0, params[1].get(), params[0].get(), params[2].get(), trx_params, tumor['t_scan'][1:], tumor['h'], dt, tumor['bcs'])
    
    
    figure, ax = plt.subplots(1,3, layout="constrained")
    p = ax[0].imshow(tumor['N'][:,:,1], clim=(0,1),cmap='jet')
    ax[0].set_title('V2 Measured')
    ax[0].set_xticks([]), ax[0].set_yticks([])
    plt.colorbar(p,fraction=0.046, pad=0.04)

    p = ax[1].imshow(N_sim[:,:,0], clim=(0,1),cmap='jet')
    ax[1].set_title('V2 Simulation')
    ax[1].set_xticks([]), ax[1].set_yticks([])
    plt.colorbar(p,fraction=0.046, pad=0.04)
    
    p = ax[2].scatter(tumor['N'][:,:,1], N_sim[:,:,0], facecolors='none', edgecolors='b')
    p1 = ax[2].plot([0,1],[0,1],'--r')
    ax[2].set_aspect('equal')
    
    
    
    
    
    