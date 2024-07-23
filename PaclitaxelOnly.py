import numpy as np
import numba as nb
import copy
import matplotlib.pyplot as plt

import Library as lib
import ForwardModels as fwd

############################# Global variables ################################
call_dict = {
    'fwd.RXDIF_2D_wAC': fwd.RXDIF_2D_wAC,
    'fwd.RXDIF_3D_wAC': fwd.RXDIF_3D_wAC,
    'fwd.OP_RXDIF_wAC': fwd.OP_RXDIF_wAC}

def predict_paclitaxel(twin, pac_regimen, dt = 0.5, threshold = 0.20, plot = False,
            visualize = False, parallel = False, treatment = None):
    """
    Runs simulations for all parameter samples based off type of model and
    calibration used. Plots if requested but always stores outputs in the
    twin object for analysis.
    Visualize creates either 2D or 3D plots of simulation vs measured data
        3D uses matplotlibs voxels which can be very slow
    
    Threshold determines cutoff level for cell identification; in 
    volume fraction.
    """
    #First check if parameters have values assigned
    for elem in twin.params:
        if twin.params[elem].value is None:
            raise ValueError('Calibrated parameters do not have values '
                             'assigned. Run calibrateTwin() first')
        
    t = twin.tumor['t_scan'] #Time variables for simulation outputs
    t_type_full = np.zeros(t.shape)   #Type of simulation for each time, 0 = calibrated, 1 = predicted
    if 'Future t_scan' in twin.tumor:
        t = np.append(t, twin.tumor['Future t_scan'])
        t_type_full = np.append(t_type_full, np.ones(twin.tumor['Future t_scan'].shape))
        pred_on = True
    else:
        pred_on = False
    
    t_true = t.copy()
    t = np.append(t, pac_regimen['t_surgery'])
    t_type_full = np.append(t_type_full, 1)
        
    t_ind = t[1:]/dt
    t_type = t_type_full[1:]
    N_meas = twin.tumor['N'].copy()
    if twin.tumor['Mask'].ndim==2:
        N0 = twin.tumor['N'][:,:,0]
        N_cal = twin.tumor['N'][:,:,1:]
        N_cal = N_cal.reshape((-1,np.atleast_3d(N_cal).shape[2]))
        if 'Future N' in twin.tumor:
            N_pred = twin.tumor['Future N'].copy()
            N_pred = N_pred.reshape((-1,np.atleast_3d(N_pred).shape[2]))
            N_meas = np.concatenate((N_meas, twin.tumor['Future N']),axis = 2)
        else:
            N_pred = []
    else:
        N0 = twin.tumor['N'][:,:,:,0]
        N_cal = twin.tumor['N'][:,:,:,1:]
        N_cal = N_cal.reshape((-1,_atleast_4d(N_cal).shape[3]))
        if 'Future N' in twin.tumor:
            N_pred = twin.tumor['Future N'].copy()
            N_pred = N_pred.reshape((-1,_atleast_4d(N_pred).shape[3]))
            N_meas = np.concatenate((N_meas, twin.tumor['Future N']),axis = 3)
        else:
            N_pred = []
    N_cal_r = twin.ROM['ReducedTumor']['N_r'][:,1:]
    if pred_on == True:
        N_pred_r = twin.ROM['ReducedTumor']['Future N_r']
    else:
        N_pred_r = []
    tspan = np.arange(0,t[-1]+dt,dt)
    
    if treatment == None:
        default_trx_params = {'t_trx': twin.tumor['t_trx']}
    else:
        default_trx_params = {}
        for elem in treatment:
            default_trx_params[elem] = treatment[elem]
        
    parameters = twin._unpackParameters()
    calibrations = []
    predictions = []
    calibrations_r = []
    predictions_r = []
    cell_tc = np.array([]).reshape(tspan.size,0)
    volume_tc = np.array([]).reshape(tspan.size,0)
    drug_tc = []
    samples = len(parameters)
        
    sim_save = np.zeros((samples, twin.ROM['V'].shape[1], tspan.size))
    for i in range(samples):
        curr = parameters[i]
        curr['alpha_pac'] = 0.3
        trx_params = default_trx_params.copy()
        trx_params = _update_trx_params(curr, default_trx_params)
        N0 = twin.ROM['ReducedTumor']['N_r'][:,0]
        operators = lib.getOperators(curr,twin.ROM)
        
        sim, drugs = fwd.OP_RXDIF_wAC_wPac(N0, operators, trx_params, tspan, dt, pac_regimen)
        
        calibrations.append(twin.ROM['V'] @ sim[:,t_ind[t_type==0].astype(int)])
        calibrations_r.append(sim[:,t_ind[t_type==0].astype(int)])
        if pred_on == True:
            predictions.append(twin.ROM['V'] @ sim[:,t_ind[t_type==1].astype(int)])
            predictions_r.append(sim[:,t_ind[t_type==0].astype(int)])
        sim_save[i,:,:] = sim
        drug_tc.append(drugs)

    data = sim_save
    cell_tc, volume_tc = _maps_to_timecourse_sampled(data, threshold, twin.tumor['theta'],twin.tumor['h'],twin.ROM['V'])
                
    cell_measured, volume_measured = _maps_to_timecourse(N_meas, threshold, 
                                       twin.tumor['theta'], twin.tumor['h'])
    if samples > 1:
        calibrations = np.mean(np.squeeze(np.array(calibrations)), axis = 0)
        predictions = np.mean(np.squeeze(np.array(predictions)), axis = 0)
    else:
        calibrations = calibrations[0]
        predictions = predictions[0]
        
    simulations = {'cell_tc': cell_tc, 'volume_tc': volume_tc,
                        'maps_cal': calibrations, 'maps_pred': predictions,
                        'maps_r_cal': calibrations_r, 'maps_r_pred': predictions_r,
                        'cell_measured': cell_measured, 
                        'volume_measured':volume_measured,'prediction':pred_on,
                        'N_cal':N_cal, 'N_pred':N_pred, 
                        'N_r_cal':N_cal_r, 'N_r_pred':N_pred_r,
                        'samples':samples,'tspan':tspan, 
                        't_meas':t_true, 't_ind':t_ind, 't_type':t_type_full,
                        'drug_tc':drug_tc, 'trx_params':default_trx_params}
    
    temp = t[-1]/dt
    dist = int((temp - t_ind[t_type==0].astype(int)[-1])/2)
    dist_third = round((temp - t_ind[t_type==0].astype(int)[-1])/3)
    simulations['t_pred_index'] = t_ind[t_type==0].astype(int)[-1]
    simulations['t_mid_index'] = t_ind[t_type==0].astype(int)[-1] + dist
    simulations['t_third_index'] = t_ind[t_type==0].astype(int)[-1] + int(dist_third)
    simulations['t_twothirds_index'] = t_ind[t_type==0].astype(int)[-1] + int(2*dist_third)
  
    if plot == True:
        simulationPlotting(simulations)
    
    return simulations

###############################################################################
############################ Internal Functions ###############################
################################# General #####################################
##################### Internal DigitalTwin functions ##########################  
def _unpackParameters(twin):
    """
    Returns list of length samples (1 for LM calibration) with a dictionary for each sample.
    """
    zeroed = {}
    found_params = []
    for elem in twin.params:
        found_params.append(elem)          
    required_params = ['d','k','alpha','beta_a','beta_c']
    for elem in set(found_params) ^ set(required_params):
        if elem == 'k' and elem != 'sigma' and twin.model != 'fwd.OP_RXDIF_wAC':
            zeroed[elem] = np.zeros(twin.tumor['Mask'].shape)
        else:
            zeroed[elem] = 0
            
    param_list = []
    for i in range(_atleast_2d(twin.params['d'].get()).shape[1]):
        temp = {}
        for elem in twin.params:
            if twin.params[elem].assignment != 'f':
                temp[elem] = _atleast_2d(twin.params[elem].get())[:,i]
            else:
                temp[elem] = twin.params[elem].get()
        for elem in zeroed:
            temp[elem] = 0
        param_list.append(temp.copy())
    return param_list

def _update_trx_params(curr_params, trx_params):
    trx_params_new = copy.deepcopy(trx_params)
    trx_params_new['beta'] = np.array([curr_params['beta_a'], 
                                       curr_params['beta_c']])
    return trx_params_new

def _maps_to_timecourse(maps, threshold, theta, h):
    cell_tc = np.zeros((maps.shape[maps.ndim-1],1))
    volume_tc = np.zeros((maps.shape[maps.ndim-1],1))
    for i in range(maps.shape[maps.ndim-1]):
        if maps.ndim == 3:
            temp = maps[:,:,i]
        else:
            temp = maps[:,:,:,i]
        cell_tc[i]   = np.sum(temp[temp>=threshold]) * theta
        volume_tc[i] = temp[temp>=threshold].size * np.prod(h)
    return cell_tc, volume_tc

@nb.njit(fastmath=True)
def _maps_to_timecourse_sampled(sampled_maps, threshold, theta, h, V):
    sh = sampled_maps.shape
    cell_tc = np.zeros((sh[sampled_maps.ndim-1],sh[0]))
    volume_tc = np.zeros((sh[sampled_maps.ndim-1],sh[0]))
    for j in range(sh[0]):
        for i in range(sh[sampled_maps.ndim-1]):
            temp = V @ sampled_maps[j,:,i]   
            cell_tc[i,j] = np.sum(temp[temp>=threshold]) * theta
            volume_tc[i,j] = temp[temp>=threshold].size * np.prod(h)
    return cell_tc, volume_tc

def _atleast_2d(arr):
    if np.atleast_1d(arr).ndim < 2:
        return np.expand_dims(np.atleast_1d(arr),axis=1)
    else:
        return arr
    
def _atleast_4d(arr):
    if arr.ndim < 4:
        return np.expand_dims(np.atleast_3d(arr),axis=3)
    else:
        return arr

def simulationPlotting(simulations):
        
    fig, ax = plt.subplots(3,2,layout = 'constrained',figsize = (8,5))
    #Cell time course plot
    _plotCI(ax[0,0], simulations['tspan'], simulations['cell_tc'], ['Time (days)', 'Cell Count','Simulation'],
            simulations['t_type'], simulations['t_meas'], simulations['cell_measured'])
    #Volume time course plot
    _plotCI(ax[0,1], simulations['tspan'], simulations['volume_tc'], ['Time (days)', 'Volume (mm^3)','Simulation'],
            simulations['t_type'], simulations['t_meas'], simulations['volume_measured'])
    #Drug A and C plots need to write but not worried about it yet
    drug_array = np.array(simulations['drug_tc'])
    _plotCI(ax[1,0], simulations['tspan'], drug_array[:,:,0].T, ['Time (days)', 'Concentration','Adriamycin'],
            simulations['t_type'], simulations['t_meas'])
    _plotCI(ax[1,1], simulations['tspan'], drug_array[:,:,1].T, ['Time (days)', 'Concentration','Cyclophosphamide'],
            simulations['t_type'], simulations['t_meas'])
    _plotCI(ax[2,0], simulations['tspan'], drug_array[:,:,2].T, ['Time (days)', 'Concentration','Paclitaxel'],
            simulations['t_type'], simulations['t_meas'])
    
def _plotCI(ax, tspan, simulation, labels, t_type, t_meas, measured = None, color = [0,0,1], arrow = True, order = 0, title = None):
    if simulation.shape[1] != 1:
        #plot confidence interval stuff
        median_sim = np.median(simulation, axis = 1)
        prctile = np.percentile(simulation, [1,99,25,75], axis = 1)
        ax.fill_between(tspan,prctile[0,:],prctile[1,:],color=color, alpha = 0.25,
                        label=labels[2] + ' - range',zorder=1 + order*4)
        ax.fill_between(tspan,prctile[2,:],prctile[3,:],color=color, alpha = 0.5,
                        label=labels[2] + ' - IQR',zorder=2 + order*4)
        line_label = labels[2] + ' - median'
    else:
        median_sim = simulation
        line_label = labels[2]
    ax.plot(tspan,  median_sim, ls = '-', color = color, linewidth = 1,label=line_label,zorder=3 + order*4)
    if measured is not None:
        ax.scatter(t_meas, measured, color = 'k', label = 'Measured',zorder=4 + order*4)
    
    if arrow == True:
        if np.any(t_type==1):
            _predictionArrow(ax, t_meas, t_type, median_sim)
        
    ax.legend(fontsize='xx-small')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if title:
        ax.set_title(title)

def _predictionArrow(ax, t_meas, t_type, median_sim):
    point = t_meas[np.where(t_type==0)[0][-1]]
    ax.axvline(point,color = 'r',linestyle = '--',linewidth = 0.5,zorder=5)
    #Get tiny triangle to signal direction of prediction
    limits = ax.get_ybound()
    ax.arrow(point+0.5,limits[0] + limits[1]*0.03, 1, 0, color = 'r',head_width = limits[1]*0.05, head_length = 0.5, length_includes_head = True,zorder=6)
    ax.arrow(point+0.5,limits[1] - limits[1]*0.03, 1, 0, color = 'r',head_width = limits[1]*0.05, head_length = 0.5, length_includes_head = True,zorder=6)
    if _findNearest(limits,np.mean(median_sim)) == 0:
        ax.text(point+2.0,limits[1] - limits[1]*0.03 - limits[1]*0.025, 'Prediction', color = 'r', fontsize = 'xx-small',zorder=7)
    else:
        ax.text(point+2.0,limits[0] + limits[1]*0.03 - limits[1]*0.025 , 'Prediction', color = 'r', fontsize = 'xx-small',zorder=7)
    
def _findNearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx