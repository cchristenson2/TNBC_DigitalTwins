# DigitalTwin.py
""" 
Defines digital twin object
Consists of:
    - Tumor data, acquired and yet to be acquired
    - Treatment controls
    - ROM (if requested)
Optional:
    - Parameters
    - Volume and cell time courses
        
Defines parameter object
Constists of:
    - parameter name; d, alpha, k, beta1, beta2
    - assignment; f, l, or g for fixed, local, or global
    - value; default = None, replaced by either fixed value or calibration

Last updated: 5/23/2024
"""
import numpy as np
import copy
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
import matplotlib.cm as cm
import matplotlib.colors as col
import scipy.ndimage as ndi
import scipy.stats as stats
import scipy.optimize as sciop
import concurrent.futures
from itertools import repeat
import time
import os

import LoadData as ld
import ReducedModel as rm
import ForwardModels as fwd
import Calibrations as cal
import Library as lib
import Optimize as opt

############################# Global variables ################################
call_dict = {
    'fwd.RXDIF_2D_wAC': fwd.RXDIF_2D_wAC,
    'fwd.RXDIF_3D_wAC': fwd.RXDIF_3D_wAC,
    'fwd.OP_RXDIF_wAC': fwd.OP_RXDIF_wAC}

###############################################################################
###############################################################################
######################### Digital twin definition #############################
class DigitalTwin:
    def __init__(self, location, load_args = {}, ROM = False, ROM_args = {}, params = None, insilico = False):
        """
        Parameters
        ----------
        location : string
            Pathway to tumor information
        load_args : dict; Default = {}
            Determines how the data should be processed.
                Keywords: downsample, inplane, inslice, crop2D, split
        ROM : boolean; Default = False
            Should a ROM be built for the data.
        ROM_args : dict; Default = {}
            Arguments for ROM building, required if ROM = True.
                Keywords: bounds (required), zipped, 
                
            augmentation = 'average', depth = 8, 
                                   samples = None
            
        """
        if insilico != True:
            self.tumor = ld.LoadTumor_mat(location, **load_args)
        else:
            self.tumor = ld.LoadInsilicoTumor_mat(location, **load_args)
        self.params = params
        if ROM:
            if ROM_args:
                if 'bounds' not in ROM_args.keys():
                    ROM_args['bounds'] = {'d': np.array([1e-6, 1e-3]),
                                          'k': np.array([1e-6, 0.1]),
                                          'alpha': np.array([1e-6, 0.8])}
                self.ROM = rm.constructROM_RXDIF(self.tumor, **ROM_args)
                self.ROM_args = ROM_args #Saved in twin object so they dont have to be repassed when data is assimilated
            else:
                raise ValueError('if ROM = True, ROM_args must be passed in')
     
    def setParams(self, params):
        self.params = params
        
    def getPriors(self, params):
        self.priors = cal.generatePriors(params)
        
########################## Calibration functions ##############################
    def assimilate(self, cal_type = None, cal_args = None, ROM_args = None):
        """
        Adds next data point from future data onto current tumor.
        Rebuilds the ROM according to the last requirements unless new ROM_args 
            is passed in.
        Calibrates according to the same problem previously used unless new
            cal_args is passed in.
        """
        #Add next scan onto acquired data in Tumor['N'] and tumor['t_scan']
        #Delete Future N and Future t_scan if last data point removed
        if 'Future t_scan' in self.tumor:
            t_save = self.tumor['Future t_scan'][0]
            self.tumor['Future t_scan'] = np.delete(self.tumor['Future t_scan'],0)
            self.tumor['t_scan'] = np.concatenate((self.tumor['t_scan'], t_save))
            if 0 in self.tumor['Future t_scan'].shape:
                self.tumor.pop('Future t_scan',None)
                
            if self.tumor['Mask'].ndim == 2:
                N_save = np.atleast_3d(self.tumor['Future N'])[:,:,0]
            else:
                N_save = np.atleast_4d(self.tumor['Future N'])[:,:,:,0]
            self.tumor['Future N'] = np.delete(self.tumor['Future N'],0,
                                               axis=self.tumor['Mask'].ndim)
            self.tumor['N'] = np.concatenate((self.tumor['N'], N_save),
                                             axis=self.tumor['Mask'].ndim)
            if 0 in self.tumor['Future N'].shape:
                self.tumor.pop('Future N',None)
            
            if hasattr(self, 'ROM'):
                N_r_save = _atleast_2d(self.ROM['ReducedTumor']['Future N_r'])[:,0]
                self.ROM['ReducedTumor']['Future N_r'] = \
                    np.delete(self.ROM['ReducedTumor']['Future N_r'],0,axis=1)
                self.ROM['ReducedTumor']['N_r'] = \
                    np.concatenate((self.ROM['ReducedTumor']['N_r'], N_r_save),
                                   axis=1)
                if 0 in self.ROM['ReducedTumor']['Future N_r'].shape:
                    self.ROM['ReducedTumor'].pop('Future N_r', None)
                #Must rebuild ROM after assimilating data
                if ROM_args == None:
                    ROM_args = self.ROM_args
                self.ROM = rm.constructROM_RXDIF(self.tumor, **ROM_args)
                self.ROM_args = ROM_args
                    
        #Calibrate with updated tumor
        if cal_type == None & hasattr(self,cal_type):
            cal_type = self.cal_type
        else:
            raise ValueError('Calibration has not ran prior to assimilation,',
                             ' must provide calibration type and arguments') 
        if cal_args == None:
            cal_args = self.cal_args
        self.calibrateTwin(cal_type, cal_args)
            
    def calibrateTwin(self, cal_type, cal_args = {}):
        """
        Updates parameters of twin based on current tumor maps. Stores stats in
        new twin attribute, dependent on type of calibration
        
        Parameters
        ----------
        cal_type : string
            Which calibration type to call. ROM based methods throw error if 
            ROM isn't in twin object.
        cal_args : dict; default = {}
            LM_FOM or LM_ROM options: dt, parallel (FOM only), plot, 
                                    options: {'e_tol','e_conv','max_it','delta',
                                              'pass','fail','lambda','j_freq'}
            gwMCMC_ROM options: dt,  parallel, plot, 
                                options: {'thin','progress','samples',
                                          'burnin','step_size','walker_factor'}
        """
        valid_cal = ['LM_FOM', 'LM_ROM', 'gwMCMC_ROM', 'ABC_ROM']
        if cal_type in valid_cal:
            if cal_type == 'LM_FOM':
                self.params, self.cal_stats, self.model , _= \
                    cal.calibrateRXDIF_LM(self.tumor, self.params, **cal_args)
            else:
                if hasattr(self, 'ROM'):
                    if cal_type == 'LM_ROM':
                        self.params, self.cal_stats, self.model , _ = \
                            cal.calibrateRXDIF_LM_ROM(self.tumor, self.ROM, 
                                                      self.params, **cal_args)
                    else:
                        if hasattr(self, 'priors'):
                            if cal_type == 'gwMCMC_ROM':
                                self.params, self.ensemble_sampler, self.model = \
                                    cal.calibrateRXDIF_gwMCMC_ROM(self.tumor, 
                                                                  self.ROM, 
                                                                  self.params, 
                                                                  self.priors, 
                                                                  **cal_args)
                            elif cal_type == 'ABC_ROM':
                                self.params, self.model = \
                                    cal.calibrateRXDIF_ABC_ROM(self.tumor, 
                                                                  self.ROM, 
                                                                  self.params, 
                                                                  self.priors, 
                                                                  **cal_args)
                            
                        else:
                            raise ValueError('Priors must be contained in twin',
                                             ' object for bayesian calibrations')                   
                else:
                    raise ValueError('ROM must be contained in twin object to',
                                     ' use ROM based calibrations')
            self.cal_type = cal_type
            self.cal_args = cal_args
        else:
            raise ValueError('Calibration type must be: "LM_FOM", "LM_ROM",',
                             ' "gwMCMC_ROM"')      
 
########################## Prediction and stats ###############################  
    def predict(self, dt = 0.5, threshold = 0.20, plot = False,
                visualize = False, parallel = False, treatment = None,
                partial = False, estimated = False, change_t = None,
                MCMC_visualization = False):
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
        for elem in self.params:
            if self.params[elem].value is None:
                raise ValueError('Calibrated parameters do not have values '
                                 'assigned. Run calibrateTwin() first')
            
        t = self.tumor['t_scan'] #Time variables for simulation outputs
        t_type_full = np.zeros(t.shape)   #Type of simulation for each time, 0 = calibrated, 1 = predicted
        if 'Future t_scan' in self.tumor:
            t = np.append(t, self.tumor['Future t_scan'])
            t_type_full = np.append(t_type_full, np.ones(self.tumor['Future t_scan'].shape))
            pred_on = True
        else:
            pred_on = False
        
        if change_t is not None:
            deleted = np.where(t>=change_t)
            t_type_full = np.delete(t_type_full, deleted)
            t_true = t.copy()
            t = np.delete(t, deleted)
            t_type_full = np.append(t_type_full, np.array([1]))
            t = np.append(t, change_t)
        else:
            t_true = t.copy()
            
        t_ind = t[1:]/dt
        t_type = t_type_full[1:]
        N_meas = self.tumor['N'].copy()
        if self.tumor['Mask'].ndim==2:
            N0 = self.tumor['N'][:,:,0]
            N_cal = self.tumor['N'][:,:,1:]
            N_cal = N_cal.reshape((-1,np.atleast_3d(N_cal).shape[2]))
            if 'Future N' in self.tumor:
                N_pred = self.tumor['Future N'].copy()
                N_pred = N_pred.reshape((-1,np.atleast_3d(N_pred).shape[2]))
                N_meas = np.concatenate((N_meas, self.tumor['Future N']),axis = 2)
            else:
                N_pred = []
        else:
            N0 = self.tumor['N'][:,:,:,0]
            N_cal = self.tumor['N'][:,:,:,1:]
            N_cal = N_cal.reshape((-1,_atleast_4d(N_cal).shape[3]))
            if 'Future N' in self.tumor:
                N_pred = self.tumor['Future N'].copy()
                N_pred = N_pred.reshape((-1,_atleast_4d(N_pred).shape[3]))
                N_meas = np.concatenate((N_meas, self.tumor['Future N']),axis = 3)
            else:
                N_pred = []
        N_cal_r = self.ROM['ReducedTumor']['N_r'][:,1:]
        if pred_on == True:
            N_pred_r = self.ROM['ReducedTumor']['Future N_r']
        else:
            N_pred_r = []
        tspan = np.arange(0,t[-1]+dt,dt)
        
        if treatment == None:
            default_trx_params = {'t_trx': self.tumor['t_trx']}
        else:
            default_trx_params = {}
            for elem in treatment:
                default_trx_params[elem] = treatment[elem]
            
        parameters = self._unpackParameters()
        calibrations = []
        predictions = []
        calibrations_r = []
        predictions_r = []
        cell_tc = np.array([]).reshape(tspan.size,0)
        volume_tc = np.array([]).reshape(tspan.size,0)
        drug_tc = []
        samples = len(parameters)
        if samples == 1:
            i = 0
            curr = parameters[i]
            trx_params = default_trx_params.copy()
            trx_params = _update_trx_params(curr, default_trx_params)
            #Run forward model
            if self.model != 'fwd.OP_RXDIF_wAC':
                sim, drugs = call_dict[self.model](N0, curr['k'], curr['d'], 
                                                   curr['alpha'], trx_params, 
                                                   tspan, self.tumor['h'], dt, 
                                                   self.tumor['bcs'])
                if self.tumor['Mask'].ndim==2:
                    calibrations.append(sim[:,:,t_ind[t_type==0].astype(int)].reshape(-1,t_ind[t_type==0].size))
                    if pred_on == True:
                        predictions.append(sim[:,:,t_ind[t_type==1].astype(int)].reshape(-1,t_ind[t_type==1].size))
                else:
                    calibrations.append(sim[:,:,:,t_ind[t_type==0].astype(int)].reshape(-1,t_ind[t_type==0].size))
                    if pred_on == True:
                        predictions.append(sim[:,:,:,t_ind[t_type==1].astype(int)].reshape(-1,t_ind[t_type==1].size))
                    temp1, temp2 = _maps_to_timecourse(sim, threshold, 
                                                       self.tumor['theta'], 
                                                       self.tumor['h'])
            else:
                N0 = self.ROM['ReducedTumor']['N_r'][:,0]
                operators = lib.getOperators(curr,self.ROM)
                sim, drugs = call_dict[self.model](N0, operators, trx_params, tspan, dt)
                calibrations.append(self.ROM['V'] @ sim[:,t_ind[t_type==0].astype(int)])
                calibrations_r.append(sim[:,t_ind[t_type==0].astype(int)])
                if pred_on == True:
                    predictions.append(self.ROM['V'] @ sim[:,t_ind[t_type==1].astype(int)])
                    predictions_r.append(sim[:,t_ind[t_type==0].astype(int)])
                temp1, temp2 = _maps_to_timecourse_R(sim, threshold, 
                                                   self.tumor['theta'], 
                                                   self.tumor['h'], self.ROM['V']) 
            cell_tc = np.concatenate((cell_tc, temp1), axis = 1)
            volume_tc = np.concatenate((volume_tc, temp2), axis = 1)
            drug_tc.append(drugs)
        else:
            #More than 1 sample, we know the problem was ROM with MCMC
            sim_save = np.zeros((samples, self.ROM['V'].shape[1], tspan.size))
            for i in range(samples):
                curr = parameters[i]
                trx_params = default_trx_params.copy()
                trx_params = _update_trx_params(curr, default_trx_params)
                N0 = self.ROM['ReducedTumor']['N_r'][:,0]
                operators = lib.getOperators(curr,self.ROM)
                sim, drugs = call_dict[self.model](N0, operators, trx_params, tspan, dt)
                calibrations.append(self.ROM['V'] @ sim[:,t_ind[t_type==0].astype(int)])
                calibrations_r.append(sim[:,t_ind[t_type==0].astype(int)])
                if pred_on == True:
                    predictions.append(self.ROM['V'] @ sim[:,t_ind[t_type==1].astype(int)])
                    predictions_r.append(sim[:,t_ind[t_type==0].astype(int)])
                sim_save[i,:,:] = sim
                drug_tc.append(drugs)
            if partial == True:
                index = t_ind[t_type==0].astype(int)[-1]
                data = sim_save[:,:,index:]
            else:
                data = sim_save
            if estimated == True:
                cell_tc, volume_tc = _maps_to_timecourse_sampled_est(data, threshold, self.tumor['theta'],self.tumor['h'],self.ROM['V'])
            else:
                cell_tc, volume_tc = _maps_to_timecourse_sampled(data, threshold, self.tumor['theta'],self.tumor['h'],self.ROM['V'])
                    
        cell_measured, volume_measured = _maps_to_timecourse(N_meas, threshold, 
                                           self.tumor['theta'], self.tumor['h'])
        if samples > 1:
            calibrations = np.mean(np.squeeze(np.array(calibrations)), axis = 0)
            predictions = np.mean(np.squeeze(np.array(predictions)), axis = 0)
        else:
            calibrations = calibrations[0]
            predictions = predictions[0]
            
        if treatment is not None:
            doses = treatment['doses']
        else:
            doses = np.ones(default_trx_params['t_trx'].size)
            
        simulations = {'cell_tc': cell_tc, 'volume_tc': volume_tc,
                            'maps_cal': calibrations, 'maps_pred': predictions,
                            'maps_r_cal': calibrations_r, 'maps_r_pred': predictions_r,
                            'cell_measured': cell_measured, 
                            'volume_measured':volume_measured,'prediction':pred_on,
                            'N_cal':N_cal, 'N_pred':N_pred, 
                            'N_r_cal':N_cal_r, 'N_r_pred':N_pred_r,
                            'samples':samples,'tspan':tspan, 
                            't_meas':t_true, 't_ind':t_ind, 't_type':t_type_full,
                            'drug_tc':drug_tc, 'trx_params':default_trx_params,
                            'doses':doses}
        
        temp = t[-1]/dt
        dist = int((temp - t_ind[t_type==0].astype(int)[-1])/2)
        dist_third = round((temp - t_ind[t_type==0].astype(int)[-1])/3)
        
        ttp_index = int(self.tumor['t_trx'][2]/dt - t_ind[t_type==0].astype(int)[-1])
        
        V3_index = int(t_true[-1]/dt - t_ind[t_type==0].astype(int)[-1])

        
        if partial == True:
            simulations['t_pred_index'] = int(0)
            simulations['t_mid_index'] = dist
            simulations['t_third_index'] = int(dist_third)
            simulations['t_twothirds_index'] = int(2*dist_third)
            simulations['ttp_index'] = int(ttp_index)
            simulations['t_V3_index'] = int(V3_index)

        else:
            simulations['t_pred_index'] = int(t_ind[t_type==0].astype(int)[-1])
            simulations['t_mid_index'] = int(t_ind[t_type==0].astype(int)[-1] + dist)
            simulations['t_third_index'] = int(t_ind[t_type==0].astype(int)[-1] + int(dist_third))
            simulations['t_twothirds_index'] = int(t_ind[t_type==0].astype(int)[-1] + int(2*dist_third))
            simulations['ttp_index'] = int(t_ind[t_type==0].astype(int)[-1] + int(ttp_index))
            simulations['t_V3_index'] = int(t_ind[t_type==0].astype(int)[-1] + int(V3_index))
            

        #Plot outputs if requested - mean sample visualized if MCMC was used
        if visualize == True:
            self.simulationVisualization(simulations, MCMC_vis = MCMC_visualization)
                    
        if plot == True:
            self.simulationPlotting(simulations)
        
        return simulations
            
    def simulationStats(self, threshold = 0.0):
        """
        Calculates the CCC, Dice, TTV, and TTC at each simulated time point
        for both calibrations and predictions.
        
        Returns in separate dictionaries; calibrations_stats, prediction_stats
        """
        C0 = self.simulations['cell_measured'][0]
        V0 = self.simulations['volume_measured'][0]
        calibration_stats = {'CCC':[],'Dice':[],'delTTC':[],'delTTV':[]}
        if self.simulations['prediction']==True:
            prediction_stats = {'CCC':[],'Dice':[],'delTTC':[],'delTTV':[]}
        for i in range(len(self.simulations['maps_cal'])):
            curr_cal = self.simulations['maps_cal'][i]
            n_cal = _atleast_2d(curr_cal).shape[1]
            CCC = []
            Dice = []
            TTC = []
            TTV = []
            for j in range(n_cal):
                out = getStats_SingleTime(_atleast_2d(curr_cal)[:,j], 
                                          _atleast_2d(self.simulations['N_cal'])[:,j],
                                          threshold, self.tumor['theta'],
                                          self.tumor['h'])
                CCC.append(out[0]), Dice.append(out[1])
                TTC.append(out[2]), TTV.append(out[3])
            calibration_stats['CCC'].append(CCC)
            calibration_stats['Dice'].append(Dice)
            calibration_stats['delTTC'].append(TTC - C0)
            calibration_stats['delTTV'].append(TTV - V0)
            if self.simulations['prediction']==True:
                curr_pred = self.simulations['maps_pred'][i]
                n_pred = _atleast_2d(curr_pred).shape[1]
                CCC = []
                Dice = []
                TTC = []
                TTV = []
                for j in range(n_pred):
                    out = getStats_SingleTime(_atleast_2d(curr_pred)[:,j], 
                                              _atleast_2d(self.simulations['N_pred'])[:,j],
                                              threshold, self.tumor['theta'],
                                              self.tumor['h'])
                    CCC.append(out[0]), Dice.append(out[1])
                    TTC.append(out[2]), TTV.append(out[3])
                prediction_stats['CCC'].append(CCC)
                prediction_stats['Dice'].append(Dice)
                prediction_stats['delTTC'].append(TTC - C0)
                prediction_stats['delTTV'].append(TTV - V0)
        if self.simulations['prediction']==True:
            self.stats = {'calibration': calibration_stats,'prediction': prediction_stats}
        else:
            self.stats = {'calibration': calibration_stats,'prediction': []}
            
########################### Plotting parameters ###############################
#Only set up for MCMC based parameters right now
    def paramVisualization(self):
        parameters = self._unpackParameters()
        samples = len(parameters)
        if samples > 1:
            num = 0
            for elem in self.params:
                if self.params[elem].assignment.lower() == 'g':
                    num += 1
                elif self.params[elem].assignment.lower() == 'r':
                    num += self.ROM['V'].shape[1]
            fig, ax = plt.subplots(1, num, layout = "constrained", figsize = (num*1.25,1.5))
            offset = 0
            for i, elem in enumerate(self.params):
                if self.params[elem].assignment.lower() == 'g':
                    ax[i+offset].hist(self.params[elem].get().reshape(-1), bins = 20, density = True, label = 'Samples', color = [0,0,1])
                    if elem == 'alpha':
                        ax[i+offset].set_title(r'$\alpha$')
                    elif elem == 'beta_a':
                        ax[i+offset].set_title(r'$\beta_A$')
                    elif elem == 'beta_c':
                        ax[i+offset].set_title(r'$\beta_C$')
                    else:
                        ax[i+offset].set_title(elem, style='italic')
                    try:
                        x = np.linspace(self.params[elem].getBounds()[0], self.params[elem].getBounds()[1], 100)
                        y = self.priors[elem].pdf(x)
                        ax[i+offset].plot(x,y,'r--',label='Prior')
                    except:
                        pass
                elif self.params[elem].assignment.lower() == 'r':
                    for j in range(self.ROM['V'].shape[1]):
                        ax[i+offset].hist(self.params[elem].get()[j,:], bins = 20, density = True, label = 'Samples', color = [0,0,1])
                        string = '$\\it{' + elem + '}_{r,' + str(j) +'}$'
                        ax[i+offset].set_title(r"{}".format(string), style='italic')
                        try:
                            x = np.linspace(self.params[elem].getCoeffBounds()[j,0], self.params[elem].getCoeffBounds()[j,1], 100)
                            y = self.priors[elem + str(j)].pdf(x)
                            ax[i+offset].plot(x,y,'r--',label='Prior')
                        except:
                            pass
                        if j < self.ROM['V'].shape[1] - 1:
                            offset += 1
                if i == 0:
                    ax[i+offset].set_ylabel('Counts')
                    ax[i+offset].legend(fontsize='xx-small')

    def simulationVisualization(self, simulations = None, MCMC_vis = False):
        if simulations == None:
            simulations = self.simulations
        if simulations['samples'] > 1 and MCMC_vis == True:
            #Since we only have samples with ROM we can visualize the reduced coefficients here.
            _visualize_MCMC_error(simulations['N_r_cal'], simulations['N_r_pred'],
                                  np.array(simulations['maps_r_cal']), np.array(simulations['maps_r_pred']),
                                  simulations['prediction'], self.ROM, self.params)
        # Visualize volume and cell time courses compared to measured data
        if self.tumor['Mask'].ndim == 2:
            _visualize_2d(simulations['N_cal'], simulations['N_pred'],
                          simulations['maps_cal'], simulations['maps_pred'],
                          simulations['prediction'], self.tumor)
        else:
            #Will make changes to 3D visualization, not a fan of current setup
            print()
            # _visualize_3d(simulations['N_cal'], simulations['N_pred'],
            #               simulations['maps_cal'], simulations['maps_pred'],
            #               simulations['prediction'], self.tumor, cut = 'axial')
            # _visualize_3d(simulations['N_cal'], simulations['N_pred'],
            #               simulations['maps_cal'], simulations['maps_pred'],
            #               simulations['prediction'], self.tumor, cut = 'sagittal')
            
    def simulationPlotting(self, simulations = None):
        if simulations == None:
            simulations = self.simulations
            
        fig, ax = plt.subplots(2,2,layout = 'constrained',figsize = (8,5))
        #Cell time course plot
        _plotCI(ax[0,0], simulations['tspan'], simulations['cell_tc'], ['Time (days)', 'Cell Count','Sim.'],
                simulations['t_type'], simulations['t_meas'], simulations['cell_measured'])
        #Volume time course plot
        _plotCI(ax[0,1], simulations['tspan'], simulations['volume_tc'], ['Time (days)', 'Volume (mm^3)','Sim.'],
                simulations['t_type'], simulations['t_meas'], simulations['volume_measured'])
        #Drug A and C plots need to write but not worried about it yet
        drug_array = np.array(simulations['drug_tc'])
        _plotCI(ax[1,0], simulations['tspan'], drug_array[:,:,0].T, ['Time (days)', 'A. concentration','Sim.'],
                simulations['t_type'], simulations['t_meas'])
        _plotCI(ax[1,1], simulations['tspan'], drug_array[:,:,1].T, ['Time (days)', 'C. concentration','Sim.'],
                simulations['t_type'], simulations['t_meas'])

########################## Optimization functions #############################
    def optimize_cellMin(self, problem, method, initial_guess = 1, test_guess = False):
        if initial_guess == 1:
            problem['doses_guess'] = opt.socInitialGuess(self,problem)
            # print('Using standard of care as guess')
        elif initial_guess == 2:
            problem['doses_guess'] = opt.randomizeInitialGuess(self,problem)
            # print('Initial guess found')
        elif initial_guess == 3:
            print()
            # print('Start point is previous result')
        
        if test_guess == True:
            x = opt.reorganize_doses(problem['doses_guess'], problem)
            test_trx = {'t_trx': np.concatenate((problem['t_trx_soc'], problem['potential_days']),axis = 0),
                             'doses': np.concatenate((problem['doses_soc'], x), axis = 0)}
            test_simulations = self.predict(treatment = test_trx,
                                            threshold = problem['threshold'],
                                            plot = False,
                                            change_t = problem['t_pred_end'])
        else:
            test_simulations = []
            
        Model = opt.CachedModel(self, problem)
        
        if method != 3:
            cons = []
            if problem['lin-constraints']:   
                cons.append(problem['lin-constraints'][0])
            if problem['nonlin-constraints']:
                nonlin_con = {'type':'ineq','fun': Model.constraints}
                cons.append(nonlin_con)
        else:
            cons = []
            if problem['lin-constraints']:   
                cons.append(problem['lin-constraints'][0])
                cons.append(sciop.NonlinearConstraint(problem['lin-constraints'][0]['fun'], 0, np.inf))
            if problem['nonlin-constraints']:
                cons.append(sciop.NonlinearConstraint(Model.constraints, 0, np.inf))
                
                
                
            
        if method == 1:
            output = sciop.minimize(Model.objective, problem['doses_guess'],
                                    bounds = [(0,1)]*problem['doses_guess'].size,
                                    constraints = cons,
                                    method = 'COBYLA', options = {'maxiter':500,'catol':1e-10})
        elif method == 2:  
            output = sciop.basinhopping(Model.objective, problem['doses_guess'],
                                        niter=20, disp = False, interval = 2,
                                        minimizer_kwargs = {'bounds':[(0,1)]*problem['doses_guess'].size,
                                                            'constraints':cons,
                                                            'method':'COBYLA','options':{'maxiter':500,'catol':1e-10}})    
        elif method == 3:
            output = sciop.minimize(Model.objective, problem['doses_guess'],
                                    bounds = [(0,1)]*problem['doses_guess'].size,
                                    constraints = cons,
                                    method = 'COBYQA', options = {'maxiter':500})
            
        else:
            output = sciop.minimize(Model.objective, problem['doses_guess'],
                                    bounds = [(0,1)]*problem['doses_guess'].size,
                                    constraints = cons
                                    ,options={'maxiter':1000})
        
        # print(output)
        temp = opt.reorganize_doses(output.x, problem)
        new_days = np.concatenate((problem['t_trx_soc'], problem['potential_days']),axis = 0)
        new_doses = np.concatenate((problem['doses_soc'], temp), axis = 0)
        optimized_trx = {'t_trx': new_days, 'doses': new_doses}
        optimal_simulations = self.predict(treatment = optimized_trx, threshold = problem['threshold'], change_t = problem['t_pred_end'])
        return optimal_simulations, optimized_trx, new_doses, test_simulations, output
    
    def optimize_doseMin(self, problem, problem_args, tol = 0.01, method = 2):
        run = 1
        dose_change = 0.5
        max_dose = problem['max_dose']
        cache = {}
        best_guess = None
        soc = np.mean(self.simulations['cell_tc'][-1,:])
        print(soc)
        while run == 1:
            print('Current dose = '+str(max_dose))
            if str(max_dose) not in cache:
                print('not in cache')
                if best_guess is None:
                    problem['doses_guess'] = opt.randomizeInitialGuess(self,problem)
                else:
                    problem['doses_guess'] = best_guess * max_dose / sum(best_guess)
                curr_sim, curr_trx, curr_dose = self.optimize_cellMin(problem,method = method)
                obj = np.mean(curr_sim['cell_tc'][-1,:])
                cache[str(max_dose)] = {'obj':obj,'sim':curr_sim,'trx':curr_trx,'doses':curr_dose}
            else:
                print('in cache')
                obj = cache[str(max_dose)]['obj'] 
            print(obj)
            
            #Check if cells are better or worse or within tolerance
            if abs(obj - soc)/soc < tol:
                #Save results tolerance reached
                optimal_dose = max_dose
                run = 0
            elif obj < soc:
                #Update problem with lower max dose
                max_dose -= dose_change
                problem_args['max_dose'] = max_dose
                problem = opt.problemSetup_cellMin(self, **problem_args)
                best_guess = curr_dose
            elif obj > soc:
                #update problem with higher max dose
                dose_change /= 2
                max_dose += dose_change
                problem_args['max_dose'] = max_dose
                problem = opt.problemSetup_cellMin(self.tumor, self.simulations, **problem_args)
            
        return cache[str(optimal_dose)]['sim'], cache[str(optimal_dose)]['trx'] , cache[str(optimal_dose)]['doses']
      
##################### Internal DigitalTwin functions ##########################  
    def _unpackParameters(self):
        """
        Returns list of length samples (1 for LM calibration) with a dictionary for each sample.
        """
        zeroed = {}
        found_params = []
        for elem in self.params:
            found_params.append(elem)          
        required_params = ['d','k','alpha','beta_a','beta_c']
        for elem in set(found_params) ^ set(required_params):
            if elem == 'k' and elem != 'sigma' and self.model != 'fwd.OP_RXDIF_wAC':
                zeroed[elem] = np.zeros(self.tumor['Mask'].shape)
            else:
                zeroed[elem] = 0
                
        param_list = []
        for i in range(_atleast_2d(self.params['d'].get()).shape[1]):
            temp = {}
            for elem in self.params:
                if self.params[elem].assignment != 'f':
                    temp[elem] = _atleast_2d(self.params[elem].get())[:,i]
                else:
                    temp[elem] = self.params[elem].get()
            for elem in zeroed:
                temp[elem] = 0
            param_list.append(temp.copy())
        return param_list
            
###############################################################################
######################### Parameter class object ##############################
class Parameter:
    def __init__(self, name, assignment, value = None):
        valid_names = ['d','k','alpha','beta_a','beta_c','sigma']
        valid_assignment = ['f','l','g']
        default_values = [5e-4, 0.05, 0.4, 0.60, 3.25, 0.0]
        
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
        elif self.assignment == 'g':
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
        
    def setCoeffBounds(self, bounds):
        self.coeff_bounds = bounds
    
    def update(self, value):
        self.value = value
        
    def get(self):
        return self.value
    
    def getCoeffBounds(self):
        return self.coeff_bounds  
    
    def getBounds(self):
        return self.bounds 
    
    def reconstruct(self):
        return self.basis @ self.value
    
    def delete(self):
        self.value = None        

###############################################################################
######################### Parameter class object ##############################
def plotCI_comparison(simulations, simulations2, color1 = [0,0,1], color2 = [0.92,0.56,0.02]):
    fig, ax = plt.subplots(2,2,layout = 'constrained',figsize = (8,5))
    #Cell time course plot
    _plotCI(ax[0,0], simulations['tspan'], simulations['cell_tc'], ['Time (days)', 'Cell Count','Sim.'],
            simulations['t_type'], simulations['t_meas'], simulations['cell_measured'], color = color1, title='Cell time course')
    #Volume time course plot
    _plotCI(ax[0,1], simulations['tspan'], simulations['volume_tc'], ['Time (days)', 'Volume (mm^3)','Sim.'],
            simulations['t_type'], simulations['t_meas'], simulations['volume_measured'], color = color1, title='Volume time course')
    #Drug A and C plots need to write but not worried about it yet
    drug_array = np.array(simulations['drug_tc'])
    _plotCI(ax[1,0], simulations['tspan'], drug_array[:,:,0].T, ['Time (days)', 'A. concentration','A.'],
            simulations['t_type'], simulations['t_meas'], color = color1, title='Ardiamycin time course')
    _plotCI(ax[1,1], simulations['tspan'], drug_array[:,:,1].T, ['Time (days)', 'C. concentration','C.'],
            simulations['t_type'], simulations['t_meas'], color = color1, title='Cyclophosphamide time course')
    
    _plotCI(ax[0,0], simulations2['tspan'], simulations2['cell_tc'], ['Time (days)', 'Cell Count','Sim. - opt.'],
            simulations['t_type'], simulations['t_meas'], color = color2, arrow = False, order = 1)
    #Volume time course plot
    _plotCI(ax[0,1], simulations2['tspan'], simulations2['volume_tc'], ['Time (days)', 'Volume (mm^3)','Sim. - opt.'],
            simulations['t_type'], simulations['t_meas'], color = color2, arrow = False, order = 1)
    #Drug A and C plots need to write but not worried about it yet
    drug_array = np.array(simulations2['drug_tc'])
    _plotCI(ax[1,0], simulations2['tspan'], drug_array[:,:,0].T, ['Time (days)', 'A. concentration','A. - opt.'],
            simulations2['t_type'], simulations2['t_meas'], color = color2, arrow = False, order = 1)
    _plotCI(ax[1,1], simulations2['tspan'], drug_array[:,:,1].T, ['Time (days)', 'C. concentration','C. - opt.'],
            simulations2['t_type'], simulations2['t_meas'], color = color2, arrow = False, order = 1)
    
###############################################################################
############################ Internal Functions ###############################
################################# General #####################################
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
    
def CCC_overlap(a, b, threshold):
    ind = np.nonzero((a>=threshold)&(b>=threshold))
    a = a[ind]
    b = b[ind]
    mu_a = np.mean(a)
    mu_b = np.mean(b)
    cov = 0
    vara = 0
    varb = 0
    n = a.size
    for i in range(a.size):
        cov += (a[i] - mu_a)*(b[i] - mu_b)
        vara += (a[i] - mu_a)**2
        varb += (b[i] - mu_b)**2
    try:
        return 2*(cov/n)/((vara/n)+(varb/n)+(mu_a - mu_b)**2)
    except:
        return 0

def Dice_calc(a, b, threshold):
    num_a = np.nonzero(a>=threshold)[0].size
    num_b = np.nonzero(b>=threshold)[0].size
    num_ab = np.nonzero((a>=threshold)&(b>=threshold))[0].size
    try:
        return 2*num_ab/(num_a + num_b)
    except:
        return 0

def getStats_SingleTime(sim, meas, threshold, theta, h, ROM = False):
    CCC = CCC_overlap(sim, meas, threshold)
    Dice = Dice_calc(sim, meas, threshold)
    TTC = np.sum(sim[sim>=threshold]) * theta
    TTV = sim[sim>=threshold].size * np.prod(h)
    
    return CCC, Dice, TTC, TTV

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

import numba as nb
@nb.njit(fastmath=True,cache=True)
def _maps_to_timecourse_R(maps, threshold, theta, h, V):
    cell_tc = np.zeros((maps.shape[maps.ndim-1],1))
    volume_tc = np.zeros((maps.shape[maps.ndim-1],1))
    for i in range(maps.shape[maps.ndim-1]):
        temp = V @ maps[:,i]
        cell_tc[i]   = np.sum(temp[temp>=threshold]) * theta
        volume_tc[i] = temp[temp>=threshold].size * np.prod(h)
    return cell_tc, volume_tc

@nb.njit(fastmath=True,cache=True)
def _maps_to_timecourse_sampled(sampled_maps, threshold, theta, h, V):
    sh = sampled_maps.shape
    cell_tc = np.zeros((sh[sampled_maps.ndim-1],sh[0]))
    volume_tc = np.zeros((sh[sampled_maps.ndim-1],sh[0]))
    for j in range(sh[0]):
        for i in range(sh[sampled_maps.ndim-1]):
            m = V @ sampled_maps[j,:,i]   
            n = m.size
            mask = np.empty(n, dtype = np.bool_)
            for k in nb.prange(n):
                mask[k] = m[k] >= threshold
            cell_tc[i,j] = np.sum(m[mask]) * theta
            volume_tc[i,j] = m[mask].size * np.prod(h)      
    return cell_tc, volume_tc

@nb.njit(fastmath=True,cache=True)
def _maps_to_timecourse_sampled_est(sampled_maps, threshold, theta, h, V):
    #Only works with cell time course, returns empty list for volumes so we dont have to adjust entire problem
    #No plotting happens when using estimated results
    sh = sampled_maps.shape
    cell_tc = np.zeros((sh[sampled_maps.ndim-1],sh[0]))
    V_sum = np.sum(V,axis=0)
    for j in range(sh[0]):
        for i in range(sh[sampled_maps.ndim-1]):
            cell_tc[i,j] = np.dot(V_sum, sampled_maps[j,:,i])*theta
    return cell_tc, np.zeros(1)


############################# General plotting ################################

def _alphaToHex(n):
    string = '{0:x}'.format(round(n*255))
    if len(string) == 1:
        string = '0'+string
    return string

def _centerMass(mat):
    mat = mat / np.sum(mat)
    com = []
    shape = mat.shape
    for i in range(len(shape)):
        d = np.sum(mat,axis=i)
        c = np.sum(d * np.arange(shape[i]))
        com.append(c)
    return com

############################ MCMC Visualization ###############################
def _visualize_MCMC_error(N_cal_r, N_pred_r, calibrations_r, predictions_r, pred_on, ROM, params):
    r = ROM['V'].shape[1]
    rows = N_cal_r.shape[N_cal_r.ndim-1]
    if pred_on == True:
        rows += N_pred_r.shape[N_pred_r.ndim-1]
    fig, ax = plt.subplots(rows, r, figsize = (r*2.5, rows*3.0), layout = "constrained")
    for i in range(_atleast_2d(N_cal_r).shape[-1]):
        for j in range(r):
            curr = np.squeeze(calibrations_r[:,j,i]) - _atleast_2d(N_cal_r)[j,i]
            _atleast_2d(ax)[i,j].hist(curr, bins = 20, density = True, label = 'Samples', color = 'blue')
            _atleast_2d(ax)[i,j].set_xlabel('Coefficient error')
            _atleast_2d(ax)[i,j].set_aspect('auto')
            if j == 0:
                _atleast_2d(ax)[i,j].set_ylabel('Visit '+str(i+2)+' counts')
            if i == 0:
                _atleast_2d(ax)[i,j].set_title('Mode '+str(j+1))
            try:
                #Get likelihood distribution if sigma is in params
                s = params['sigma'].get()
                _atleast_2d(ax)[i,j].plot(np.linspace(-3*s, 3*s, 1000),
                                          stats.norm.pdf(np.linspace(-3*s, 3*s, 1000), 0, s),
                                          'r--',label='Likelihood')
            except:
                pass

    if pred_on == True:
        spacer = _atleast_2d(N_cal_r).shape[-1]
        for i in range(_atleast_2d(N_pred_r).shape[-1]):
            for j in range(r):
                curr = np.squeeze(predictions_r[:,j,i]) - _atleast_2d(N_pred_r)[j,i]
                _atleast_2d(ax)[i+spacer,j].hist(curr, bins = 20, density = True, label = 'Samples', color = 'blue')
                _atleast_2d(ax)[i+spacer,j].set_xlabel('Coefficient error')
                _atleast_2d(ax)[i+spacer,j].set_aspect('auto')
                if j == 0:
                    _atleast_2d(ax)[i+spacer,j].set_ylabel('Visit '+str(i+3)+' counts')
                try:
                    #Get likelihood distribution if sigma is in params
                    s = params['sigma'].get()
                    _atleast_2d(ax)[i+spacer,j].plot(np.linspace(-3*s, 3*s, 1000),
                                                     stats.norm.pdf(np.linspace(-3*s, 3*s, 1000), 0, s),
                                                     'r--',label='Likelihood')
                except:
                    pass
    
############################# 2D Visualization ################################            
def _visualize_2d(N_cal, N_pred, calibration, prediction, pred_on, tumor):
    rows = N_cal.shape[N_cal.ndim-1]
    if pred_on == True:
        rows += N_pred.shape[N_pred.ndim-1]
    fig, ax = plt.subplots(rows, 3, layout = "constrained")
    for i in range(_atleast_2d(N_cal).shape[-1]):
        temp_N = _atleast_2d(N_cal)[:,i].reshape(tumor['Mask'].shape)
        temp_sim = _atleast_2d(calibration)[:,i].reshape(tumor['Mask'].shape)
        p = ax[i,0].imshow(temp_N, clim=(0,1),cmap='jet')
        ax[i,0].set_ylabel('Visit '+str(i+2))
        if i == 0:
            ax[i,0].set_title('Measured')
        ax[i,0].set_xticks([]), ax[i,0].set_yticks([])
        plt.colorbar(p,fraction=0.046, pad=0.04)
        p = ax[i,1].imshow(temp_sim, clim=(0,1),cmap='jet')
        if i == 0:
            ax[i,1].set_title('Simulation')
        ax[i,1].set_xticks([]), ax[i,1].set_yticks([])
        plt.colorbar(p,fraction=0.046, pad=0.04)
        p = ax[i,2].imshow(temp_sim - temp_N, clim=(-1,1),cmap='jet')
        if i == 0:
            ax[i,2].set_title('Error')
        ax[i,2].set_xticks([]), ax[i,2].set_yticks([])
        plt.colorbar(p,fraction=0.046, pad=0.04)
    if pred_on == True:
        spacer = _atleast_2d(N_cal).shape[-1]
        for i in range(_atleast_2d(N_pred).shape[-1]):
            temp_N = _atleast_2d(N_pred)[:,i].reshape(tumor['Mask'].shape)
            temp_sim = _atleast_2d(prediction)[:,i].reshape(tumor['Mask'].shape)
            p = ax[i+spacer,0].imshow(temp_N, clim=(0,1),cmap='jet')
            ax[i+spacer,0].set_ylabel('Visit '+str(i+2+spacer))
            ax[i+spacer,0].set_xticks([]), ax[i+spacer,0].set_yticks([])
            plt.colorbar(p,fraction=0.046, pad=0.04)
            p = ax[i+spacer,1].imshow(temp_sim, clim=(0,1),cmap='jet')
            ax[i+spacer,1].set_xticks([]), ax[i+spacer,1].set_yticks([])
            plt.colorbar(p,fraction=0.046, pad=0.04)
            p = ax[i+spacer,2].imshow(temp_sim - temp_N, clim=(-1,1),cmap='jet')
            ax[i+spacer,2].set_xticks([]), ax[i+spacer,2].set_yticks([])
            plt.colorbar(p,fraction=0.046, pad=0.04)

############################# 3D Visualization ################################
def _visualize_3d(N_cal, N_pred, calibration, prediction, pred_on, tumor, cut = 'axial'):
    rows = N_cal.shape[N_cal.ndim-1]
    if pred_on == True:
        rows += N_pred.shape[N_pred.ndim-1]
    fig, ax = plt.subplots(rows, 3, subplot_kw={"projection": "3d"}, layout = "constrained")
    for i in range(_atleast_2d(N_cal).shape[-1]):
        temp_N = _atleast_2d(N_cal)[:,i].reshape(tumor['Mask'].shape).copy()
        temp_sim = _atleast_2d(calibration)[:,i].reshape(tumor['Mask'].shape).copy()
        temp_mask = tumor['Mask'].copy()
        # center = round(_centerMass(temp_N))
        center = ndi.center_of_mass(temp_N)
        if cut == 'axial':
            temp_N[:,:,round(center[2]):] = 0
            temp_sim[:,:,round(center[2]):] = 0
            temp_sim[0,:,:] = 0
            temp_mask[:,:,round(center[2]):] = 0
        elif cut == 'sagittal':
            temp_N[:,round(center[1]):,:] = 0
            temp_sim[:,round(center[1]):,:] = 0
            temp_sim[0,:,:] = 0
            temp_mask[:,round(center[1]):,:] = 0
            
        if i == 0:
            title_string = 'Measured'
        else:
            title_string = None   
        voxelarray_tumor, colorarray_tumor, voxelarray_breast, colorarray_breast, sm = _getTumorInBreast(temp_N, temp_mask)
        _plotVoxelArray(ax[i,0], voxelarray_tumor, colorarray_tumor, voxelarray_breast, colorarray_breast, sm, visit = i+2, title = title_string)
        
        if i == 0:
            title_string = 'Simulated'
        else:
            title_string = None 
        voxelarray_tumor, colorarray_tumor, voxelarray_breast, colorarray_breast, sm = _getTumorInBreast(temp_sim, temp_mask)
        _plotVoxelArray(ax[i,1], voxelarray_tumor, colorarray_tumor, voxelarray_breast, colorarray_breast, sm, title = title_string)
        
        if i == 0:
            title_string = 'Error'
        else:
            title_string = None 
        voxelarray_tumor, colorarray_tumor, voxelarray_breast, colorarray_breast, sm = _getTumorInBreast(temp_sim - temp_N, temp_mask, clim = [-1,1])
        _plotVoxelArray(ax[i,2], voxelarray_tumor, colorarray_tumor, voxelarray_breast, colorarray_breast, sm, title = title_string)
    if pred_on == True:
        spacer = _atleast_2d(N_cal).shape[-1]
        for i in range(_atleast_2d(N_pred).shape[-1]):
            temp_N = _atleast_2d(N_pred)[:,i].reshape(tumor['Mask'].shape).copy()
            temp_sim = _atleast_2d(prediction)[:,i].reshape(tumor['Mask'].shape).copy()
            temp_mask = tumor['Mask'].copy()
            # center = round(_centerMass(temp_N))
            center = ndi.center_of_mass(temp_N)
            if cut == 'axial':
                temp_N[:,:,round(center[2]):] = 0
                temp_sim[:,:,round(center[2]):] = 0
                temp_sim[0,:,:] = 0
                temp_mask[:,:,round(center[2]):] = 0
            elif cut == 'sagittal':
                temp_N[:,round(center[1]):,:] = 0
                temp_sim[:,round(center[1]):,:] = 0
                temp_sim[0,:,:] = 0
                temp_mask[:,round(center[1]):,:] = 0
            voxelarray_tumor, colorarray_tumor, voxelarray_breast, colorarray_breast, sm = _getTumorInBreast(temp_N, temp_mask)
            _plotVoxelArray(ax[i+spacer,0], voxelarray_tumor, colorarray_tumor, voxelarray_breast, colorarray_breast, sm, visit = i+2+spacer)
            
            voxelarray_tumor, colorarray_tumor, voxelarray_breast, colorarray_breast, sm = _getTumorInBreast(temp_sim, temp_mask)
            _plotVoxelArray(ax[i+spacer,1], voxelarray_tumor, colorarray_tumor, voxelarray_breast, colorarray_breast, sm)
            
            voxelarray_tumor, colorarray_tumor, voxelarray_breast, colorarray_breast, sm = _getTumorInBreast(temp_sim - temp_N, temp_mask, clim = [-1,1])
            _plotVoxelArray(ax[i+spacer,2], voxelarray_tumor, colorarray_tumor, voxelarray_breast, colorarray_breast, sm)
        
def _getTumorInBreast(curr, breast, clim = [0,1]):   
    norm = col.Normalize(vmin = clim[0], vmax = clim[1])
    sm = cm.ScalarMappable(norm = norm, cmap = 'jet')
    color_map = sm.to_rgba(curr.reshape(-1)).reshape((breast.shape + (4,)))
    tumor_colors = np.empty(curr.shape, dtype=object)
    for i in range(curr.shape[0]):
        for j in range(curr.shape[1]):
            for k in range(curr.shape[2]):
                tumor_colors[i,j,k] = col.rgb2hex(color_map[i,j,k,:-1]) + _alphaToHex(1.0)
                
    voxelarray_breast = np.array(np.array(breast, dtype=bool))
    colorarray_breast = np.empty(voxelarray_breast.shape, dtype=object)
    colorarray_breast[np.array(breast, dtype=bool)] = '#bfbfbf' + _alphaToHex(0.3)
    
    voxelarray_tumor = np.array(np.array(curr, dtype=bool))
    colorarray_tumor = np.empty(voxelarray_tumor.shape, dtype=object)
    colorarray_tumor[np.array(curr, dtype=bool)] = tumor_colors[np.array(curr, dtype=bool)]
    return voxelarray_tumor, colorarray_tumor, voxelarray_breast, colorarray_breast, sm

def _plotVoxelArray(ax, voxelarray_tumor, colorarray_tumor, voxelarray_breast, colorarray_breast, sm, visit = None, title = None):
    light = col.LightSource(azdeg = 0, altdeg = 30)
    ax.voxels(voxelarray_breast, facecolors = colorarray_breast, shade = True, lightsource = light)
    ax.voxels(voxelarray_tumor, facecolors = colorarray_tumor, shade = True, lightsource = light)
    if visit != None:
        ax.set_zlabel('Visit '+str(visit))
    if title != None:
        ax.set_title(title)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.azim = 60
    ax.dist = 5
    ax.set_aspect('equal')
    plt.colorbar(sm, ax = ax,fraction=0.046, pad=0.04)
    
############################## Line plotting ##################################
def _plotCI(ax, tspan, simulation, labels, t_type, t_meas, measured = None, color = [0,0,1], arrow = True, order = 0, title = None):
    if simulation.shape[1] != 1:
        #plot confidence interval stuff
        median_sim = np.median(simulation, axis = 1)
        prctile = np.percentile(simulation, [0,100,25,75], axis = 1)
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