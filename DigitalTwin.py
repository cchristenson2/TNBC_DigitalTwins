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

Last updated: 5/6/2024
"""
import numpy as np
import copy
import matplotlib.pyplot as plt

import LoadData as ld
import ReducedModel as rm
import ForwardModels as fwd
import Calibrations as cal
import Library as lib

############################# Global variables ################################
call_dict = {
    'fwd.RXDIF_2D_wAC': fwd.RXDIF_2D_wAC,
    'fwd.RXDIF_3D_wAC': fwd.RXDIF_3D_wAC,
    'fwd.OP_RXDIF_wAC': fwd.OP_RXDIF_wAC}

###############################################################################
###############################################################################
######################### Digital twin definition #############################
class DigitalTwin:
    def __init__(self, location, load_args = {}, ROM = False, ROM_args = {}, params = None):
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
        self.tumor = ld.LoadTumor_mat(location, **load_args)
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
        valid_cal = ['LM_FOM', 'LM_ROM', 'gwMCMC_ROM']
        if cal_type in valid_cal:
            if cal_type == 'LM_FOM':
                self.params, self.cal_stats, self.model = \
                    cal.calibrateRXDIF_LM(self.tumor, self.params, **cal_args)
                self.cal_type = 'LM'
            else:
                if hasattr(self, 'ROM'):
                    if cal_type == 'LM_ROM':
                        self.params, self.cal_stats, self.model = \
                            cal.calibrateRXDIF_LM_ROM(self.tumor, self.ROM, 
                                                      self.params, **cal_args)
                        self.cal_type = 'LM'
                    else:
                        if hasattr(self, 'priors'):
                            if cal_type == 'gwMCMC_ROM':
                                self.params, self.ensemble_sampler, self.model = \
                                    cal.calibrateRXDIF_gwMCMC_ROM(self.tumor, 
                                                                  self.ROM, 
                                                                  self.params, 
                                                                  self.priors, 
                                                                  **cal_args)
                                self.cal_type = 'Bayes'
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
    def predict(self, dt = 0.5, threshold = 0.0, plot = False, parallel = False):
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
        t_ind = t[1:]/dt
        t_type = t_type_full[1:]
        N_meas = self.tumor['N']
        if self.tumor['Mask'].ndim==2:
            N0 = self.tumor['N'][:,:,0]
            if 'Future N' in self.tumor:
                N_meas = np.concatenate((N_meas, self.tumor['Future N']),axis = 2)
        else:
            N0 = self.tumor['N'][:,:,:,0]
            if 'Future N' in self.tumor:
                N_meas = np.concatenate((N_meas, self.tumor['Future N']),axis = 3)
        tspan = np.arange(0,t[-1]+dt,dt)
        default_trx_params = {'t_trx': self.tumor['t_trx']}
            
        parameters = self._unpackParameters()
        calibrations = []
        predictions = []
        cell_tc = np.array([]).reshape(tspan.size,0)
        volume_tc = np.array([]).reshape(tspan.size,0)
        samples = len(parameters)
        for i in range(len(parameters)):
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
                    cal = sim[:,:,int(t_ind[t_type==0])].reshape(-1,t_ind[t_type==0].size)
                    pred = sim[:,:,int(t_ind[t_type==1])].reshape(-1,t_ind[t_type==1].size)
                else:
                    cal = sim[:,:,:,int(t_ind[t_type==0])].reshape(-1,t_ind[t_type==0].size)
                    pred = sim[:,:,:,int(t_ind[t_type==1])].reshape(-1,t_ind[t_type==1].size)
                    temp1, temp2 = _maps_to_timecourse(sim, threshold, 
                                                       self.tumor['theta'], 
                                                       self.tumor['h'], 
                                                       reduced = False, V = None)
            else:
                N0 = self.ROM['ReducedTumor']['N_r'][:,0]
                operators = lib.getOperators(curr,self.ROM)
                sim, drugs = call_dict[self.model](N0, operators, trx_params, tspan, dt)
                cal = self.ROM['V'] @ sim[:,int(t_ind[t_type==0])]
                pred = self.ROM['V'] @ sim[:,int(t_ind[t_type==1])]
                temp1, temp2 = _maps_to_timecourse(sim, threshold, 
                                                   self.tumor['theta'], 
                                                   self.tumor['h'], 
                                                   reduced = True, V = self.ROM['V']) 
            cell_tc = np.concatenate((cell_tc, temp1), axis = 1)
            volume_tc = np.concatenate((volume_tc, temp2), axis = 1)
            calibrations.append(cal)
            predictions.append(pred)
        
        cell_measured, volume_measured = _maps_to_timecourse(N_meas, threshold, 
                                           self.tumor['theta'], self.tumor['h'], 
                                           reduced = False, V = None)
        self.simulations = {'cell_tc': cell_tc, 'volume_tc': volume_tc,
                            'maps_cal': calibrations, 'maps_pred': predictions, 
                            'cell_measured': cell_measured, 
                            'volume_measured':volume_measured}

        #Plot outputs if requested - mean sample visualized if MCMC was used
        if plot == True:
            if samples > 1:
                mean_calibration = np.mean(np.array(calibrations), axis = 0)
                mean_prediction = np.mean(np.array(predictions), axis = 0)
            else:
                mean_calibration = np.array(calibrations)
                mean_prediction = np.array(predictions)
            #Visualize volume and cell time courses compared to measured data
            #Ignored for now, need to decide how to visualize best
            # rows = mean_calibration.shape(mean_calibration.ndim-1) + mean_prediction.shape(mean_prediction.ndim-1)
            # if self.tumor['Mask'].ndim == 2:
            #     print(0)
            # else:     
            #     fig, ax = plt.subplots(rows, 3, subplot_kw={"projection": "3d"})
                
            fig, ax = plt.subplots(2,1)
            #Cell time course plot
            _plotCI(ax[0], tspan, cell_tc, ['Time (days)', 'Cell Count'], t_type_full, t, cell_measured)
            #Volume time course plot
            _plotCI(ax[1], tspan, volume_tc, ['Time (days)', 'Volume (mm^3)'], t_type_full, t, volume_measured)
            #Drug A and C plots need to write but not worried about it yet

##################### Internal DigitalTwin functions ##########################  
    def _unpackParameters(self):
        """
        Returns list of length samples (1 for LM calibration) with a dictionary for each sample.
        Represents

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
        default_values = [5e-3, 0.05, 0.4, 0.60, 3.25, 0.0]
        
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
        else:
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
    
    def delete(self):
        self.value = None        

###############################################################################
############################ Internal Functions ###############################
def _atleast_2d(arr):
    if np.atleast_1d(arr).ndim < 2:
        return np.expand_dims(np.atleast_1d(arr),axis=1)
    else:
        return arr
    
def _atleast_4d(arr):
    if arr.nidm < 4:
        return np.expand_dims(np.atleast_3d(arr),axis=3)
    else:
        return arr
    
def CCC_calc(a, b):
    return 0

def Dice_calc(a, b):
    return 0

def getStats_SingleTime(a, b):
    return 0 

def _update_trx_params(curr_params, trx_params):
    trx_params_new = copy.deepcopy(trx_params)
    trx_params_new['beta'] = np.array([curr_params['beta_a'], 
                                       curr_params['beta_c']])
    return trx_params_new

def _maps_to_timecourse(maps, threshold, theta, h, reduced = False, V = None):
    cell_tc = np.array([])
    volume_tc = np.array([])
    for i in range(maps.shape[maps.ndim-1]):
        if reduced == False:
            if maps.ndim == 3:
                temp = maps[:,:,i]
            else:
                temp = maps[:,:,:,i]
        else:
            temp = V @ maps[:,i]
        cell_tc = np.append(cell_tc, np.sum(temp[temp>=threshold]) * theta)
        volume_tc = np.append(volume_tc, temp[temp>=threshold].size * np.prod(h))
    return cell_tc.reshape((-1,1)), volume_tc.reshape((-1,1))

# def _alphaToHex(n):
#     string = '{0:x}'.format(round(n*255))
#     if len(string) == 1:
#         string = '0'+string
#     return string
 
def _plotCI(ax, tspan, simulation, labels, t_type, t_meas, measured = None):
    if simulation.shape[1] != 1:
        #plot confidence interval stuff
        mean_sim = np.mean(simulation, axis = 1)
        prctile = np.percentile(simulation, [1,99,25,75], axis = 1)
        ax.fill_between(tspan,prctile[0,:],prctile[1,:],color=[0,0,1,0.1],
                        label='Simulation - range',zorder=1)
        ax.fill_between(tspan,prctile[2,:],prctile[3,:],color=[0,0,1,0.5],
                        label='Simulation - IQR',zorder=2)
        line_label = 'Simulation - mean'
    else:
        mean_sim = simulation
        line_label = 'Simulation'
    ax.plot(tspan, mean_sim, 'b-', linewidth = 1,label=line_label,zorder=3)
    if measured is not None:
        ax.scatter(t_meas, measured, color = 'k', label = 'Measured',zorder=4)
    
    if np.any(t_type==1):
        point = t_meas[np.where(t_type==0)[0][-1]]
        ax.axvline(point,color = 'r',linestyle = '--',linewidth = 0.5,zorder=5)
        #Get tiny triangle to signal direction of prediction
        limits = ax.get_ybound()
        ax.arrow(point+0.5,limits[0] + limits[1]*0.03, 1, 0, color = 'r',head_width = limits[1]*0.05, head_length = 0.5, length_includes_head = True,zorder=6)
        ax.arrow(point+0.5,limits[1] - limits[1]*0.03, 1, 0, color = 'r',head_width = limits[1]*0.05, head_length = 0.5, length_includes_head = True,zorder=6)
        if _findNearest(limits,np.mean(measured)) == 0:
            ax.text(point+2.0,limits[0] + limits[1]*0.03 - limits[1]*0.025, 'Prediction', color = 'r', fontsize = 'xx-small',zorder=7)
        else:
            ax.text(point+2.0,limits[1] - limits[1]*0.03 - limits[1]*0.025 , 'Prediction', color = 'r', fontsize = 'xx-small',zorder=7)
        
    ax.legend(fontsize='xx-small')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    
def _findNearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx