# DigitalTwin.py
""" Defines digital twin object
    Consists of:
        - Tumor data, acquired and yet to be acquired
        - Treatment controls
        - ROM (if requested)
    Optional:
        - Parameters
        - Volume and cell time courses

Last updated: 5/2/2024
"""
import numpy as np

import LoadData as ld
import ReducedModel as rm
import Calibrations as cal

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
            else:
                raise ValueError('if ROM = True, ROM_args must be passed in')
     
    def setParams(self, params):
        self.params = params

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
        if cal_type == 'LM_FOM':
            self.params, self.cal_stats = cal.calibrateRXDIF_LM(self.tumor, self.params, **cal_args)
        else:
            if hasattr(self, 'ROM'):
                if cal_type == 'LM_ROM':
                    self.params, self.cal_stats = cal.calibrateRXDIF_LM_ROM(self.tumor, self.ROM, self.params, **cal_args)
                else:
                    if hasattr(self, 'priors'):
                        if cal_type == 'gwMCMC_ROM':
                            self.params, self.ensemble_sampler = cal.calibrateRXDIF_gwMCMC_ROM(self.tumor, self.ROM, self.params, self.priors, **cal_args)
                    else:
                        raise ValueError('Priors must be contained in twin object for bayesian calibrations')                   
            else:
                 raise ValueError('ROM must be contained in twin object to use ROM based calibrations')   
    
    def getPriors(self, params):
        self.priors = cal.generatePriors(params)
        
    @staticmethod
    def CCC_calc(a, b):
        return 0
    
    @staticmethod
    def Dice_calc(a, b):
        return 0
    
    @staticmethod
    def getStats_SingleTime(a, b):
        return 0 
    
    def predict(self, threshold = 0.25, plot = False):
        return 0
        #First check if parameters have values assigned
        for elem in self.params:
            if elem.value is None:
                raise ValueError('Some required parameters do not have values '
                                 'assigned. Run calibrateTwin() first')

###############################################################################
# Parameter class object and functions
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