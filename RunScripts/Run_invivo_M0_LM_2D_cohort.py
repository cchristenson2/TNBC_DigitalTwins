import numpy as np
import os
import matplotlib.pyplot as plt

import DigitalTwin as dtwin
import Statistics as st

if __name__ == '__main__':
    #Set args for problem
    load_args = {'crop2D': True, 'split': 2}
    
    bounds = {'d': np.array([1e-6, 1e-3]), 'k': np.array([1e-6, 0.1]),
              'alpha': np.array([1e-6, 0.8])}
    
    required_ops = ('A','B','H','T')
    params_ops = ('d','k','k','alpha')
    type_ops = ('G','l','l','G')
    
    cal_args = {'output': False, 'options':{'max_it':100,'j_freq':2}}
    
    #Set paths
    home = os.path.dirname(os.getcwd())
    datapath = home + '\Data\PatientData_ungrouped\\'
    files = os.listdir(datapath)
    
    sim_ttv = []
    meas_ttv = []
    sim_ttc = []
    meas_ttc = []
    for i in range(len(files)):
        #No idea why but a zip can only be used once so it breaks after first try
        zip_object = zip(required_ops, params_ops, type_ops)
        ROM_args = {'bounds': bounds, 'zipped': zip_object}
        
        twin = dtwin.DigitalTwin(datapath + files[i], load_args = load_args,
                                      ROM = True, ROM_args = ROM_args)

        params = {'d':dtwin.Parameter('d','g'), 
                  'k':dtwin.ReducedParameter('k','r',twin.ROM['V']),
                  'alpha':dtwin.Parameter('alpha','g'),
                  'beta_a':dtwin.Parameter('beta_a','g'), 
                  'beta_c': dtwin.Parameter('beta_c','g')}
        
        params['d'].setBounds(np.array([1e-6,1e-3]))
        params['k'].setBounds(np.array([1e-6,0.1]))
        params['k'].setCoeffBounds(twin.ROM['Library']['B']['coeff_bounds'])
        params['alpha'].setBounds(np.array([1e-6,0.8]))
        params['beta_a'].setBounds(np.array([0.35, 0.85]))
        params['beta_c'].setBounds(np.array([1.0, 5.5]))
        twin.setParams(params)
        
        twin.calibrateTwin('LM_ROM', cal_args)
        twin.predict(dt = 0.5, threshold = 0.25, plot = False, parallel = False)
        twin.simulationStats(threshold = 0.25)
        
        sim_ttv.append([twin.stats['calibration']['delTTV'], twin.stats['prediction']['delTTV']])
        sim_ttc.append([twin.stats['calibration']['delTTC'], twin.stats['prediction']['delTTC']])
        meas_ttv.append([twin.simulations['volume_measured'][1:] - twin.simulations['volume_measured'][0]])
        meas_ttc.append([twin.simulations['cell_measured'][1:] - twin.simulations['cell_measured'][0]])
        
        del twin
        print('Patient '+str(i)+' complete.')
        
    sim_ttv = np.squeeze(np.array(sim_ttv))
    sim_ttc = np.squeeze(np.array(sim_ttc))
    meas_ttc = np.squeeze(np.array(meas_ttc))
    meas_ttv = np.squeeze(np.array(meas_ttv))

    fig, ax = plt.subplots(2,2,layout="constrained")
    for i in range(2):
        if i == 0:
            string = 'Calibrated'
        else:
            string = 'Predicted'
        ax[0,i].scatter(meas_ttc[:,i], sim_ttc[:,i], s = 20, c = 'b', edgecolor = 'k', zorder=2)
        ax[0,i].plot(ax[0,i].get_xbound(), ax[0,i].get_ybound(), 'r--', zorder = 1)
        ax[0,i].set_xlabel('Measured $\Delta$TTC')
        ax[0,i].set_ylabel(string+' $\Delta$TTC')
        ax[0,i].set_title('CCC = '+ f'{st.CCC(meas_ttc[:,i], sim_ttc[:,i]):.3f}')
        
        ax[1,i].scatter(meas_ttv[:,i], sim_ttv[:,i], s = 20, c = 'b', edgecolor = 'k', zorder=2)
        ax[1,i].plot(ax[1,i].get_xbound(), ax[1,i].get_ybound(), 'r--', zorder = 1)
        ax[1,i].set_xlabel('Measured $\Delta$TTV')
        ax[1,i].set_ylabel(string+' $\Delta$TTV')
        ax[1,i].set_title('CCC = '+ f'{st.CCC(meas_ttv[:,i], sim_ttv[:,i]):.3f}')
        