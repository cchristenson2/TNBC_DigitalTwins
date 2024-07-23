import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import LoadData as ld

import PaclitaxelOnly as pac

def find_load_calibration(name, files, path):
    for i in range(len(files)):
        temp = files[i].removesuffix('.pkl')
        if temp == name:
            return pickle.load(open(path + files[i],'rb'))

if __name__ == '__main__':
    #Set paths
    home = os.path.dirname(os.getcwd())
    datapath = home + '\Data\PatientData_ungrouped_pac\DoseDense\\'
    
    calpath = home + '\Results\ABC_calibrated_twins_3TP\\'
    #Get results in folder
    files = os.listdir(datapath)
    n = len(files)
    cal_files = os.listdir(calpath)
    
    pac_regimen = ld.LoadPaclitaxelRegimen(datapath + files[1])
    name = files[1].removesuffix('.mat')
    
    twin = find_load_calibration(name, cal_files, calpath)
    
    pac_sim = pac.predict_paclitaxel(twin, pac_regimen, threshold = 0.25, plot = True)
    
    