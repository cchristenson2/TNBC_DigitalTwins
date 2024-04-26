import os
import LoadData as ld


#Set paths
home = os.path.dirname(os.getcwd())
datapath = home + '\Data\PatientData_ungrouped\\'
#Get tumor information in folder
files = os.listdir(datapath)

#Load the first patient
print(files[0])
tumor = ld.LoadTumor_mat(datapath + files[0], crop2D = 'false')