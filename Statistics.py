# Statistics.py
"""
Random statisical analysis functions

Last updated: 5/8/2024
"""
import numpy as np

def CCC(a,b):
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
    return 2*(cov/n)/((vara/n)+(varb/n)+(mu_a - mu_b)**2)