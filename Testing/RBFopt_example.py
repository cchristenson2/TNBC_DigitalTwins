# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 18:49:35 2024

@author: Chase Christenson
"""

from amplpy import modules
import rbfopt
import numpy as np
def obj_funct(x):
  return x[0]*x[1] - x[2]

bb = rbfopt.RbfoptUserBlackBox(3, np.array([0] * 3), np.array([10] * 3),
                               np.array(['R', 'I', 'R']), obj_funct)
settings = rbfopt.RbfoptSettings(max_evaluations=50,minlp_solver_path=modules.find('bonmin'),nlp_solver_path=modules.find('ipopt'))
alg = rbfopt.RbfoptAlgorithm(settings, bb)
val, x, itercount, evalcount, fast_evalcount = alg.optimize()