# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:39:39 2024

@author: Chase Christenson
"""

import pickle
import DigitalTwin as dtwin

with open('test.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    twin = pickle.load(f)
    
print(twin.tumor)