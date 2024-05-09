# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:44:39 2024

@author: Chase Christenson
"""

from multiprocessing import Pool

def square(x):
    print(x**2)
    return x ** 2

if __name__ =='__main__':
    with Pool() as pool:
        list(pool.map(square, range(10)))