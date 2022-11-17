# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:44:32 2022

@author: Julien Demange-Chryst
"""

#%% Modules

import numpy as np

#%% Cantilever beam computer model

def fleche(x):
    FX,FY,E,w,t,L = x
    E = 10**9*E
    D = 4*L**3/(E*w*t) * np.sqrt((FX/w**2)**2 + (FY/t**2)**2)
    return [D]