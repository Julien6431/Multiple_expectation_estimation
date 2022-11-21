# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:31:11 2022

@author: Julien Demange-Chryst
"""


#%% Modules

import numpy as np
import openturns as ot

import sys
sys.path.append("../src")
from aISCV_algorithm import multiple_expectations_iscv

from tqdm import tqdm

#%% Settting and reference values

input_distr = ot.Normal(1)
J = 10

def phi(x):
    xp = np.array(x)
    xp = xp[0]
    return [xp**(2*j) for j in range(1,J+1)]

ot_phi = ot.PythonFunction(1,J,phi)


ref_values = np.zeros(J)
for i in tqdm(range(1,J+1)):
    ref_values[i-1] = np.array(input_distr.getMoment(2*i))
   
    
   
#%% Execution of the algorithm

n_rep = 2*10**2
MC = np.zeros((n_rep,J))
ISCV = np.zeros((n_rep,J))

N = 2*10**4

weights = np.ones(J)/ref_values**2

for n in tqdm(range(n_rep)):
    X = np.array(input_distr.getSample(N))
    for j in range(1,J+1):
        MC[n,j-1] = np.mean(X**(2*j))
    ISCV[n],_,_,_ = multiple_expectations_iscv(ot_phi,weights,input_distr,N_max=N,cross_entropy="SG")

print("\nVariances : \n")
print(f"MC : {np.sum(weights*np.var(MC,axis=0))}")
print(f"ISCV : {np.sum(weights*np.var(ISCV,axis=0))}")

#%% Save data

np.savez("data/Gaussian_moment_10.npz",
         ref_values=ref_values,
         MC=MC,
         ISCV=ISCV)
