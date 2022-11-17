# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 13:36:05 2022

@author: jdemange
"""

#%% Modules

import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from cantilver_beam_model import fleche

import sys
sys.path.append("../src")
from adaptative_CV import multiple_expectations_cv

from tqdm import tqdm

#%% Input

def get_input_distr(parameters):

    delta1,delta2,delta3,delta4,delta5,delta6,delta7,delta8,delta9 = parameters    

    dim = 6
    
    mu_FX,sigma_FX,_ = ot.LogNormalMuSigmaOverMu(delta1,0.08).evaluate()
    distr_FX = ot.LogNormal(mu_FX,sigma_FX)
    mu_FY,sigma_FY,_ = ot.LogNormalMuSigmaOverMu(delta2,0.08).evaluate()
    distr_FY = ot.LogNormal(mu_FY,sigma_FY)
    mu_E,sigma_E,_ = ot.LogNormalMuSigmaOverMu(delta3,0.06).evaluate()
    distr_E = ot.LogNormal(mu_E,sigma_E)
    
    distr_w = ot.Normal(delta4,0.1*delta4)
    distr_t = ot.Normal(delta5,0.1*delta5)
    distr_L = ot.Normal(delta6,0.1*delta6)
    
    corr_matrix = ot.CorrelationMatrix(dim)
    corr_matrix[3,4] = delta7#-0.55
    corr_matrix[3,5] = delta8#0.45
    corr_matrix[4,5] = delta9#0.45
        
    Copule_normale = ot.NormalCopula(corr_matrix)
    input_distr = ot.ComposedDistribution([distr_FX,distr_FY,distr_E,distr_w,distr_t,distr_L],Copule_normale)

    return input_distr

#%%

distr_param1 = ot.Uniform(525,575)#525,575
distr_param2 = ot.Uniform(425,475)#425,475
distr_param3 = ot.Uniform(175,225)#175.225
distr_param4 = ot.Uniform(0.06,0.07)
distr_param5 = ot.Uniform(0.09,0.1)
distr_param6 = ot.Uniform(3,6)#4,5
distr_param7 = ot.Uniform(-0.6,0)#-.6,0
distr_param8 = ot.Uniform(0,0.5)#0,0.5
distr_param9 = ot.Uniform(0,0.5)#0,0.5

distr_params = ot.ComposedDistribution([distr_param1,distr_param2,distr_param3,distr_param4,distr_param5,distr_param6,distr_param7,distr_param8,distr_param9])

#%%

J = 10**2#1.5*10**1
sample_params = np.array(ot.LHSExperiment(distr_params, J).generate())
#sample_params = np.array(distr_params.getSample(J))
input_distrs = [get_input_distr(params) for params in sample_params]

phi = ot.PythonFunction(6,1,fleche)

#%%

ref_values = np.zeros(J)
for i in tqdm(range(J)):
    d = input_distrs[i]
    X = d.getSample(10**6)
    Y = np.array(phi(X))
    ref_values[i] = np.mean(Y)
    

#%%

n_rep = 2*10**2
MC = np.zeros((n_rep,J))
CV = np.zeros((n_rep,J))

N = 2*10**4

mc_distr = ot.Mixture(input_distrs)
def hsd(x):
    return [d.computePDF(x) for d in input_distrs]
compute_input_PDF = ot.PythonFunction(6,J,hsd)

weights = np.ones(J)#/ref_values**2

for n in tqdm(range(n_rep)):
    X = mc_distr.getSample(N)
    f_X = np.array(compute_input_PDF(X))
    h_X = np.array(mc_distr.computePDF(X)) 
    W = f_X/h_X
    Y = np.array(phi(X))
    MC[n] = np.mean(Y*W,axis=0)
    CV[n],_,_ = multiple_expectations_cv(phi,weights,input_distrs,N_max=N,cross_entropy="SG")

print("\nVariances : \n")
print(f"MC : {np.sum(weights*np.var(MC,axis=0))}")
print(f"CV : {np.sum(weights*np.var(CV,axis=0))}")


#%%

fig,ax = plt.subplots(2,5,figsize=(25,15))
for i in range(2):
    for j in range(5):
        ax[i,j].boxplot(ref_values[5*i+j].reshape((-1,1)),sym='k+',positions=[0])
        ax[i,j].boxplot(MC[:,5*i+j],sym='k+',positions=[.5])
        ax[i,j].boxplot(CV[:,5*i+j],sym='k+',positions=[1])
        
#%%

fig,ax = plt.subplots(10,10,figsize=(25,25))
for i in range(10):
    for j in range(10):
        ax[i,j].boxplot(ref_values[10*i+j].reshape((-1,1)),sym='k+',positions=[0])
        ax[i,j].boxplot(MC[:,10*i+j],sym='k+',positions=[.5])
        ax[i,j].boxplot(CV[:,10*i+j],sym='k+',positions=[1])
        
#%%

np.savez("data/Cantilever_beam_expectations_36.npz",ref_values=ref_values,MC=MC,CV=CV,distr_params=sample_params)