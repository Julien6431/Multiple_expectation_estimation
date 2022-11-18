# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 13:36:05 2022

@author: Julien Demange-Chryst
"""

#%% Modules

import numpy as np
import openturns as ot
from tqdm import tqdm
from cantilever_beam_model import fleche

import sys
sys.path.append("../src")
from adaptative_CV import multiple_expectations_cv


#%% Input distribution

def get_input_distr(parameters):

    dim = 6
    delta1,delta2,delta3,delta4,delta5,delta6,delta7,delta8,delta9 = parameters    
    
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
    corr_matrix[3,4] = delta7
    corr_matrix[3,5] = delta8
    corr_matrix[4,5] = delta9
        
    Copule_normale = ot.NormalCopula(corr_matrix)
    input_distr = ot.ComposedDistribution([distr_FX,distr_FY,distr_E,distr_w,distr_t,distr_L],Copule_normale)

    return input_distr

#%% Marginal distributions of the parameters

distr_param1 = ot.Uniform(525,575)
distr_param2 = ot.Uniform(425,475)
distr_param3 = ot.Uniform(175,225)
distr_param4 = ot.Uniform(0.06,0.07)
distr_param5 = ot.Uniform(0.09,0.1)
distr_param6 = ot.Uniform(4,5)
distr_param7 = ot.Uniform(-0.6,0)
distr_param8 = ot.Uniform(0,0.5)
distr_param9 = ot.Uniform(0,0.5)

distr_params = ot.ComposedDistribution([distr_param1,distr_param2,distr_param3,distr_param4,distr_param5,distr_param6,distr_param7,distr_param8,distr_param9])

#%% Generation of a set of parameters

J = 10**2
sample_params = np.array(ot.LHSExperiment(distr_params, J).generate())
input_distrs = [get_input_distr(params) for params in sample_params]

phi = ot.PythonFunction(6,1,fleche)

#%% Execution of the algorithm

n_rep = 2*10**2
MC_individual = np.zeros((n_rep,J))
MC_mixture = np.zeros((n_rep,J))
ISCV = np.zeros((n_rep,J))

N = 2*10**4

mc_distr = ot.Mixture(input_distrs)
def get_PDFs(x):
    return [d.computePDF(x) for d in input_distrs]
compute_input_PDF = ot.PythonFunction(6,J,get_PDFs)

weights = np.ones(J)

for n in tqdm(range(n_rep)):
    #MC individual
    Nj = N//J
    for j in range(J):
        Xj = input_distrs[j].getSample(Nj)
        Yj = np.array(phi(Xj))
        MC_individual[n,j] = np.mean(Yj)
    
    #MC mixture
    X = mc_distr.getSample(N)
    f_X = np.array(compute_input_PDF(X))
    h_X = np.array(mc_distr.computePDF(X)) 
    W = f_X/h_X
    Y = np.array(phi(X))
    MC_mixture[n] = np.mean(Y*W,axis=0)
    
    #Control variates
    ISCV[n],_,_,_ = multiple_expectations_cv(phi,weights,input_distrs,N_max=N,cross_entropy="SG")
    
print("\nVariances : \n")
print(f"MC individual : {np.sum(weights*np.var(MC_individual,axis=0))}")
print(f"MC mixture : {np.sum(weights*np.var(MC_mixture,axis=0))}")
print(f"ISCV : {np.sum(weights*np.var(ISCV,axis=0))}")


#%% Save data

ref_values = np.load("data/Cantilever_beam_expectations.npz")['ref_values']

np.savez("data/Cantilever_beam_expectations.npz",
         ref_values=ref_values,
         MC_ind=MC_individual,
         MC_mixt=MC_mixture,
         ISCV=ISCV,
         distr_params=sample_params)