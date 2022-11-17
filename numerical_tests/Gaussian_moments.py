# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:31:11 2022

@author: jdemange
"""


#%% Modules

import numpy as np
import openturns as ot
import matplotlib.pyplot as plt

import sys
sys.path.append("../src")
from adaptative_CV import multiple_expectations_cv

from tqdm import tqdm

#%% Settting

input_distr = ot.Normal(1)
J = 10

def phi(x):
    xp = np.array(x)
    xp = xp[0]
    return [xp**(2*j) for j in range(1,J+1)]

ot_phi = ot.PythonFunction(1,J,phi)

#%%

ref_values = np.zeros(J)
for i in tqdm(range(1,J+1)):
    ref_values[i-1] = np.array(input_distr.getMoment(2*i))
   
    
   
#%%

n_rep = 200#2*10**2
MC = np.zeros((n_rep,J))
CV = np.zeros((n_rep,J))

N = 2*10**4

weights = np.ones(J)/ref_values**2
for n in tqdm(range(n_rep)):
    X = np.array(input_distr.getSample(N))
    for j in range(1,J+1):
        MC[n,j-1] = np.mean(X**(2*j))
    
    CV[n],_,_,_ = multiple_expectations_cv(ot_phi,weights,input_distr,N_max=N,cross_entropy="SG")

print("\nVariances : \n")
print(f"MC : {np.sum(weights*np.var(MC,axis=0))}")
print(f"CV : {np.sum(weights*np.var(CV,axis=0))}")

#%% Save data

np.savez("data/Gaussian_moment_10.npz",
         ref_values=ref_values,
         MC=MC,
         CV=CV)

#%%

fig,ax = plt.subplots(1,5,figsize=(25,15))
for j in range(5):
    ax[j].boxplot(ref_values[j].reshape((-1,1)),sym='k+',positions=[0])
    ax[j].boxplot(MC[:,j],sym='k+',positions=[.5])
    ax[j].boxplot(CV[:,j],sym='k+',positions=[1])
    
#%%

fig,ax = plt.subplots(2,5,figsize=(25,15))
for i in range(2):
    for j in range(5):
        ax[i,j].boxplot(ref_values[5*i+j].reshape((-1,1)),sym='k+',positions=[0])
        ax[i,j].boxplot(MC[:,5*i+j],sym='k+',positions=[.5])
        ax[i,j].boxplot(CV[:,5*i+j],sym='k+',positions=[1])


#%%

weights = np.ones(J)/ref_values**2
hat_I,_,_,g_alphas = multiple_expectations_cv(ot_phi,weights,input_distr,N_max=N,cross_entropy="SG")


fig,ax = plt.subplots(figsize=(18,12))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


xx = np.linspace(-10,10,1001).reshape((-1,1))

legends = ["$\mathcal{N}_1(0,1)$"] + ["$g_{\\alpha_1}$","$g_{\\alpha_2}$","$g_{\\alpha_3}$"]

for i,g in enumerate(g_alphas):
    yy = np.array(g.computePDF(xx)).flatten()
    ax.plot(xx,yy,label=legends[i],linewidth=4)
ax.set_xlabel("x",fontsize="40")
ax.set_ylabel("PDF",fontsize="40")
ax.tick_params(axis='x',labelsize='30')
ax.tick_params(axis='y',labelsize='30')
ax.legend(legends[:len(g_alphas)],fontsize="40")
ax.set_title("PDF",fontdict={'fontsize':'40','fontweight' : 'bold','verticalalignment': 'baseline'})

fig.savefig("../Figures/Gaussian_moments_10_example.pdf",bbox_inches='tight',dpi=700)





