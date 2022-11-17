# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 13:36:05 2022

@author: Julien Demange-Chryst
"""

#%% Modules

import numpy as np
import openturns as ot
from SquareF import SquareF
from cantilever_beam_model import fleche

import sys
sys.path.append("../src")
from adaptative_CV import multiple_expectations_cv

from tqdm import tqdm

#%% Sobol Monte Carlo

def compute_sobol_MC(phi,input_distr,N):
    dim = input_distr.getDimension()
    ot_phi = ot.PythonFunction(dim,1,phi)
    
    X1 = input_distr.getSample(N)
    X2 = input_distr.getSample(N)
    Y = ot_phi(X1)
    
    X1 = np.array(X1)
    X2 = np.array(X2)
    Y = np.array(Y)
    
    esp_phi = np.mean(Y)
    esp_phi_square = np.mean(Y**2)
    var_phi = np.var(Y,ddof=1)
    
    sobols = np.zeros(dim)
    hat_I = np.zeros(dim+2)
    
    hat_I[-2] = esp_phi
    hat_I[-1] = esp_phi_square
    
    for i in range(dim):
        u_arr = np.array([i])
        com_u = np.delete(np.arange(dim),i)
        
        Xu = np.zeros((N,dim))
        Xu[:,u_arr] = X1[:,u_arr]
        Xu[:,com_u] = X2[:,com_u]
        
        Yu = np.array(ot_phi(ot.Sample(Xu)))
        
        sobols[i] = (np.mean(Y*Yu) - esp_phi**2)  
        hat_I[i] = np.mean(Y*Yu)

    return sobols/var_phi,hat_I


#%%

dim=6

mu_FX,sigma_FX,_ = ot.LogNormalMuSigmaOverMu(556.8,0.08).evaluate()
distr_FX = ot.LogNormal(mu_FX,sigma_FX)
mu_FY,sigma_FY,_ = ot.LogNormalMuSigmaOverMu(453.6,0.08).evaluate()
distr_FY = ot.LogNormal(mu_FY,sigma_FY)
mu_E,sigma_E,_ = ot.LogNormalMuSigmaOverMu(200,0.06).evaluate()
distr_E = ot.LogNormal(mu_E,sigma_E)
distr_w = ot.Normal(0.062,0.1*0.062)
distr_t = ot.Normal(0.0987,0.1*0.0987)
distr_L = ot.Normal(4.29,0.1*4.29)
input_distr = ot.ComposedDistribution([distr_FX,distr_FY,distr_E,distr_w,distr_t,distr_L])

r = compute_sobol_MC(fleche,input_distr,10**7)


#%% Run algorithms

def get_ot_func(phi,dim):
    
    def pf_functions(x):
        xp = np.array(x)
        x1,x2 = xp[:dim],xp[dim:]
        res = []
        for i in range(dim):
            u_arr = np.array([i])
            com_u = np.delete(np.arange(dim),i)
            xu = np.zeros(dim)
            xu[u_arr] = x1[u_arr]
            xu[com_u] = x2[com_u]
            res.append(phi(x1)[0]*phi(xu)[0])
        # res.append((phi(x1)[0]+phi(x2)[0])/2)
        # res.append((phi(x1)[0]**2+phi(x2)[0]**2)/2)
        res.append(phi(x1)[0])
        res.append(phi(x1)[0]**2)
        return res
        
    return ot.PythonFunction(2*dim,dim+2,pf_functions)


def compute_sobol(phi,input_distr,N_tot,n_rep,weights):
    dim = input_distr.getDimension()
    input_distr_std = ot.Normal(dim)
    square_input_distr = SquareF(input_distr_std)
    
    def qsdg(u):
        X = np.zeros(dim)
        for i in range(dim):
            d = input_distr.getMarginal(i)
            y = ot.Normal(1).computeCDF(u[i])
            X[i] = np.array(d.computeQuantile(y))
                
        return phi(X)
    
    func = get_ot_func(qsdg,dim)
    
    sobol_mc = np.zeros((n_rep,dim))
    sobol_cv = np.zeros((n_rep,dim))
    expectations_mc = np.zeros((n_rep,dim+2))
    expectations_cv = np.zeros((n_rep,dim+2))
    
    for n in tqdm(range(n_rep)):
        #with control variates
        hat_I,X,W = multiple_expectations_cv(func,weights,square_input_distr,N_tot,cross_entropy="SG",diag=False)
        expectations_cv[n] = hat_I
        esp_phi = hat_I[-2]
        var_phi = hat_I[-1] - esp_phi**2
        
        #Control variates
        sobol_cv[n] = (hat_I[:-2] - esp_phi**2)/var_phi
        
        #Monte Carlo
        sobol_mc[n],expectations_mc[n] = compute_sobol_MC(phi, input_distr, N_tot)
    
    return sobol_mc,sobol_cv,expectations_mc,expectations_cv

def run_algorithms(N_tot=10**4,n_rep=10**2,weights=np.ones(dim+2)):
    mu_FX,sigma_FX,_ = ot.LogNormalMuSigmaOverMu(556.8,0.08).evaluate()
    distr_FX = ot.LogNormal(mu_FX,sigma_FX)
    mu_FY,sigma_FY,_ = ot.LogNormalMuSigmaOverMu(453.6,0.08).evaluate()
    distr_FY = ot.LogNormal(mu_FY,sigma_FY)
    mu_E,sigma_E,_ = ot.LogNormalMuSigmaOverMu(200,0.06).evaluate()
    distr_E = ot.LogNormal(mu_E,sigma_E)
    distr_w = ot.Normal(0.062,0.1*0.062)
    distr_t = ot.Normal(0.0987,0.1*0.0987)
    distr_L = ot.Normal(4.29,0.1*4.29)
    input_distr = ot.ComposedDistribution([distr_FX,distr_FY,distr_E,distr_w,distr_t,distr_L])
    
    phi = fleche
        
    sobol_mc,sobol_cv,expectations_mc,expectations_cv = compute_sobol(phi,input_distr,N_tot,n_rep,weights)

    return sobol_mc,sobol_cv,expectations_mc,expectations_cv,input_distr


#%%

if __name__ == '__main__':
    
    keys = ["-N_tot=","-n_rep="]         
                            
    arguments = {"N_tot" : "20000",
                 "n_rep" : "200"}    
                                                                  
    for i in range(1,len(sys.argv)):                                                                                                       
        for key in keys:     
            if sys.argv[i].find(key) == 0:    
                arguments[key[1:-1]] = sys.argv[i][len(key):]                                                                                          
                print(f"The Given value is: {sys.argv[i][len(key):]}")                                                                     
                break   

    n_rep = int(arguments["n_rep"])
    N_tot = int(arguments["N_tot"])
        
    sobol_mc,sobol_cv,expectations_mc,expectations_cv,input_distr = run_algorithms(N_tot,n_rep)
        
    
    print("Variances expectations: \n")
    print(f"MC : {np.sum(np.var(expectations_mc,axis=0))}")
    print(f"CV : {np.sum(np.var(expectations_cv,axis=0))}")
    
    print("Variances Sobol: \n")
    print(f"MC : {np.sum(np.var(sobol_mc,axis=0))}")
    print(f"CV : {np.sum(np.var(sobol_cv,axis=0))}")
    
    
#%%

ref_values = np.load("data/ref_values_cantilever_beam_sobol.npy")
print(ref_values)

#%%

np.savez("data/Cantilever_beam_Sobol.npz",
         ref_values=r,
         sobolMC=sobol_mc,
         sobolCV=sobol_cv,
         espMC=expectations_mc,
         espCV=expectations_cv)

    
#%%    


ref_values = np.load("data/ref_values_cantilever_beam_sobol.npy")
fig,ax = plt.subplots(2,3,figsize=(15,9))
for i in range(2):
    for j in range(3):
        ax[i,j].boxplot(ref_values[3*i+j].reshape((-1,1)),positions=[0])
        ax[i,j].boxplot(sobol_mc[:,3*i+j],sym='k+',positions=[.5])
        ax[i,j].boxplot(sobol_cv[:,3*i+j],sym='k+',positions=[1])
fig.suptitle("Boxplots Sobol indices Cantilever beam in Gaussian space")
#fig.savefig("../Figures/Sobol_cantilever_beam_std.pdf",bbox_inches='tight')
plt.show(block=True)
