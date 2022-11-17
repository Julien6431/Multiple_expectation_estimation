# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:52:00 2022

@author: jdemange
"""

#%% Modules

import numpy as np
import openturns as ot
from EMGM import EMGM
from sklearn.cluster import KMeans
from scipy.optimize import minimize,LinearConstraint,Bounds
from kneed import KneeLocator
# sys.path.append("../distributions/")
# from BananaShape import BananaShape

import matplotlib.pyplot as plt

#%% Optimisation

def get_alpha_star(alpha_0,Y,weights,f_X,h_X,g_curr_distr,beta):
    
    nb_esp = len(alpha_0)
    
    def g_alpha(alpha):
        a = np.sum(alpha*g_curr_distr,axis=1)
        a[a==0] = np.inf
        return a.reshape((-1,1))
    
    b = weights*(f_X*Y - beta*g_curr_distr)**2
    b = np.sum(b,axis=1).reshape((-1,1))
    
    f = lambda alpha:np.mean(b/g_alpha(alpha)/h_X)
    
    def grad_f(alpha):
        den = g_alpha(alpha)**2*h_X
        a = b/den
        a = a*g_curr_distr
        
        return -np.mean(a,axis=0)
    
            
    c = np.ones(nb_esp)

    bounds = Bounds(9*1e-12, 1)
    constraint = LinearConstraint(c.T, 1, 1)

    res = minimize(f,x0=alpha_0,method="SLSQP",jac=grad_f,bounds = bounds,constraints=constraint)
    return res.x



#%%


def multiple_expectations_cv(func,weights,input_distr,N_max=10**4,cross_entropy="SG",diag=False):    
    N1 = N_max//2
    Nk = N1//5
    
    #initialisation
    dim = func.getInputDimension()
    if type(input_distr)==list:
        nb_esp = max(func.getOutputDimension(),len(input_distr))
        init_distr = ot.Mixture(input_distr)
        def hsd(x):
            return [d.computePDF(x) for d in input_distr]
        compute_input_PDF = ot.PythonFunction(dim,nb_esp,hsd)
        aux_distrs = [init_distr]
    else:
        nb_esp = func.getOutputDimension()
        init_distr = input_distr
        compute_input_PDF = lambda x:input_distr.computePDF(x)
        if input_distr.getName()=="banana":
            aux_distrs = []
        elif input_distr.getName()=="SquareF":
            aux_distrs = []
        else:
            aux_distrs = [input_distr]
        
    X = init_distr.getSample(Nk)
    Y = np.array(func(X))
    
    f_X = np.array(compute_input_PDF(X))
    # print("")
    # print(f_X)
    h_X = np.array(init_distr.computePDF(X))#.flatten()
    
    # print("")
    # print(h_X)
    W = f_X/h_X
    
    hat_I = np.mean(Y*W,axis=0)
    curr_var = np.var(Y*W,axis=0,ddof=1)
                
    k=1
    N_eval = Nk
    Nks = [Nk]
                
    #initialisation of alpha and beta
    alpha_star = np.sqrt(weights)*hat_I/np.sum(np.sqrt(weights)*hat_I)
    beta_star = hat_I
            
    g_alphas = [input_distr]
    
    #iterations
    while N_eval<N1:
        
        #W = W.flatten()       
        curr_mixture = []
        WY_ce = W*Y
        
        #I = ot.Interval(2*[0,3/0.6,0,0,0,0,0,0,0,0],2*[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,1,np.inf,np.inf,1])
        I = ot.Interval(2*[0,0,0,0,0],2*[np.inf,np.inf,np.inf,np.inf,1])
        
        #cross entropy
        if cross_entropy=="SG":
            for j in range(nb_esp):
                #WY_ce = (W*Y[:,j]).reshape((-1,1))
                mu_hat = np.mean(WY_ce[:,j].reshape((-1,1))*np.array(X),axis=0) / np.mean(WY_ce[:,j])
                                
                Xtmp = np.array(X) - mu_hat
                Xo = np.sqrt(WY_ce[:,j]).reshape((-1,1)) * (Xtmp)
                if diag==True:
                    sigma_hat = np.zeros((dim,dim))
                    for n in range(dim):
                        sigma_hat[n,n] = np.sum(Xo[:,n]**2) / np.sum(WY_ce[:,j]) + 1e-6
                else:
                    sigma_hat = np.matmul(Xo.T, Xo) / np.sum(WY_ce[:,j]) + 1e-6 * np.eye(dim)
                                
                ot_mu = ot.Point(mu_hat)
                ot_sigma = ot.CovarianceMatrix(sigma_hat)
                #dq = ot.Normal(ot_mu,ot_sigma)
                #trun = ot.Distribution(ot.TruncatedDistribution(dq,I))
                curr_mixture.append(ot.Normal(ot_mu,ot_sigma))
                #curr_mixture.append(trun)
                            
        elif cross_entropy=="GM":
            """nb_gaussian_max = 10
            inertia = np.zeros(nb_gaussian_max)
            for nb_gaussian in range(1,nb_gaussian_max+1):
                Km = KMeans(n_clusters=nb_gaussian)
                Km.fit(np.array(X),sample_weight=W)
                current_inertia = Km.inertia_
                inertia[nb_gaussian-1] = current_inertia
            
            best_k = KneeLocator(np.arange(1,len(inertia)+1), inertia, curve='convex', direction='decreasing').knee
            print(best_k)
            
            plt.plot(range(1,11),inertia,label=str(k))"""            
            
            for j in range(nb_esp):
                nb_mix=2
                
                #WY_ce = (W*Y[:,j]).reshape((-1,1))
                [mu_hat, sigma_hat, pi_hat] = EMGM(np.array(X).T, WY_ce[:,j], nb_mix,diag)
                mu_hat = mu_hat.T
                nb_mix = len(pi_hat)
                                
                collDist = [ot.Normal(ot.Point(mu_hat[i]),ot.CovarianceMatrix(sigma_hat[:,:,i])) for i in range(mu_hat.shape[0])]
                curr_mixture.append(ot.Mixture(collDist,pi_hat))
            
        #optimisation in alpha
        def curr_mixture_PDF(x):
            return [d.computePDF(x) for d in curr_mixture]
        
        ot_curr_mixture_PDF = ot.PythonFunction(dim,nb_esp,curr_mixture_PDF)
        g_curr_distr = np.array(ot_curr_mixture_PDF(X))
        alpha_star = get_alpha_star(alpha_star, Y, weights, f_X, h_X, g_curr_distr, beta_star)
        g_alpha_star = ot.Mixture(curr_mixture,alpha_star)
                        
        #draw samples from the new distribution
        Xg = g_alpha_star.getSample(Nk)
        Yg = np.array(func(Xg))
        # print(len(np.unique(np.where(Yg==0)[0])))
        f_Xg = np.array(compute_input_PDF(Xg))
        #f_Xg = np.array(input_distr.computePDF(Xg)).flatten()
        
        X.add(Xg)
        Y = np.vstack((Y,Yg))
        f_X = np.vstack((f_X,f_Xg))
        
        g_curr_distr = np.array(ot_curr_mixture_PDF(X))
        
        #create new mixture, update of h
        aux_distrs.append(g_alpha_star)
        Nks.append(Nk)
        
        if type(input_distr)==list:
            h = ot.Mixture(aux_distrs,Nks)
            h_X = np.array(h.computePDF(X))
        else:
            if input_distr.getName() == "banana":
                h = input_distr.pdf_mixture(aux_distrs,Nks)
                h_X = np.array(h(X))
            elif input_distr.getName() == "SquareF":
                h = input_distr.pdf_mixture(aux_distrs,Nks)
                h_X = np.array(h(X))
            else:
                h = ot.Mixture(aux_distrs,Nks)
                h_X = np.array(h.computePDF(X))
        
        #.flatten()
                
        #computation of beta star
        g_X = np.array(g_alpha_star.computePDF(Xg))
        g_curr_distr_g = np.array(ot_curr_mixture_PDF(Xg))
        
        Wg = f_Xg / g_X
        #Wg = Wg.reshape((-1,1))
        bvc = g_curr_distr_g/g_X#.reshape((-1,1))
        fdg = Wg*Yg
        # fdg = Wg.reshape((-1,1))*Yg
       
        
        W = f_X / h_X
        
        cov = np.zeros(nb_esp)
        for j in range(nb_esp):
            m1,m2 = np.mean(bvc[:,j]),np.mean(fdg[:,j])
            cov[j] = np.mean(bvc[:,j]*fdg[:,j]) - m1*m2
        var = np.var(bvc,axis=0,ddof=1)
                    
        beta_star = cov/var
        
        N_eval += Nk
        k+=1
        
        values = (f_Xg*Yg - beta_star*g_curr_distr_g)/g_X.reshape((-1,1))
        new_var = np.var(values,axis=0,ddof=1)
                
        if np.sum(weights*curr_var)/(N_max-(N_eval-Nk))<=np.sum(weights*new_var)/(N_max-N_eval):
            g_alphas.append(g_alpha_star)
            #print(k,N_eval)
            break
           
        else:
            curr_var = new_var
            g_alphas.append(g_alpha_star)
                     
    X_final = g_alpha_star.getSample(N_max-N_eval)
    Y_final = np.array(func(X_final))
    f_X_final = np.array(compute_input_PDF(X_final))#np.array(input_distr.computePDF(X_final)).flatten()
    g_X_final = np.array(g_alpha_star.computePDF(X_final))
    g_curr_distr_final = np.array(ot_curr_mixture_PDF(X_final))
    
    W_final = f_X_final/g_X_final
    W_final = W_final
    
    values_final = (f_X_final*Y_final - beta_star*g_curr_distr_final)/g_X_final.reshape((-1,1))
    hat_I_final = np.mean(values_final,axis=0) + beta_star
            
    return hat_I_final,X_final,W_final,g_alphas


# #%%

# import itertools
# def powerset(d):
#     "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
#     s = list(range(d))
#     return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(d+1))


# dim = 2
# J = 4

# cov = np.eye(dim)
# cov = ot.CovarianceMatrix(cov)
# distr = ot.Normal(ot.Point(dim),cov)
# subsets = list(powerset(dim))
# i_u = np.random.choice(range(2**dim),replace=False,size=J)

# means = 1*np.ones((J,dim))
# for i in range(J):
#     sub = np.array(subsets[i_u[i]])
#     if len(sub)>0:
#         means[i][sub] = -1
        
# def f(x):
#     a = ot.CovarianceMatrix(dim)
#     return [10**4*ot.Normal(ot.Point(means[i]),a).computePDF(x) for i in range(J)]

# func = ot.PythonFunction(dim,J,f)


# def f2(x):
#     a = ot.CovarianceMatrix(dim)
#     return [10**4*ot.Normal(ot.Point(means[0]),a).computePDF(x)]

# func2 = ot.PythonFunction(dim,1,f2)
# d1 = ot.Normal(dim)
# cov = 2*np.eye(dim)
# cov = ot.CovarianceMatrix(cov)
# d2 = ot.Normal(ot.Point(dim),cov)
# cov = .5*np.eye(dim)
# cov = ot.CovarianceMatrix(cov)
# d3 = ot.Normal(ot.Point(dim),cov)
# cov = .75*np.eye(dim)
# cov = ot.CovarianceMatrix(cov)
# d4 = ot.Normal(ot.Point(dim),cov)



# a,_,_ = multiple_expectations_cv(func,np.ones(J),[d1,d2,d3,d4],N_max=10**4,cross_entropy="SG")
# print(a)