# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:52:00 2022

@author: Julien Demange-Chryst
"""

#%% Modules

import numpy as np
import openturns as ot
from EMGM import EMGM
from scipy.optimize import minimize,LinearConstraint,Bounds

#%% Optimisation of alpha

def get_alpha_star(alpha_0,Y,weights,f_X,h_X,g_curr_distr,beta):
    
    nb_esp = len(alpha_0)
    
    def g_alpha(alpha):
        mixture_alpha = np.sum(alpha*g_curr_distr,axis=1)
        mixture_alpha[mixture_alpha==0] = np.inf
        return mixture_alpha.reshape((-1,1))
    
    numerator_terms = weights*(f_X*Y - beta*g_curr_distr)**2
    numerator = np.sum(numerator_terms,axis=1).reshape((-1,1))
    
    f = lambda alpha:np.mean(numerator/g_alpha(alpha)/h_X)
    
    def grad_f(alpha):
        denominator = g_alpha(alpha)**2*h_X
        a = numerator/denominator
        a = a*g_curr_distr
        return -np.mean(a,axis=0)
    
    constraint = np.ones(nb_esp)

    bounds = Bounds(9*1e-12, 1)
    eq_constraint = LinearConstraint(constraint.T, 1, 1)

    res = minimize(f,x0=alpha_0,method="SLSQP",jac=grad_f,bounds = bounds,constraints=eq_constraint)
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
        def get_PDFs(x):
            return [d.computePDF(x) for d in input_distr]
        compute_input_PDF = ot.PythonFunction(dim,nb_esp,get_PDFs)
        aux_distrs = [init_distr]
        
    else:
        nb_esp = func.getOutputDimension()
        init_distr = input_distr
        compute_input_PDF = lambda x:input_distr.computePDF(x)
        if input_distr.getName()=="SquareF":
            aux_distrs = []
        else:
            aux_distrs = [input_distr]
        
    X = init_distr.getSample(Nk)
    Y = np.array(func(X))
    
    f_X = np.array(compute_input_PDF(X))
    h_X = np.array(init_distr.computePDF(X))
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
        
        curr_mixture = []
        WY_ce = W*Y
        
        #cross entropy
        if cross_entropy=="SG":
            for j in range(nb_esp):
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
                curr_mixture.append(ot.Normal(ot_mu,ot_sigma))
                            
        elif cross_entropy=="GM":
            for j in range(nb_esp):
                nb_mix=2
                
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
        f_Xg = np.array(compute_input_PDF(Xg))
        
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
            if input_distr.getName() == "SquareF":
                h = input_distr.pdf_mixture(aux_distrs,Nks)
                h_X = np.array(h(X))
            else:
                h = ot.Mixture(aux_distrs,Nks)
                h_X = np.array(h.computePDF(X))
           
        W = f_X / h_X     
           
        #computation of beta star
        g_X = np.array(g_alpha_star.computePDF(Xg))
        g_curr_distr_g = np.array(ot_curr_mixture_PDF(Xg))
        
        Wg = f_Xg / g_X
        first_term_in_cov = g_curr_distr_g/g_X
        second_term_in_cov = Wg*Yg
       
        covariance_numerator = np.zeros(nb_esp)
        for j in range(nb_esp):
            m1,m2 = np.mean(first_term_in_cov[:,j]),np.mean(second_term_in_cov[:,j])
            covariance_numerator[j] = np.mean(first_term_in_cov[:,j]*second_term_in_cov[:,j]) - m1*m2
        variance_denominator = np.var(first_term_in_cov,axis=0,ddof=1)
                    
        beta_star = covariance_numerator/variance_denominator
        
        N_eval += Nk
        k+=1
        
        values = (f_Xg*Yg - beta_star*g_curr_distr_g)/g_X.reshape((-1,1))
        new_var = np.var(values,axis=0,ddof=1)
                
        if np.sum(weights*curr_var)/(N_max-(N_eval-Nk))<=np.sum(weights*new_var)/(N_max-N_eval):
            g_alphas.append(g_alpha_star)
            break
           
        else:
            curr_var = new_var
            g_alphas.append(g_alpha_star)
                     
    X_final = g_alpha_star.getSample(N_max-N_eval)
    Y_final = np.array(func(X_final))
    f_X_final = np.array(compute_input_PDF(X_final))
    g_X_final = np.array(g_alpha_star.computePDF(X_final))
    g_curr_distr_final = np.array(ot_curr_mixture_PDF(X_final))
    
    W_final = f_X_final/g_X_final
    W_final = W_final
    
    values_final = (f_X_final*Y_final - beta_star*g_curr_distr_final)/g_X_final.reshape((-1,1))
    hat_I_final = np.mean(values_final,axis=0) + beta_star
            
    return hat_I_final,X_final,W_final,g_alphas