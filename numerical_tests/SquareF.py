# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:52:32 2022

@author: Julien Demange-Chryst
"""

#%% Modules

import openturns as ot
import numpy as np

#%% Class product distribution fxf

class SquareF:
    def __init__(self,input_distr):
        self.input_distr = input_distr
        self.dim = 2*input_distr.getDimension()
        self.name = "SquareF"
        self.pdf = self.ot_pdf()
        
    
    def getDimension(self):
        return self.dim
    
    def getName(self):
        return self.name
    
    def getSample(self,N):
        X = self.input_distr.getSample(N)
        Y = self.input_distr.getSample(N)
        X.stack(Y)
        return X

    def ot_pdf(self):
        def pdf(x):
            xp = np.array(x)
            x1,x2 = xp[:self.dim//2],xp[self.dim//2:]
            return [self.input_distr.computePDF(x1) * self.input_distr.computePDF(x2)]
        return ot.PythonFunction(self.dim,1,pdf)

    def computePDF(self,x):
        return self.pdf(x)
    
    def pdf_mixture(self,distrs,weights):
        N_tot = np.sum(weights)
        n_squ,n_mix = weights[0],weights[1:]
        ot_mixture = ot.Mixture(distrs,n_mix)
        
        def pdf(x):
            return n_squ/N_tot*self.computePDF(x) + (1-n_squ/N_tot)*ot_mixture.computePDF(x)
                
        return ot.PythonFunction(self.dim,1,pdf)
    
    def mixture_getSample(self,N,distrs,weights):
        nb_mix = len(weights)
        n_tot = sum(weights)
        norm_weights = [w/n_tot for w in weights]
        indices_mixture = np.random.choice(range(nb_mix),size=N,replace=True,p=norm_weights)
        sample_per_mix = np.bincount(indices_mixture)
                
        sample = self.getSample(int(sample_per_mix[0]))
        for i in range(1,nb_mix):
            sample.add(distrs[i-1].getSample(int(sample_per_mix[i])))
        
        return sample