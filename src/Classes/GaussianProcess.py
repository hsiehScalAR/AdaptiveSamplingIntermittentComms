#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:22:42 2019

@author: hannes
"""

#General imports
import numpy as np
import pyGPs
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GaussianProcess:
    def __init__(self):
        self.model = pyGPs.GPR()

    
    def initializeGP(self, robot):
        x = robot.currentLocation
        y = robot.mapping
        self.model.getPosterior(x, y)
        
    def demo(self):
        mean = [0, 0]
        cov = [[1, 0], [0, 100]]  # diagonal covariance
        scale = 0.001
        x, y = np.random.multivariate_normal(mean, cov, 100).T
        
        xx, yy = np.meshgrid(x,y)
        pos = np.dstack((xx,yy))
        
        rv = multivariate_normal(mean, cov)
        pdf = rv.pdf(pos)/scale

        X, Y = np.mgrid[-5:5:0.1, -5:5:0.1]
        pos = np.dstack((X,Y))
        plt.figure()
        plt.contourf(X, Y, rv.pdf(pos)/scale)
        plt.colorbar()
        plt.show()


        posMeas = np.dstack((x,y))
        posMeas = posMeas.reshape(100,2)
        
        y = pdf[0][:]
        mean = pyGPs.mean.Zero()
        cov = pyGPs.cov.RBF()
        model = pyGPs.GPR()      # specify model (GP regression)
        model.setPrior(mean=mean, kernel=cov)
        model.getPosterior(posMeas, y) # fit default model (mean zero & rbf kernel) with data
        model.optimize(posMeas, y)     # optimize hyperparamters (default optimizer: single run minimize)
        
        
        z = np.dstack((np.ravel(X),np.ravel(Y)))
        z = z.reshape(10000,2)

        model.predict(z)         # predict test cases

        plt.figure()
        plt.contourf(X, Y, model.ym.reshape(100,100))
        plt.colorbar()

        
        
        
        