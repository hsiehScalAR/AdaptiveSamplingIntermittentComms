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
        
    def demo1(self):
#        demoData = np.load('Data/regression_data.npz')
#        x = demoData['x']      # training data
#        y = demoData['y']      # training target
#        z = demoData['xstar']  # test data
        
        x = np.array([[4,2],[2,5],[5,5],[8,8]])
        y = np.array([5,2,4,3])
        z = np.array([[6,6],[7,3]])
        model = pyGPs.GPR()      # specify model (GP regression)
        model.getPosterior(x, y) # fit default model (mean zero & rbf kernel) with data
        model.optimize(x, y)     # optimize hyperparamters (default optimizer: single run minimize)
        model.predict(z)         # predict test cases
#        model.plot()             # and plot result
        

        ys22 = np.reshape(model.ys2,(model.ys2.shape[0],))
        X, Y = np.mgrid[0:10:1, 0:10:1]
        
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y
        plt.figure()
        rv = multivariate_normal(model.ym.reshape(2), ys22).pdf(pos)


        plt.contour(X, Y, rv)
        plt.colorbar()
        
    def demo(self):
        mean = [0, 0]
        cov = [[1, 0], [0, 100]]  # diagonal covariance
        
        x, y = np.random.multivariate_normal(mean, cov, 10).T
        
        xx, yy = np.meshgrid(x,y)
        pos = np.dstack((xx,yy))
        
        rv = multivariate_normal(mean, cov)
        pdf = rv.pdf(pos)

        X, Y = np.mgrid[-5:5:0.1, -5:5:0.1]
        pos = np.dstack((X,Y))
        plt.figure()
        plt.contourf(X, Y, rv.pdf(pos))
        plt.colorbar()
        plt.show()


        posMeas = np.dstack((x,y))
        posMeas = posMeas.reshape(10,2)
        
        y = pdf[0][:]

        model = pyGPs.GPR()      # specify model (GP regression)
        model.getPosterior(posMeas, y) # fit default model (mean zero & rbf kernel) with data
        model.optimize(posMeas, y)     # optimize hyperparamters (default optimizer: single run minimize)
        
        
        z = np.dstack((np.ravel(X),np.ravel(Y)))
        z = z.reshape(10000,2)

        model.predict(z)         # predict test cases
#        model.plot()             # and plot result
        
        
        
        plt.figure()
        plt.contourf(X, Y, model.ym.reshape(100,100))
        plt.colorbar()

        
        
        
        