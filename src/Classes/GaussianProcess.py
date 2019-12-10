#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:22:42 2019

@author: hannes
"""

#General imports
import numpy as np
import GPy

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GaussianProcess:
    def __init__(self):
        self.kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.,ARD=1)
#        self.kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
    
    def initializeGP(self, robot):
        print('Initialize GP')
        r,c = np.where(robot.mapping)
        
        y = robot.mapping[r,c]
        y = y.reshape(-1,1)
        
        x = np.dstack((r,c))
        x = x.reshape(-1,2)

        self.model = GPy.models.GPRegression(x,y, self.kernel)
#        self.model = GPy.models.SparseGPRegression(x,y, kernel=self.kernel, num_inducing=10)
        self.model.optimize(messages=False,max_f_eval = 100,ipython_notebook=False)

        
    def updateGP(self, robot):
        print('Updating GP')
        print('Time: %.1f' %robot.totalTime)
        r,c = np.where(robot.mapping > 0)
        
        y = robot.mapping[r,c]
        y = y.reshape(-1,1)
        
        x = np.dstack((r,c))
        x = x.reshape(-1,2)
        print(y.shape)
        self.model.set_XY(x,y)
        
        self.model.optimize(messages=False,max_f_eval = 100,ipython_notebook=False)
        
    def inferGP(self, robot):
        X, Y = np.mgrid[0:robot.discretization[0]:1, 0:robot.discretization[1]:1]
        z = np.dstack((np.ravel(X),np.ravel(Y)))
        z = z.reshape(-1,2)

        ym, ys = self.model.predict(z)         # predict test cases
        robot.expectedMeasurement = ym.reshape(robot.discretization)
        return ym,ys
    
    def plotInferGP(self, robot):
        X, Y = np.mgrid[0:robot.discretization[0]:1, 0:robot.discretization[1]:1]

        ym, ys = self.inferGP(robot)

        plt.figure()
#        plt.contourf(Y, X, ym.reshape(robot.discretization))
#        plt.colorbar()
#        
        plt.imshow(ym.reshape(robot.discretization), origin='lower');
        plt.colorbar()
    
        plt.show()
        
    def demo(self):
        
        # sample inputs and outputs
        X = np.random.uniform(-3.,3.,(50,2))
        Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(50,1)*0.05
        
        # define kernel
        ker = GPy.kern.Matern52(2,ARD=True) + GPy.kern.White(2)
        
        # create simple GP model
        m = GPy.models.GPRegression(X,Y,ker)
        
        # optimize and plot
        m.optimize(messages=False,max_f_eval = 1000)

        m.plot()
        
    def demoGPy(self):
        mean = [0, 0]
        cov = [[1, 0], [0, 100]]  # diagonal covariance
        scale = 1 #0.001
        
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
        y = y.reshape(100,1)
        kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
        m = GPy.models.GPRegression(posMeas,y, kernel)
        m.optimize(messages=False,max_f_eval = 1000,ipython_notebook=False)        
        m.plot()
        
        z = np.dstack((np.ravel(X),np.ravel(Y)))
        z = z.reshape(10000,2)

        ym, ys = m.predict(z)         # predict test cases

        plt.figure()
        plt.contourf(X, Y, ym.reshape(100,100))
        plt.colorbar()

        
        
        
        