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
        self.kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.,ARD=True,useGPU=True)
#        self.kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
    
    def initializeGP(self, robot):
        r,c = np.where(robot.mapping)
        
        y = robot.mapping[r,c]
        y = y.reshape(-1,1)
        
        x = np.dstack((r,c))
        x = x.reshape(-1,2)
        
        self.model = GPy.models.GPRegression(x,y, self.kernel)
#        self.model = GPy.models.BayesianGPLVM(y,2,X=x,kernel = self.kernel)
#        self.model = GPy.models.SparseGPLVM(y,2,X=x,kernel = self.kernel)
#        self.model = GPy.models.GPRegressionGrid(x, y, kernel=self.kernel)
        
#        self.model = GPy.models.SparseGPRegression(x,y, kernel=self.kernel, num_inducing=10)
        self.model.optimize(messages=False,max_f_eval = 100,ipython_notebook=False)

        
    def updateGP(self, robot):
        
        print('Time: %.1f' %robot.endTotalTime)
        r,c = np.where(robot.mapping > 0)
        
        y = robot.mapping[r,c]
        y = y.reshape(-1,1)
        
        x = np.dstack((r,c))
        x = x.reshape(-1,2)
        print(y.shape)
        if y.shape[0] == 0:
            return
        self.model.set_XY(x,y)
        
        self.model.optimize(messages=False,max_f_eval = 100,ipython_notebook=False)
        print('Updated GP\n')
        
    def inferGP(self, robot):
        X, Y = np.mgrid[0:robot.discretization[0]:1, 0:robot.discretization[1]:1]
        z = np.dstack((np.ravel(X),np.ravel(Y)))
        z = z.reshape(-1,2)

        ym, ys = self.model.predict(z)

        robot.expectedMeasurement = ym.reshape(robot.discretization)
        robot.expectedVariance = ys.reshape(robot.discretization)
        return ym,ys
    
    def plotInferGP(self, robot):
#        X, Y = np.mgrid[0:robot.discretization[0]:1, 0:robot.discretization[1]:1]

#        ym, ys = self.inferGP(robot)
        ym = robot.expectedMeasurement
        plt.figure()        
#        plt.imshow(ym.reshape(robot.discretization), origin='lower');
        plt.imshow(ym, origin='lower');
        
        plt.colorbar()
    
        plt.show()
        
#        slices = [100, 200, 300, 400, 500]
#        figure = GPy.plotting.plotting_library().figure(len(slices), 1)
#        for i, y in zip(range(len(slices)), slices):
#            self.model.plot(figure=figure, fixed_inputs=[(0,y)], row=(i+1), plot_data=False)
        
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
#        m.plot()
        
        z = np.dstack((np.ravel(X),np.ravel(Y)))
        z = z.reshape(10000,2)

        ym, ys = m.predict(z)         # predict test cases

        plt.figure()
        plt.contourf(X, Y, ym.reshape(100,100))
        plt.colorbar()
       
        
        
        