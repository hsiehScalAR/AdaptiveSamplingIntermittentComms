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

ITERATIONS = 100

class GaussianProcess:
    def __init__(self):
        """Initialize kernel for the GPs"""
        # No input
        
        self.kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.,ARD=True,useGPU=False)
#        self.kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
    
    def initializeGP(self, robot):
        """Initialize model for the GPs"""
        # Input arguments:
        # robot = robot whose GP is to be initialized
        
        r,c = np.where(robot.mapping != 0)
        
        y = robot.mapping[r,c]
        y = y.reshape(-1,1)
        
        x = np.dstack((r,c))
        x = x.reshape(-1,2)
        
        self.model = GPy.models.GPRegression(x,y, self.kernel)                                 # Works good
#        self.model = GPy.models.BayesianGPLVM(y,2,X=x,kernel = self.kernel)                    # Error
#        self.model = GPy.models.SparseGPLVM(y,2,X=x,kernel = self.kernel, num_inducing=100)    # Don't know how to induce correctly
        
#        self.model = GPy.models.GPRegressionGrid(x, y, kernel=self.kernel)                     # Error       

#        self.model = GPy.models.SparseGPRegression(x,y, kernel=self.kernel, num_inducing=100)  # Don't know how to induce correctly
#        self.model.inference_method=GPy.inference.latent_function_inference.FITC()
        
#        self.model.optimize(optimizer= 'scg',messages=False,max_iters = ITERATIONS,ipython_notebook=False)       # Don't see difference, maybe slower
        self.model.optimize(optimizer='lbfgsb',messages=False,max_f_eval = ITERATIONS,ipython_notebook=False)    # Works good

        
    def updateGP(self, robot):
        """Update function for GPs, adds new measurements to model"""
        # Input arguments:
        # robot = robot whose GP is to be updated
        
        print('Updating GP for robot %d' %robot.ID)
        print('Time: %.1f' %robot.endTotalTime)
        r,c = np.where(robot.mapping > 0.05)
        
        y = robot.mapping[r,c]
        y = y.reshape(-1,1)
        
        x = np.dstack((r,c))
        x = x.reshape(-1,2)
        print(y.shape)
        if y.shape[0] == 0:
            return

        self.model.set_XY(x,y)
        
#        self.model.optimize(optimizer= 'scg',messages=False,max_iters = ITERATIONS,ipython_notebook=False)       # Don't see difference, maybe slower       
        self.model.optimize(optimizer='lbfgsb',messages=False,max_f_eval = ITERATIONS,ipython_notebook=False)    # Works good
        print('GP Updated\n')
        
    def inferGP(self, robot, pos=None):
        """Calculates estimated measurement at location"""
        # Input arguments:
        # robot = robot whose GP should calculate estimate
        # pos = if single position, else whole grid is calculated
        
        if isinstance(pos,np.ndarray):
            z = pos.reshape(-1,2)
            ym, ys = self.model.predict(z)
            return ym, ys
        else:
            X, Y = np.mgrid[0:robot.discretization[0]:1, 0:robot.discretization[1]:1]
            z = np.dstack((np.ravel(X),np.ravel(Y)))
            z = z.reshape(-1,2)
        print('Inferring GP')
        ym, ys = self.model.predict(z)

        robot.expectedMeasurement = ym.reshape(robot.discretization)
        robot.expectedVariance = ys.reshape(robot.discretization)
        print('GP inferred\n')
        
    def plotGP(self, robot):
        """Plotting model of the GP"""
        # Input arguments:
        # robot = robot whose GP is to be plotted
        
        self.inferGP(robot)
        ym = robot.expectedMeasurement

        fig, ax = plt.subplots()
        ax.set_title('Robot %d' %robot.ID)     
        plt.imshow(ym, origin='lower');        
        plt.colorbar()
        plt.show()
        
#        self.model.plot()
#
#        slices = [100, 200, 300, 400, 500]
#        figure = GPy.plotting.plotting_library().figure(len(slices), 1)
#        for i, y in zip(range(len(slices)), slices):
#            self.model.plot(figure=figure, fixed_inputs=[(0,y)], row=(i+1), plot_data=False)
        
    def demoGPy(self):
        """Demo function to demonstrate GPy library"""
        # No input
        
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
       
        
        
        