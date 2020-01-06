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

# Personal imports
from Utilities.VisualizationUtilities import plotMeasurement
                                              
ITERATIONS = 1000

class GaussianProcess:
    def __init__(self, spatiotemporal):
        """Initialize kernel for the GPs"""
        # Input arguments:
        # spatiotemporal = bool if using time dependent kernel
        
        self.spatiotemporal = spatiotemporal
        
        if spatiotemporal:
            #TODO: lengthscale for time is very important!
            # 1: very quick decay of estimate
            # 2: maybe ok
            # 10: good spatial but not predictive in time
            # ARD True: good spatial but not predictive in time
            # ARD False: very quick decay of estimate

            self.kernel = (GPy.kern.RBF(input_dim=2, variance=1., lengthscale=10., active_dims=[0,1],ARD=True) 
                           * GPy.kern.RBF(input_dim=1, variance=0.5, lengthscale=10., active_dims=[2], ARD=False))
            # self.kernel = GPy.kern.RBF(input_dim=3, variance=1., lengthscale=[1.,1.,1.],ARD=True,useGPU=False)
        else:
            self.kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
    
    def initializeGP(self, robot):
        """Initialize model for the GPs"""
        # Input arguments:
        # robot = robot whose GP is to be initialized
        
        r,c = np.where(robot.mapping[:,:,0] != 0)
        
        y = robot.mapping[r,c,0]
        y = y.reshape(-1,1)
        if self.spatiotemporal:
            t = robot.mapping[r,c,1]
            x = np.dstack((r,c,t))
            x = x.reshape(-1,3)
        else:
            x = np.dstack((r,c))
            x = x.reshape(-1,2)
        
        self.model = GPy.models.GPRegression(x,y, self.kernel)                                 # Works good
        
        self.model.optimize(optimizer='lbfgsb',messages=False,max_f_eval = ITERATIONS,ipython_notebook=False)    # Works good

        
    def updateGP(self, robot):
        """Update function for GPs, adds new measurements to model"""
        # Input arguments:
        # robot = robot whose GP is to be updated
        
        print('Updating GP for robot %d' %robot.ID)
        print('Time: %.1f' %robot.currentTime)
        r,c = np.where(robot.mapping[:,:,0] > 0.05)
        
        y = robot.mapping[r,c,0]
        y = y.reshape(-1,1)

        if self.spatiotemporal:
            t = robot.mapping[r,c,1]
            x = np.dstack((r,c,t))
            x = x.reshape(-1,3)
        else:
            x = np.dstack((r,c))
            x = x.reshape(-1,2)

        print(y.shape)
        if y.shape[0] == 0:
            return

        self.model.set_XY(x,y)
        
#        self.model.optimize(optimizer= 'scg',messages=False,max_iters = ITERATIONS,ipython_notebook=False)       # Don't see difference, maybe slower       
        self.model.optimize(optimizer='lbfgsb',messages=False,max_f_eval = ITERATIONS,ipython_notebook=False)    # Works good
        print('GP Updated\n')

        
    def inferGP(self, robot, pos=None, time=None):
        """Calculates estimated measurement at location"""
        # Input arguments:
        # robot = robot whose GP should calculate estimate
        # pos = if single position, else whole grid is calculated
        
        if isinstance(pos,np.ndarray):
            if self.spatiotemporal:
                z = np.dstack((pos[0], pos[1], robot.currentTime))
                z = z.reshape(-1,3)
            else:
                z = np.dstack((pos[0], pos[1]))
                z = z.reshape(-1,2)
            ym, ys = self.model.predict(z)
            return ym, ys
        else:
            X, Y = np.mgrid[0:robot.discretization[0]:1, 0:robot.discretization[1]:1]
            if self.spatiotemporal:
                if time == None:
                    T = np.ones(len(np.ravel(X)))*robot.currentTime
                else:
                    T = np.ones(len(np.ravel(X)))*time
                z = np.dstack((np.ravel(X),np.ravel(Y),np.ravel(T)))
                z = z.reshape(-1,3)
            else:
                z = np.dstack((np.ravel(X),np.ravel(Y)))
                z = z.reshape(-1,2)
                
        print('Inferring GP')
        ym, ys = self.model.predict(z)

        robot.expectedMeasurement = ym.reshape(robot.discretization)
        robot.expectedVariance = ys.reshape(robot.discretization)
        print('GP inferred\n')
        
        if robot.ID >= 0:
            fig, ax = plt.subplots(1,3,figsize=(18, 6))
            fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
            if time == None:
                dispTime = robot.currentTime
            else:
                dispTime = time

            title = 'Robot %d, Time %.1f' %(robot.ID,dispTime)
            fig.suptitle(title)

            ax[0].set_title('Expected Measurement')  
            im = ax[0].imshow(robot.expectedMeasurement, origin='lower')
            fig.colorbar(im,ax=ax[0])

            ax[1].set_title('Expected Variance')  
            im = ax[1].imshow(robot.expectedVariance, origin='lower')        
            fig.colorbar(im,ax=ax[1])

            ax[2].set_title('GroundTruth') 
            im = ax[2].imshow(robot.mappingGroundTruth, origin='lower')
            fig.colorbar(im,ax=ax[2])
            fig.savefig('Results/Figures/Tmp/' + title + '.png')
        
    def plotGP(self, robot, time=None):
        """Plotting model of the GP"""
        # Input arguments:
        # robot = robot whose GP is to be plotted

        # self.inferGP(robot)
        self.inferGP(robot, time=time)
    
        # self.model.plot(title='Robot %d, End' %(robot.ID))

        # slices = [100, 200, 300, 400, 500]
        # figure = GPy.plotting.plotting_library().figure(len(slices), 1)
        # for i, y in zip(range(len(slices)), slices):
        #     self.model.plot(figure=figure, fixed_inputs=[(0,y)], row=(i+1), plot_data=False)
        
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

        ym, _ = m.predict(z)         # predict test cases

        plt.figure()
        plt.contourf(X, Y, ym.reshape(100,100))
        plt.colorbar()
       
        
        
        