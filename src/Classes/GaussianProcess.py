#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:22:42 2019

@author: hannes
"""

#General imports
import numpy as np
import GPy

# import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from skimage.metrics import structural_similarity as ssim
from scipy.spatial import procrustes
                                              
ITERATIONS = 1000

class GaussianProcess:
    def __init__(self, spatiotemporal, specialKernel,logFile, path):
        """Initialize kernel for the GPs

        Input arguments:
        spatiotemporal = bool if using time dependent kernel
        specialKernel = bool if own kernel should be used
        logFile = logFile class which allows to write to file
        path = savePath
        """


        # Configure GPy -> matplotlib -> Agg
        # GPy.plotting.change_plotting_library("matplotlib")

        self.spatiotemporal = spatiotemporal
        self.specialKernel = specialKernel
        self.logFile = logFile
        if self.specialKernel:
            self.filterThreshold = 0.2 # was 0.2
            self.timeFilter = 40 # was 50

            spatialLengthScale = 20.
            tempLengthScale = 2.  
            spatialVariance = 4. 
            tempVariance = 2.
        else:
            self.filterThreshold = 0.05 # was 0.05
            self.timeFilter = 40 

            spatialLengthScale = 20.
            tempLengthScale = 2.  
            spatialVariance = 4. 
            tempVariance = 2.

        spatialARD = True
        tempARD = False


        """Write parameters to logfile"""
        parameters = {
                    'timeFilter     ': self.timeFilter,
                    'filterThreshold': self.filterThreshold,
                    'spatialLenScale': spatialLengthScale,
                    'tempLenScale   ': tempLengthScale,
                    'spatialVariance': spatialVariance,
                    'tempVariance   ': tempVariance,
                    'spatialARD     ': spatialARD,
                    'tempARD        ': tempARD,
                    }
        self.logFile.writeParameters(**parameters)

        self.path = path

        if spatiotemporal:   
            if self.specialKernel:
                self.kernel = GPy.kern.SpatioTemporal()
            else:
                self.kernel = (GPy.kern.RBF(input_dim=2, variance=spatialVariance, lengthscale=spatialLengthScale, active_dims=[0,1],ARD=spatialARD) 
                            * GPy.kern.RBF(input_dim=1, variance=tempVariance, lengthscale=tempLengthScale, active_dims=[2], ARD=tempARD))
        else:
            self.kernel = GPy.kern.RBF(input_dim=2, variance=spatialVariance, lengthscale=spatialLengthScale,ARD=spatialARD)
    
    def initialize(self, robot):
        """Initialize model for the GPs

        Input arguments:
        robot = robot whose GP is to be initialized
        """

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
    
        self.model = GPy.models.GPRegression(x,y, self.kernel)     # Works good
        # self.model = GPy.models.GPLVM(y,input_dim=2,init='PCA',X=x,kernel=self.kernel)
        # self.model = GPy.models.SparseGPRegression(x,y,self.kernel, num_inducing=500)
        
        self.model.constrain_bounded(0.005,300) # was 0.5 to 300 
        self.model.Gaussian_noise.variance.unconstrain()

        if self.spatiotemporal and not self.specialKernel:             
            self.model.mul.rbf_1.lengthscale.constrain_bounded(20,300) 

        self.model.optimize(optimizer='lbfgsb',messages=False,max_f_eval = ITERATIONS,ipython_notebook=False)    # Works good

        
    def update(self, robot):
        """Update function for GPs, adds new measurements to model

        Input arguments:
        robot = robot whose GP is to be updated
        """

        print('Updating GP for robot %d' %robot.ID)
        print('Time: %.1f' %robot.currentTime)
        if self.spatiotemporal:
            if self.timeFilter != None:
                r,c = np.where((robot.mapping[:,:,0] > self.filterThreshold) & (robot.mapping[:,:,1] > (robot.currentTime-self.timeFilter))) # was 0.05
        else:
            r,c = np.where(robot.mapping[:,:,0] > self.filterThreshold) # was 0.05
        
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

        # for sparse gp
        # i = np.random.permutation(x.shape[0])[:min(500, x.shape[0])]
        # Z = x.view(np.ndarray)[i].copy()
        # self.model.set_Z(Z, trigger_update=True)
        # self.model.Z.unconstrain()

        self.model.optimize(optimizer='lbfgsb',messages=False,max_f_eval = ITERATIONS,ipython_notebook=False)    # Works good
        
        # fig, ax = plt.subplots()
        # self.model.plot(ax=ax)
        # ax.figure.savefig(self.path + 'Model' + '.png')
        # fig['dataplot'].savefig(self.path + 'Model' + '.png')
        print('GP Updated\n')

        
    def infer(self, robot, pos=None, time=None):
        """Calculates estimated measurement at location

        Input arguments:
        robot = robot whose GP should calculate estimate
        pos = if single position, else whole grid is calculated
        time = if we do it for specific future inference time
        """

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

        scaling = 1
        robot.expectedVariance = ys.reshape(robot.discretization)
        print('GP inferred\n')
        
        dissimilarity = self.errorCalculation(robot)

        if self.specialKernel:
            if robot.ID >= 0:
                fig, ax = plt.subplots(1,3,figsize=(18, 6))
                if time == None:
                    dispTime = robot.currentTime
                else:
                    dispTime = time

                title = 'Robot %d, Time %.1f, Dissimilarity: %.2f' %(robot.ID,dispTime,dissimilarity)
                fig.suptitle(title)

                ax[0].set_title('Expected Measurement')  
                ax[0].set_xlabel('x')
                ax[0].set_ylabel('y') 
                im = ax[0].imshow(robot.expectedMeasurement, origin='lower')
                fig.colorbar(im, ax=ax[0])

                ax[1].set_title('Expected Variance')
                ax[1].set_xlabel('x')
                ax[1].set_ylabel('y')        
                im = ax[1].imshow(robot.expectedVariance, origin='lower') 
                fig.colorbar(im, ax=ax[1])

                ax[2].set_title('Ground Truth') 
                ax[0].set_xlabel('x')
                ax[0].set_ylabel('y') 
                x,y = zip(*robot.trajectory)
                ax[2].plot(y,x, '-', label='Robot %d'%robot.ID)
                ax[2].legend()

                im = ax[2].imshow(robot.mappingGroundTruth, origin='lower')
                fig.colorbar(im, ax=ax[2])

                fig.savefig(self.path + title + '.png')
                
                plt.close(fig)
        else:
            if robot.ID >= 0:
                fig, ax = plt.subplots(1,3,figsize=(18, 6))
                fig.subplots_adjust(bottom=0.06, right=0.8, top=0.94, wspace=0.2,hspace=0.1)
                if time == None:
                    dispTime = robot.currentTime
                else:
                    dispTime = time

                title = 'Robot %d, Time %.1f, Dissimilarity: %.2f' %(robot.ID,dispTime,dissimilarity)
                fig.suptitle(title)

                ax[0].set_title('Expected Measurement')  
                ax[0].set_xlabel('x')
                ax[0].set_ylabel('y') 
                im = ax[0].imshow(robot.expectedMeasurement, origin='lower', vmin=-1, vmax=15*scaling)

                ax[1].set_title('Expected Variance')
                ax[1].set_xlabel('x')
                ax[1].set_ylabel('y')   
                im = ax[1].imshow(robot.expectedVariance, origin='lower', vmin=-1, vmax=15*scaling)        

                ax[2].set_title('Ground Truth') 
                ax[2].set_xlabel('x')
                ax[2].set_ylabel('y') 
                x,y = zip(*robot.trajectory)
                ax[2].plot(y,x, '-', label='Robot %d'%robot.ID)
                ax[2].legend()

                im = ax[2].imshow(robot.mappingGroundTruth, origin='lower', vmin=-1, vmax=15*scaling)
                cbar_ax = fig.add_axes([0.83, 0.2, 0.01, 0.6])
                fig.colorbar(im, cax=cbar_ax)
                im.set_clim(-1, 15*scaling)
                fig.savefig(self.path + title + '.png')
                
                plt.close(fig)
            

    def plot(self, robot, time=None):
        """Plotting model of the GP

        Input arguments:
        robot = robot whose GP is to be plotted
        time = if we do it for specific future inference time
        """

        self.infer(robot, time=time)
        print(self.model[''])
        # self.model.save_model(self.path + 'RobotModel_'+ str(robot.ID), compress=False, save_data=False)

    def errorCalculation(self, robot):
        """Error calculation of modelling, computes different errors and writes to file

        Input arguments:
        robots = instance of the robots
        logFile = where to save the output
        """

        #TODO: use nrmse next time or fnorm

        rmse = np.sqrt(np.square(robot.mappingGroundTruth - robot.expectedMeasurement).mean())
        self.logFile.writeError(robot.ID,rmse,robot.currentTime, 'RMSE')

        # nrmse = 100 * rmse/(np.max(robot.mappingGroundTruth)-np.min(robot.mappingGroundTruth))
        # self.logFile.writeError(robot.ID,nrmse,robot.currentTime, 'NRMSE')
        
        # rmse = np.sqrt(np.sum(np.square(robot.mappingGroundTruth - robot.expectedMeasurement)))
        # fnorm = rmse/(np.sqrt(np.sum(np.square(robot.mappingGroundTruth))))
        # self.logFile.writeError(robot.ID,fnorm,robot.currentTime, 'FNORM')

        similarity = ssim(robot.mappingGroundTruth,robot.expectedMeasurement, gaussian_weights=False)
        self.logFile.writeError(robot.ID,similarity,robot.currentTime, 'SSIM')

        _,_,procru = procrustes(robot.mappingGroundTruth,robot.expectedMeasurement)
        self.logFile.writeError(robot.ID,procru,robot.currentTime, 'Dissim')

        # plotProcrustes(robot, mat1,mat2, self.path)
        return procru
       
        
        
        
