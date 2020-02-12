#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 6 09:53:55 2020

@author: hannes
"""

#General imports
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import compare_ssim as ssim
from scipy.spatial import procrustes

# Personal imports
# from Utilities.VisualizationUtilities import plotMeasurement, plotProcrustes

PATH = 'Results/Tmp/'
ENERGY = 0.99

class ReducedOrderModel:
    def __init__(self, spatiotemporal, logFile):
        """Initialize parameters for the PODs
        
        Input arguments:
        spatiotemporal = bool if using time dependent kernel
        specialKernel = bool if own kernel should be used
        logFile = logFile class which allows to write to file
        """

        self.spatiotemporal = spatiotemporal
        # self.specialKernel = specialKernel
        self.logFile = logFile
        self.filterThreshold = 0.0 # was 0.05
        self.timeFilter = 120 # was 50

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

    
    def initialize(self, robot):
        """Initialize model for the PODs
        
        Input arguments:
        robot = robot whose POD is to be initialized
        """

        # y = np.where(robot.mapping[:,:,0] != 0, robot.mapping[:,:,:],0)
        y = robot.mapping
        # y = robot.mapping[r,c,0]
        # y = y.reshape(-1,1)
        # if self.spatiotemporal:
        #     t = robot.mapping[r,c,1]
        #     x = np.dstack((r,c,t))
        #     x = x.reshape(-1,3)
        # else:
        #     x = np.dstack((r,c))
        #     x = x.reshape(-1,2)

        
        # TODO: Make first basis calculation and maybe give set of regions
        
        # y = y.reshape((-1,1))
        print(y.shape)
        self.calculateBasis(y, robot.numbMeasurements, robot.sensorPeriod)

        
    def update(self, robot):
        """Update function for PODs, adds new measurements to model
        
        Input arguments:
        robot = robot whose POD is to be updated
        """
        
        print('Updating POD for robot %d' %robot.ID)
        print('Time: %.1f' %robot.currentTime)
        if self.spatiotemporal:
            if self.timeFilter != None:
                y = robot.mapping
                # y = np.where((robot.mapping[:,:,0] > self.filterThreshold) & (robot.mapping[:,:,1] > (robot.currentTime-self.timeFilter)),robot.mapping[:,:,:],0) # was 0.05
        else:
            y = robot.mapping
            # y = np.where(robot.mapping[:,:,0] > self.filterThreshold, robot.mapping[:,:,:],0)
        
        # y = robot.mapping[r,c,0]
        # y = y.reshape(-1,1)

        # if self.spatiotemporal:
        #     t = robot.mapping[r,c,1]
        #     x = np.dstack((r,c,t))
        #     x = x.reshape(-1,3)
        # else:
        #     x = np.dstack((r,c))
        #     x = x.reshape(-1,2)

        print(y.shape)
        if y.shape[0] == 0:
            return


        # TODO: Make new basis calculation and maybe give set of regions     
        self.calculateBasis(y, robot.numbMeasurements, robot.sensorPeriod)

        # self.model.set_XY(x,y)
        # self.model.optimize(optimizer='lbfgsb',messages=False,max_f_eval = ITERATIONS,ipython_notebook=False)    # Works good
        print('POD Updated\n')

        
    def infer(self, robot, pos=None, time=None):
        """Calculates estimated measurement at location
        
        Input arguments:
        robot = robot whose GP should calculate estimate
        pos = if single position, else whole grid is calculated
        time = if we do it for specific future inference time
        """
        
        if isinstance(pos,np.ndarray):
            
            # TODO: Calculate mean and if possible expected variance
            ym = self.phiReduced @ self.timeDepCoeff
            ys = np.zeros_like(ym)

            return ym[pos,0], ys[pos,1]
        else:
            ym = self.phiReduced @ self.timeDepCoeff
            
            ys = np.zeros_like(ym)
                
        print('Inferring POD')
        # TODO: Calculate mean and if possible expected variance

        robot.expectedMeasurement = ym.reshape([600,600,2])[:,:,0]

        scaling = 1
        # robot.expectedVariance = ys.reshape([600,600,2])[:,:,1]

        index = np.where(robot.mapping[:,:,1] != 0, 14 - 14*robot.mapping[:,:,1]/robot.currentTime,14)
        robot.expectedVariance = index
        print('POD inferred\n')
        
        dissimilarity = self.errorCalculation(robot)

        if robot.ID >= 0:
            fig, ax = plt.subplots(1,3,figsize=(18, 6))
            fig.subplots_adjust(left=0.02, bottom=0.06, right=0.8, top=0.94, wspace=0.12,hspace=0.1)
            if time == None:
                dispTime = robot.currentTime
            else:
                dispTime = time

            title = 'Robot %d, Time %.1f, Dissimilarity: %.2f' %(robot.ID,dispTime,dissimilarity)
            fig.suptitle(title)

            ax[0].set_title('Expected Measurement')  
            im = ax[0].imshow(robot.expectedMeasurement, origin='lower', vmin=-1, vmax=15*scaling)

            ax[1].set_title('Trajectory Uncertainty')  
            im = ax[1].imshow(robot.expectedVariance, origin='lower', vmin=-1, vmax=15*scaling)        

            ax[2].set_title('Ground Truth') 

            x,y = zip(*robot.trajectory)
            ax[2].plot(y,x, '-', label='Robot %d'%robot.ID)
            ax[2].legend()

            im = ax[2].imshow(robot.mappingGroundTruth, origin='lower', vmin=-1, vmax=15*scaling)
            cbar_ax = fig.add_axes([0.83, 0.1, 0.01, 0.8])
            fig.colorbar(im, cax=cbar_ax)
            im.set_clim(-1, 15*scaling)
            fig.savefig(PATH + title + '.png')
            
            plt.close(fig)
            

    def plot(self, robot, time=None):
        """Plotting model of the POD
        
        Input arguments:
        robot = robot whose POD is to be plotted
        time = if we do it for specific future inference time
        """
        
        self.infer(robot, time=time)

    def errorCalculation(self, robot):
        """Error calculation of POD, computes different errors and writes to file
        
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

        if np.max(robot.expectedMeasurement) == 0:
            return 1
        
        _,_,procru = procrustes(robot.mappingGroundTruth,robot.expectedMeasurement)
        self.logFile.writeError(robot.ID,procru,robot.currentTime, 'Dissim')

        # plotProcrustes(robot, mat1,mat2)
        return procru

    def calculateCovariance(self, measurements, numbMeasurements, sensingPeriod):
        """Calculation of the covariance of the time dependent data
        
        Input arguments:
        measurements = measurement data of robot
        numbMeasurements = how many measurements have been taken
        sensingPeriod = interval between measurements
        """

        sumXXT = np.zeros_like(measurements[:,:,0])
        tol = 1e-4
        for t in range(0, numbMeasurements):
            
            X = np.where((measurements[:,:,1] > t*sensingPeriod-tol) & (measurements[:,:,1] < t*sensingPeriod+tol), measurements[:,:,0],0)
            XXT = X @ X.T
            sumXXT = sumXXT + XXT

        self.covariance = 1/numbMeasurements * sumXXT

    def createReducedBasis(self, eigVectors, eigValues):
        """Calculation of the reduced basis
        
        Input arguments:
        eigVectors = sorted eigenvectors
        eigValues = sorted eigenvalues
        """

        cumSum = 0
        totalSum = sum(eigValues).real
        numBase = 0 

        for i in range(0, len(eigValues)):
            cumSum += eigValues[i].real
            percentageEnergy = cumSum/totalSum

            if percentageEnergy < ENERGY:
                numBase = i
            else:
                numBase = i
                break

        phi = eigVectors[:, 0:numBase+1]

        return phi.real
    
    
    def calculateTimeDepCoeff(self, y):
        """Calculation of the time dependent coefficients
        
        Input arguments:
        y = measurement data of robot
        """

        a = self.phiReduced.T @ self.phiReduced
        b = self.phiReduced.T @ y

        self.timeDepCoeff = np.linalg.inv(a) @ b


    def calculateBasis(self, measurements, numbMeasurements, sensingPeriod):
        """Calculation of the reduced basis and time dependent coefficients
        
        Input arguments:
        measurements = measurement data of robot
        numbMeasurements = how many measurements have been taken
        sensingPeriod = interval between measurements
        """

        self.calculateCovariance(measurements, numbMeasurements, sensingPeriod)

        eigValues, eigVectors = np.linalg.eig(self.covariance)

        eigVectors = eigVectors.real
        eigValues = eigValues.real
        I = np.argsort(eigValues)[::-1]

        eigValues = eigValues[I]
        eigVectors = eigVectors[:,I]

        self.phiReduced = self.createReducedBasis(eigVectors, eigValues)
        self.calculateTimeDepCoeff(measurements)

    def copy(self):
        """Copy of class
        
        No input arguments
        """

        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result


    