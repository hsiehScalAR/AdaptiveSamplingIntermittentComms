#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:22:42 2019

@author: hannes
"""

#General imports
import numpy as np
import pyGPs

class GaussianProcess:
    def __init__(self):
        self.model = pyGPs.GPR()

    
    def initializeGP(self, robot):
        x = robot.currentLocation
        y = robot.mapping
        self.model.getPosterior(x, y)
        
    def demo(self):
        demoData = np.load('Data/regression_data.npz')
        x = demoData['x']      # training data
        y = demoData['y']      # training target
        z = demoData['xstar']  # test data
        
        model = pyGPs.GPR()      # specify model (GP regression)
        model.getPosterior(x, y) # fit default model (mean zero & rbf kernel) with data
        model.optimize(x, y)     # optimize hyperparamters (default optimizer: single run minimize)
        model.predict(z)         # predict test cases
        model.plot()             # and plot result
        
        