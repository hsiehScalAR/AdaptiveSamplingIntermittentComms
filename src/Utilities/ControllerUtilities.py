#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:54:18 2019

@author: hannes
"""

#General imports
import numpy as np

def measurement(numRobots, sensorPeriod): 
    #TODO check how we measure stuff, if single value since each robot measure one place or measurement over time for all robots
    """Simulates a measurement for all robots over time and for a single robot at one time instance"""
    # Input Arguments
    # numRobots = how many robots
    # sensorPeriod = period of sensing in ms
    
    allMeasurementsOverTime = np.random.uniform(0,1,(numRobots, sensorPeriod))
    singleMeasurement = np.random.uniform(0,1)
    return allMeasurementsOverTime, singleMeasurement