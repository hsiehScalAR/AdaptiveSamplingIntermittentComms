#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:54:18 2019

@author: hannes
"""

#General imports
import numpy as np

def communicateToTeam():
    print('communicateToTeam')
    return False

def moveAlongPath(robot, deltaT, uMax):
    
    if robot.pathCounter >= len(robot.paths[robot.scheduleCounter]) or robot.atEndLocation:
        robot.pathCounter = 0
        robot.atEndLocation = True
        return True
    
    currentNode = robot.paths[robot.scheduleCounter][robot.pathCounter]
    goalPos = robot.totalGraph.nodes[currentNode]['pos']

    distance = goalPos - robot.currentLocation
    normDist = np.sqrt(np.sum((goalPos - robot.currentLocation)**2))
    
    travelTime = normDist/uMax
    
    if travelTime <= deltaT:
        robot.currentLocation = goalPos
        robot.pathCounter += 1
    else:
        step = np.around(uMax*deltaT*distance/normDist)
        robot.currentLocation = robot.currentLocation + step
        
    robot.trajectory.append(robot.currentLocation)
    
    singleMeasurement = measurement()
    robot.createMap(singleMeasurement, robot.currentLocation)  # Create Map
    
    return False
    
def measurement():
    #TODO check how we measure stuff, if single value since each robot measure one place or measurement over time for all robots
    """Simulates a measurement for a single robot at one time instance"""
    # No input
    
    singleMeasurement = np.random.uniform(0,1)

    return singleMeasurement

#def measurement(numRobots, sensorPeriod): 
#    #TODO check how we measure stuff, if single value since each robot measure one place or measurement over time for all robots
#    """Simulates a measurement for all robots over time and for a single robot at one time instance"""
#    # Input Arguments
#    # numRobots = how many robots
#    # sensorPeriod = period of sensing in ms
#    
#    allMeasurementsOverTime = np.random.uniform(0,1,(numRobots, sensorPeriod))
#    singleMeasurement = np.random.uniform(0,1)
#    return allMeasurementsOverTime, singleMeasurement