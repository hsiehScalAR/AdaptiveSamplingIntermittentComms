#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:54:18 2019

@author: hannes
"""

#General imports
import numpy as np

def checkMeetingLocation(positions, commRadius):
    """check if the robots are at the same location since robots from different teams could be waiting to communicate
    
    Input arguments:
    positions = current positions of the robots
    commRadius = communication radius of robots
    """

    normDist = np.sqrt(np.sum((positions[0] - positions)**2, axis=1))

    if all(x <= commRadius for x in normDist):
        return True
    else:
        return False

def communicateToTeam(robots, MODEL=True, POD=False):
    """communicate sensor measurements between robots of same team at meeting location
    
    Input arguments:
    robots = robots of same team
    MODEL = bool if we are using models
    POD = bool if we are using POD
    """

    mapping = np.zeros([robots[0].discretization[0],robots[0].discretization[0],2])
    
    for r in range(0, len(robots)):
        pixels = np.where(robots[r].mapping[:,:,1] > mapping[:,:,1], True,False)
        mapping[pixels] = robots[r].mapping[pixels]
    
    for r in range(0, len(robots)):
        robots[r].mapping = mapping
        
    if MODEL:
        robots[0].model.update(robots[0])
        if robots[0].optPath:
            robots[0].model.infer(robots[0])
        for r in range(1,len(robots)):
            if POD:
                robots[r].model = robots[0].model.copy()
            else:
                robots[r].model.model = robots[0].model.model.copy()
            
            if robots[0].optPath:
                robots[r].expectedMeasurement = robots[0].expectedMeasurement
                robots[r].expectedVariance = robots[0].expectedVariance
                
    return robots[0].expectedMeasurement

def moveAlongPath(robot, deltaT):
    """move the robot along the planned path and take measurements on the way

    Input arguments:
    robot = which robot we are moving
    deltaT = sensor time step so that we move and take a measurement at each deltaT
    """

    # TODO: Add different motion model

    robot.currentTime += deltaT
    
    if robot.pathCounter >= len(robot.paths[robot.scheduleCounter]) or robot.atEndLocation:
        robot.pathCounter = 0
        robot.atEndLocation = True
        robot.trajectory.append(robot.currentLocation)
        return True
    
    currentNode = robot.paths[robot.scheduleCounter][robot.pathCounter]
    goalPos = robot.totalGraph.nodes[currentNode]['pos']

    distance = goalPos - robot.currentLocation
    normDist = np.sqrt(np.sum((goalPos - robot.currentLocation)**2))
    
    travelTime = normDist/robot.uMax
    
    if travelTime <= deltaT:
        robot.currentLocation = goalPos
        robot.pathCounter += 1
    else:
        step = np.around(robot.uMax*deltaT*distance/normDist)
        robot.currentLocation = robot.currentLocation + step
    
    robot.trajectory.append(robot.currentLocation)
    
    meas, measTime = measurement(robot)
    robot.createMap(meas, measTime, robot.currentLocation)  # Create Map
    
    return False
    
def measurement(robot):
    """Simulates a measurement for a single robot at one time instance

    Input arguments:
    robot = robot with currentlocation and ground truth measurement map
    """
    
    x = np.int(robot.currentLocation[0])
    y = np.int(robot.currentLocation[1])

    # Add noise
    sigma = 0.2
    mean = 0

    robot.numbMeasurements += 1
    
    if robot.sensingRange < 1:
        newData = robot.mappingGroundTruth[x, y]
        newData = newData + sigma*np.random.randn() + mean
        
        return newData, robot.currentTime
    robot.measurementRangeX = np.array([robot.sensingRange, robot.sensingRange +1])
    robot.measurementRangeY = np.array([robot.sensingRange, robot.sensingRange +1])
    
    if x-robot.sensingRange <= 0:
        robot.measurementRangeX[0] = x
    elif x+robot.sensingRange >= robot.discretization[0]:
        robot.measurementRangeX[1] = robot.discretization[0] - x
    if y-robot.sensingRange <= 0:
        robot.measurementRangeY[0] = y
    elif y+robot.sensingRange >= robot.discretization[1]:
        robot.measurementRangeY[1] = robot.discretization[1] - y
        
    newData = robot.mappingGroundTruth[x-robot.measurementRangeX[0]:x+robot.measurementRangeX[1], 
                                       y-robot.measurementRangeY[0]:y+robot.measurementRangeY[1]]
    
    newData = newData + sigma*np.random.randn() + mean
        
    return newData, robot.currentTime