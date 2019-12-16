#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:54:18 2019

@author: hannes
"""

#General imports
import numpy as np

def checkMeetingLocation(positions, commRadius):
    """check if the robots are at the same location since robots from different teams could be waiting to communicate"""
    # Input arguments
    # positions = current positions of the robots
    # commRadius = communication radius of robots
    
    normDist = np.sqrt(np.sum((positions[0] - positions)**2, axis=1))

    if all(x <= commRadius for x in normDist):
        return True
    else:
        return False

def communicateToTeam(robots, GP=True):
    """communicate sensor measurements between robots of same team at meeting location"""
    # Input arguments
    # robots = robots of same team
    
    mapping = np.zeros(robots[0].discretization)
    
    for r in range(0, len(robots)):
        pixels = np.where(robots[r].mapping != 0, True,False)
        mapping[pixels] = robots[r].mapping[pixels]
    
    for r in range(0, len(robots)):
        robots[r].mapping = mapping
#        if GP:
#            robots[r].GP.updateGP(robots[r])
        
    # TODO: copy models to robots of same teams
    if GP:
        robots[0].GP.updateGP(robots[0])
        if robots[0].optPath:
            robots[0].GP.inferGP(robots[0])
        for r in range(1,len(robots)):
            robots[r].GP.model = robots[0].GP.model.copy()
            if robots[0].optPath:
                robots[r].expectedMeasurement = robots[0].expectedMeasurement
                robots[r].expectedVariance = robots[0].expectedVariance

def moveAlongPath(robot, deltaT):
    # TODO: Add different motion model
    
    """move the robot along the planned path and take measurements on the way"""
    # Input arguments
    # robot = which robot we are moving
    # deltaT = sensor time step so that we move and take a measurement at each deltaT
    
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
    
    singleMeasurement = measurement(robot)
    robot.createMap(singleMeasurement, robot.currentLocation)  # Create Map
    
    return False
    
def measurement(robot):
    """Simulates a measurement for a single robot at one time instance"""
    # Input arguments
    # robot = robot with currentlocation and ground truth measurement map
    
#    singleMeasurement = np.random.uniform(0,1)
    x = np.int(robot.currentLocation[0])
    y = np.int(robot.currentLocation[1])

    # Add noise
    sigma = 0.2
    mean = 0
    
    if robot.sensingRange < 1:
        newData = robot.mappingGroundTruth[x, y]
        newData = newData + sigma*np.random.randn() + mean
        
        return newData
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
        
    return newData

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