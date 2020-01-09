#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:48:29 2019

@author: hannes
"""

#General imports
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx

#Personal imports
from Classes.Scheduler import Schedule
from Classes.Robot import Robot
from Setup import getSetup, setupMatlabFileMeasurementData, loadMeshFiles

from Utilities.ControllerUtilities import moveAlongPath, communicateToTeam, measurement
from Utilities.VisualizationUtilities import (plotMeasurement, plotMeetingGraphs, plotMeetingPaths, 
                                              clearPlots, plotTrajectory, plotTrajectoryAnimation,
                                              plotTrajectoryOverlayGroundTruth)
from Utilities.PathPlanningUtilities import (sampleVrand, findNearestNode, steer, buildSetVnear, 
                                             extendGraph, rewireGraph, getPath, 
                                             getInformationGainAlongPath, sampleNPoints,
                                             calculateGeometricCenter)
from Utilities.LogUtilities import LogFile

def main():
    """main test loop"""
    # no inputs 
    
    """Remove Tmp results file"""  
    for filename in os.listdir(FOLDER):
        file_path = os.path.join(FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    """Write parameters to new logfile"""
    parameters = {
                'Connectivity   ': 'all-time',  
                'TOTALTIME      ': TOTALTIME,
                'CASE           ': CASE,
                'CORRECTTIMESTEP': CORRECTTIMESTEP,
                'GAUSSIAN       ': GAUSSIAN,
                'OPTPATH        ': OPTPATH,
                'OPTPOINT       ': OPTPOINT,
                'SPATIOTEMPORAL ': SPATIOTEMPORAL,
                'STATIONARY     ': STATIONARY,
                'STATIONARYTIME ': STATIONARYTIME,
                'PREDICTIVETIME ': PREDICTIVETIME,
                'SENSINGRANGE   ': SENSINGRANGE,
                'COMMRANGE      ': COMMRANGE,
                'SENSORPERIOD   ': SENSORPERIOD
                }
    logFile = LogFile(LOGFILE,FOLDER +'/')
    logFile.writeParameters(**parameters)

    """Create Measurement Data"""
    measurementGroundTruthList, maxTime = loadMeshFiles(SENSORPERIOD,CORRECTTIMESTEP)

    if maxTime < TOTALTIME:
        print('**********************************************************************\n')
        print('WARNING: Given Total Time is bigger than available data (maxTime = %d), terminating application' %maxTime)
        print('**********************************************************************\n')
        exit()

    if STATIONARY:
        measurementGroundTruth = measurementGroundTruthList[np.int(STATIONARYTIME/SENSORPERIOD)]
    else:
        measurementGroundTruth = measurementGroundTruthList[0]

           
    """create robot to team correspondence"""
    numRobots, positions = getSetup(CASE, COMMRANGE)
    
    """Variables"""
    if isinstance(positions, np.ndarray):
        locations = positions
    else:
        locations = randomStartingPositions(numRobots) #locations or robots

    """Initialize robots"""
    robots = initializeRobots(numRobots, logFile)
    virtualRobot = Robot(numRobots, DISCRETIZATION, UMAX, SENSORPERIOD, OPTPATH, OPTPOINT, SPATIOTEMPORAL, logFile)
    
    """create the initial plans for all periods"""
    initialTime = 0
    
    print('Initializing Environment')
    
    geoCenter = calculateGeometricCenter(locations, numRobots)

    for r in range(0,numRobots):
        robots[r].vnew = locations[r]
        robots[r].currentLocation = locations[r]
        robots[r].totalTime = initialTime
        robots[r].sensingRange = SENSINGRANGE
        robots[r].mappingGroundTruth = measurementGroundTruth
        robots[r].distGeoCenter = geoCenter - locations[r]

    virtualRobot.vnew = geoCenter
    virtualRobot.currentLocation = geoCenter
    virtualRobot.totalTime = initialTime 
    virtualRobot.mappingGroundTruth = measurementGroundTruth
    virtualRobot.sensingRange = SENSINGRANGE
    """Initialize GP"""
    for r in range(0,numRobots):
        meas, measTime = measurement(robots[r])
        virtualRobot.createMap(meas, measTime, robots[r].currentLocation, robots[r])
    if GAUSSIAN:
        virtualRobot.GP.initializeGP(virtualRobot)
    
    robots.append(virtualRobot)

    print('Environment Initialized\n')
    
    print('Initializing Paths')

    updatePaths(robots)

    for r in range(0,numRobots+1):
        robots[r].composeGraphs() 

    print('Paths Initialized\n')    
    
    """Control loop"""
    
    print('Starting ControlLoop')
    currentTime = initialTime

    for t in range(0,np.int(TOTALTIME/SENSORPERIOD)):
        if not STATIONARY:
            for r in range(0,numRobots+1):
                robots[r].mappingGroundTruth = measurementGroundTruthList[t]
            
        currentTime = update(currentTime, robots)
        

    print('ControlLoop Finished\n')
    
    print('Starting Plotting')
    if DEBUG:
        plotMeasurement(measurementGroundTruth, 'Ground truth measurement map')
        plotTrajectory(robots[0:numRobots])
        totalMap = robots[-1].mapping[:,:,0]
        plotMeasurement(totalMap, 'Measurements of robots after communication events')
    if ANIMATION:
        plotTrajectoryAnimation(robots)
    
    if GAUSSIAN:

        plotTrajectoryOverlayGroundTruth(robots,0)

        robots[-1].GP.updateGP(robots[-1])
        robots[-1].GP.plotGP(robots[-1])
        if PREDICTIVETIME != None:
            
            if PREDICTIVETIME >= maxTime:
                predictiveTime = maxTime-SENSORPERIOD
            else:
                predictiveTime = np.int(PREDICTIVETIME/SENSORPERIOD)
            robots[-1].currentTime = predictiveTime*SENSORPERIOD
            robots[-1].mappingGroundTruth = measurementGroundTruthList[predictiveTime]
            robots[-1].GP.plotGP(robots[-1], robots[-1].currentTime)
     
    
    print('Plotting Finished\n')

    errorCalculation(robots, logFile)

def update(currentTime, robots):
    """Update procedure of intermittent communication"""
    # Input arguments:
    # currentTime = current time of the execution
    # robots = instances of the robots that are to be moved
    # teams = teams of the robots
    # commPeriod = how many schedules there are
    
    atEndPoint = np.zeros(len(robots)-1)
    robots[-1].currentTime += SENSORPERIOD
    for i, rob in enumerate(robots[0:len(robots)-1]):
        atEndPoint[i] = moveAlongPath(rob, robots[-1], SENSORPERIOD)

    currentTime += SENSORPERIOD
        
    if np.all(atEndPoint):                
        for r in range(0,len(robots)-1):   
            robots[r].meetings.append(robots[r].currentLocation)         
            robots[r].atEndLocation = False
            
            robots[r].endTotalTime  = currentTime
        robots[-1].meetings.append(robots[0].currentLocation)

        communicateToTeam(robots[-1], GAUSSIAN)
        
        print('Updating Paths')
        updatePaths(robots)
        print('Paths Updated\n')
        
        for rob in robots:
            rob.composeGraphs()

    return round(currentTime,1)

def updatePaths(robots):
    """Update procedure of intermittent communication"""
    # Input arguments:
    # robots = instances of the robots that are to be moved
    
    #add node v0 to list of nodes for each robot             
    
    #make last node equal to the first of new period
    for rob in robots:
        if rob.nodeCounter > 0:
            rob.nodeCounter -= 1

        rob.startNodeCounter = rob.nodeCounter
        rob.startLocation = rob.vnew
        rob.startTotalTime = rob.endTotalTime 
        rob.nearestNodeIdx = rob.endNodeCounter
        
        rob.initializeGraph()

        rob.addNode(firstTime = True)
            
    #sample new nodes and create path
    distribution = 'uniform'
    rangeSamples = DISCRETIZATION
    
    for sample in range(0,TOTALSAMPLES):
        if sample == RANDOMSAMPLESMAX-1:
            mean = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
            stdDev = 4*COMMRANGE*COMMRANGE*np.identity(DIMENSION)
            distribution = 'gaussian'
            rangeSamples = [mean,stdDev]
        
        if sample >= RANDOMSAMPLESMAX:
            vrand = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
                
        if distribution == 'uniform':
            #looking for either optimal position or path based on GP modeling
            if OPTPATH:              
                maxVariance = 0
                vrand = np.array([0, 0])
                nearNIdx = 0
                
                for _ in range(0,10):
                    point = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
                    #find nearest node to random sample
                    nearNIdx = findNearestNode(robots[-1].graph,point)
                    var = getInformationGainAlongPath(robots[-1], point, nearNIdx, EPSILON)
                    
                    if var >= maxVariance:
                        maxVariance = var
                        vrand = point                                                          
            elif OPTPOINT:
                vrand = sampleNPoints(robots[-1], DISCRETIZATION, rangeSamples, distribution)
            else: 
                vrand = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
                
            robots[-1].vrand = vrand
            nearestNodeIdx = findNearestNode(robots[-1].graph,vrand)
                    
            robots[-1].nearestNodeIdx = nearestNodeIdx
            
        else:
            robots[-1].vrand = vrand
            
            #find nearest node to random sample
            nearestNodeIdx = findNearestNode(robots[-1].graph,vrand)
            robots[-1].nearestNodeIdx = nearestNodeIdx
        
        #find new node towards max distance to random sample and incorporate time delay, that is why it is outside of previous loop since we need all the nearest nodes from the other robots
        steer(robots[-1], EPSILON, COMMRANGE)

        # get all nodes close to new node
        buildSetVnear(robots[-1], EPSILON, GAMMARRT)
        
        extendGraph(robots[-1])
        for r in range(0,len(robots)-1):
            robots[r].vnew = np.around(robots[-1].vnew - robots[r].distGeoCenter)
            robots[r].totalTime = robots[-1].totalTime
            robots[r].vnewInformation = robots[-1].vnewInformation
            robots[r].nearestNodeIdx = robots[-1].nearestNodeIdx
            robots[r].vnewCost = robots[-1].vnewCost
            robots[r].addNode()
        robots[-1].addNode()
        rewireGraph(robots[-1])

    # Get the path
    getPath(robots[-1])
    for r in range(0,len(robots)-1):
        robots[r].path = robots[-1].path
        robots[r].paths = robots[-1].paths
    # for r in range(0, len(robots)):  
    #     getPath(robots[r])
    
def initializeRobots(numRobots, logFile):
    """initialize the robot class"""
    # Input arguments:
    # numRobots = how many robots

    robots = []
    for r in range(0, numRobots):
        rob = Robot(r, DISCRETIZATION, UMAX, SENSORPERIOD, OPTPATH, OPTPOINT, SPATIOTEMPORAL, logFile)
        robots.append(rob)
    
    #Print test information
    if DEBUG:
        print('Robot 0 ID')
        print(robots[0].ID)
    
    return robots

def randomStartingPositions(numRobots):
    """Ensures that the starting position are exclusive within communication radius"""
    # Input arguments:
    # numRobots = how many robots
    
    locations = np.zeros([numRobots, 2])
    pos = np.random.randint(0,2*COMMRANGE, size=2)
    locations[0] = pos
    
    for i in range(1,numRobots):
        equal = True
        while equal:
            pos = np.random.randint(0, 2*COMMRANGE, size=2)
            equal = False
            for l in range(0, i):
                if np.array_equal(pos,locations[l]):
                    equal = True
                    break
        locations[i] = pos
        
    if DEBUG:
        print('Locations')
        print(locations)
        
    return locations.astype(int)

def errorCalculation(robots,logFile):
    for robot in robots:
        # rmse = np.sqrt(np.square(robot.mappingGroundTruth - robot.expectedMeasurement).mean())
        # nrmse = 100 * rmse/(np.max(robot.mappingGroundTruth)-np.min(robot.mappingGroundTruth))
        # logFile.writeError(robot.ID,nrmse,robot.currentTime, endTime=True)

        rmse = np.sqrt(np.sum(np.square(robot.mappingGroundTruth - robot.expectedMeasurement)))
        fnorm = rmse/(np.sqrt(np.sum(np.square(robot.mappingGroundTruth))))
        logFile.writeError(robot.ID,fnorm,robot.currentTime, endTime=True)

if __name__ == "__main__":
    """Entry in Test Program"""
    
    """Setup"""
    np.random.seed(1994)
    
    clearPlots()
    
    TOTALTIME = 60 #total execution time of program
    CASE = 3 #case corresponds to which robot structure to use (1 = 8 robots, 8 teams, 2 = 8 robots, 5 teams, 3 = 4 robots 4 teams)
    CORRECTTIMESTEP = True #If dye time steps should be matched to correct time steps or if each time step in dye corresponds to time step here
    
    DEBUG = False #debug to true shows prints
    ANIMATION = False #if animation should be done
    GAUSSIAN = True #if GP should be calculated
    OPTPATH = GAUSSIAN == True #if path optimization should be used, can not be true if optpoint is used
    OPTPOINT = GAUSSIAN != OPTPATH == True #if point optimization should be used, can not be true if optpath is used
    
    SPATIOTEMPORAL = False
    STATIONARY = not SPATIOTEMPORAL #if we are using time varying measurement data or not
    STATIONARYTIME = 5 #which starting time to use for the measurement data, if not STATIONARY, 0 is used for default
    PREDICTIVETIME = None #Time for which to make a prediction at the end, has to be bigger than total time

    SENSINGRANGE = 0 # Sensing range of robots
    COMMRANGE = 6 # communication range for robots
    
    DISCRETIZATION = np.array([600, 600]) #grid space
    DIMENSION = 2 #dimension of robot space
    RANDOMSAMPLESMAX = 30 #how many random samples before trying to converge for communication
    TOTALSAMPLES = 50 #how many samples in total

    SENSORPERIOD = 0.1 #time between sensor measurement or between updates of data
       
    UMAX = 50 # Max velocity, pixel/second
    EPSILON = DISCRETIZATION[0]/10 # Maximum step size of robots
    GAMMARRT = 100 # constant for rrt* algorithm, can it be calculated?
    
    FOLDER = 'Results/Tmp'
    LOGFILE = 'logFile'
    
    main()
