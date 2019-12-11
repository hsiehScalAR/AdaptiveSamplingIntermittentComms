#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:48:29 2019

@author: hannes
"""

#General imports
import numpy as np

#Personal imports
from Classes.Scheduler import Schedule
from Classes.Robot import Robot
from Setup import getSetup, setupMatlabFileMeasurementData, loadMeshFiles

from Utilities.ControllerUtilities import moveAlongPath, communicateToTeam, checkMeetingLocation, measurement
from Utilities.VisualizationUtilities import (plotMultivariateNormal, plotMeasurement, plotMeetingGraphs, plotMeetingPaths, 
                                              clearPlots, plotTrajectory, plotTrajectoryAnimation)
from Utilities.PathPlanningUtilities import (sampleVrand, findNearestNode, steer, buildSetVnear, 
                                             extendGraph, rewireGraph, calculateGoalSet, 
                                             checkGoalSet, leastCostGoalSet, getPath)

def main():
    """main test loop"""
    # no inputs 
    
    """Create Measurement Data"""
#    measurementGroundTruth = setupMatlabFileMeasurementData(DISCRETIZATION, invert=True)
    measurementGroundTruth = loadMeshFiles()
    plotMeasurement(measurementGroundTruth, 'Ground truth measurement map')
           
    """create robot to team correspondence"""
    numTeams, numRobots, robTeams, positions = getSetup(CASE)
    
    """Variables"""
    if positions.any():
        locations = positions
    else:
        locations = randomStartingPositions(numRobots) #locations or robots

    """Initialize schedules and robots"""
    schedule, teams, commPeriod = initializeScheduler(numRobots, numTeams, robTeams)
    robots = initializeRobots(numRobots, teams, schedule)

    """create the initial plans for all periods"""
    initialTime = 0
    for r in range(0,numRobots):
        robots[r].vnew = locations[r]
        robots[r].currentLocation = locations[r]
        robots[r].totalTime = initialTime
        robots[r].sensingRange = SENSINGRANGE
        
        # TODO: This needs to be changed if different measurements are used like time series data
        robots[r].mappingGroundTruth = measurementGroundTruth
        
        """Initialize GPs"""
        meas = measurement(robots[r])
        robots[r].createMap(meas, robots[r].currentLocation)
        if GAUSSIAN:
            robots[r].GP.initializeGP(robots[r])
    
    
    for period in range(0,schedule.shape[1]):
        teamsDone = np.zeros(len(teams))
    
        #find out which team has a meeting event at period k=0
        for team in schedule[:, period]:
            if teamsDone[team] or team < 0:
                continue
            
            robs = []
            for r in teams[team][0]:
                robs.append(robots[r-1])
            
            updatePaths(robs)
            teamsDone[team] = True
            
        for r in range(0,numRobots):
            robots[r].composeGraphs() 
            
    """Control loop"""
    currentTime = initialTime
    
    while currentTime < TOTALTIME:
        currentTime = update(currentTime, robots, teams, commPeriod)


    
    if DEBUG:
        subplot = 1
        team = 0
        for r in teams:
            r = np.asarray(r[0]) -1
            plotMeetingGraphs(robots, r, team, subplot, len(teams))
            plotMeetingPaths(robots, r, team, subplot, len(teams))
            subplot += 1
            team += 1
        
        plotTrajectory(robots)
        if ANIMATION:
            plotTrajectoryAnimation(robots)

    totalMap = robots[0].mapping
    plotMeasurement(totalMap, 'Measurements of robots after communication events')
    
    
    if GAUSSIAN:
        robots[0].GP.plotInferGP(robots[0])
        robots[1].GP.plotInferGP(robots[1])
        robots[2].GP.plotInferGP(robots[2])
        robots[3].GP.plotInferGP(robots[3])
    
    """    
    dataSensorMeasurements, totalMap = update(currentTime, robots, numRobots, locations)
    """

def update(currentTime, robots, teams, commPeriod):
    """Update procedure of intermittent communication"""
    # Input arguments:
    # currentTime = current Time of the execution
    # robots = instances of the robots that are to be moved
    # teams = teams of the robots
    # commPeriod = how many schedules there are
    
    atEndPoint = np.zeros(len(robots))
    
    for i, rob in enumerate(robots):
        atEndPoint[i] = moveAlongPath(rob, SENSORPERIOD, UMAX)

    currentTime += SENSORPERIOD
    
    for team in teams:
        
        if np.all(atEndPoint[team-1]):         
            
            currentLocation = []
            for r in team[0]: 
                currentLocation.append(robots[r-1].currentLocation)
            currentLocation = np.asarray(currentLocation)
            
            if not checkMeetingLocation(currentLocation, COMMRANGE):
                continue
            
            robs = []         
            for r in team[0]:                
                robots[r-1].scheduleCounter += 1
                robots[r-1].atEndLocation = False
                
                robots[r-1].endTotalTime  = currentTime
                robs.append(robots[r-1])
            
            communicateToTeam(robs, GAUSSIAN)
            
            updatePaths(robs)
            
            for r in team[0]:
                robots[r-1].composeGraphs()

    return currentTime
    
    """
        # Collect and send sensing data
        for r in range(0, numRobots):
            dataValues, singleMeasurement = measurement(numRobots, SENSORPERIOD)  # Measurements for all robots during one sensor period
            robots[r].addNewData(dataValues, locations, currentTime, currentTime + SENSORPERIOD, 'sensor')  # Set data matrices
            robots[r].createMap(singleMeasurement, locations[r])  # Create Map

        currentTime += SENSORPERIOD
        

    dataSensorMeasurements = Robot.constructDataMatrix()  # Aggregated matrix of estimated values from robots
    totalMap = Robot.getTotalMap()

    if DEBUG:
        
        print('mean random Sample')
        print(mean)
        print('stdDev random Sample')
        print(stdDev)
        
        print('Data Measurements')
        print(np.any(dataSensorMeasurements))
        plotMatrix(totalMap)

        
    return dataSensorMeasurements, totalMap
    """

def updatePaths(robots):
    """Update procedure of intermittent communication"""
    # Input arguments:
    # robots = instances of the robots that are to be moved
    
    #add node v0 to list of nodes for each robot       
    for r in range(0, len(robots)):        
        #make last node equal to the first of new period
        if robots[r].nodeCounter > 0:
            robots[r].nodeCounter -= 1

        robots[r].startNodeCounter = robots[r].nodeCounter
        robots[r].startLocation = robots[r].vnew
        robots[r].startTotalTime = robots[r].endTotalTime 
        robots[r].nearestNodeIdx = robots[r].endNodeCounter
        
        robots[r].initializeGraph()
        robots[r].addNode(firstTime = True)
             
    connected = False
    
    while not connected:    
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
            
            #find which robot is in team and get nearest nodes and new random samples for them
            for r in range(0, len(robots)):       
                
                if distribution == 'uniform':
                    vrand = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
                
                robots[r].vrand = vrand
                
                #find nearest node to random sample
                nearestNodeIdx = findNearestNode(robots[r].graph,vrand)
                robots[r].nearestNodeIdx = nearestNodeIdx
            
            #find new node towards max distance to random sample and incorporate time delay, that is why it is outside of previous loop since we need all the nearest nodes from the other robots
            steer(robots, UMAX, EPSILON)
            
            for r in range(0, len(robots)): 
                # get all nodes close to new node
                buildSetVnear(robots[r], EPSILON, GAMMARRT)
                
                extendGraph(robots[r], UMAX)
                
                robots[r].addNode()
                
            # finding out if vnew should be in goal set
            # TODO should I really just start checking once we sample for meeting locations?
            if sample >= RANDOMSAMPLESMAX: 
                calculateGoalSet(robots, COMMRANGE, TIMEINTERVAL)
            
            rewireGraph(robots, UMAX, TIMEINTERVAL, DEBUG)
            
        # check if we have a path
        for r in range(0, len(robots)):  
            connected = checkGoalSet(robots[r].graph)
            
            if not connected:
                robots[r].nodeCounter = robots[r].startNodeCounter
                robots[r].vnew = robots[r].startLocation
                robots[r].totalTime = robots[r].startTotalTime
                robots[r].initializeGraph()
                robots[r].addNode(firstTime = True)
            else:
                leastCostGoalSet(robots[r], DEBUG)
                robots[r].vnew = robots[r].endLocation
                robots[r].totalTime = robots[r].endTotalTime
                getPath(robots[r])
                
    
def initializeRobots(numRobots, teams, schedule):
    """initialize the robot class"""
    # Input arguments:
    # numRobots = how many robots
    # teams = team assignments
    # schedule = schedule for meeting events

    robots = []
    for r in range(0, numRobots):
        belongsToTeam = []
        for t in range(0,len(teams)):    
#            if r in teams[t]:
            if r+1 in teams[t]:
#                print('Robot ', str(r), ' in team ', str(t))
                belongsToTeam.append(t)
        rob = Robot(r, np.asarray(belongsToTeam), schedule[r], DISCRETIZATION)
        robots.append(rob)
    
    #Print test information
    if DEBUG:
        print('Robot 0 ID')
        print(robots[0].ID)
        
        print('Robot 0 schedule')
        print(robots[0].schedule)
        
        print('Robot 0 teams')
        print(robots[0].teams)
    
    return robots

def initializeScheduler(numRobots, numTeams, robTeams):
    """initialize schedule and create teams and schedule"""  
    # Input arguments:
    # numRobots = how many robots
    # numTeams = how many teams
    # robTeams = which robots are in which teams, comes from initial graph design; robot i belongs to team j in matrix
    
    #initializer
    scheduleClass = Schedule(numRobots, numTeams, robTeams)
    
    #Assigns robot numbers to teams
    T = scheduleClass.createTeams()
    #creates schedule
    S = scheduleClass.createSchedule()
    #communication period is equall to number of robots
    communicationPeriod = np.shape(S)[1]  # Communication schedule repeats infinitely often

    #Print test information
    if DEBUG:
        print('Teams')
        print(*T)
        
        print('Schedule')
        print(S)
        
        print('Period')
        print(communicationPeriod)
    
    return S, T, communicationPeriod

def randomStartingPositions(numRobots):
    """Ensures that the starting position are exclusive within communication radius"""
    # Input arguments:
    # numRobots = how many robots
    
    locations = np.zeros([numRobots, 2])
    pos = np.random.randint(0, COMMRANGE, size=2)
    locations[0] = pos
    
    for i in range(1,numRobots):
        equal = True
        while equal:
            pos = np.random.randint(0, numRobots, size=2)
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

if __name__ == "__main__":
    """Entry in Test Program"""
    
    """Setup"""
    np.random.seed(1994)
    
    clearPlots()
    
    CASE = 3 #case corresponds to which robot structure to use (1 = 8 robots, 8 teams, 2 = 8 robots, 5 teams, 3 = 4 robots 4 teams)
    DEBUG = False #debug to true shows prints
    ANIMATION = False #if animation should be done
    GAUSSIAN = True #if GP should be calculated
    
    SENSINGRANGE = 0 # Sensing range of robots, 0 or bigger than 2
    COMMRANGE = 3 # communication range for robots
    TIMEINTERVAL = 1 # time interval for communication events
    
    DISCRETIZATION = np.array([600, 600]) #grid space
    DIMENSION = 2 #dimension of robot space
    RANDOMSAMPLESMAX = 30 #how many random samples before trying to converge for communication
    TOTALSAMPLES = 50 #how many samples in total

    SENSORPERIOD = 0.1 #time between sensor measurement or between updates of data
    EIGENVECPERIOD = 0.04 #time between POD calculations
    
    TOTALTIME = 50 #total execution time of program
    
    UMAX = 50 # Max velocity, pixel/second
    EPSILON = DISCRETIZATION[0]/10 # Maximum step size of robots
    GAMMARRT = 100 # constant for rrt* algorithm, can it be calculated?
    
    main()
