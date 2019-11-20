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
from Utilities.ControllerUtilities import measurement
from Utilities.VisualizationUtilities import plotMatrix, plotMeetingGraphs, plotMeetingPaths, clearPlots
from Utilities.PathPlanningUtilities import (sampleVrand, findNearestNode, steer, buildSetVnear, 
                                             extendGraph, rewireGraph, calculateGoalSet, 
                                             checkGoalSet, leastCostGoalSet, getPath)


def main():
    """main test loop"""
    # no inputs 
               
    """create robot to team correspondence"""
    #robot i belongs to team j
    if CASE == 1:
        numTeams = 8
        numRobots = 8
        robTeams = np.array([[1, 1, 0, 0, 0, 0, 0, 0], 
                             [0, 1, 1, 0, 0, 0, 0, 0], 
                             [0, 0, 1, 1, 0, 0, 0, 0], 
                             [0, 0, 0, 1, 1, 0, 0, 0], 
                             [0, 0, 0, 0, 1, 1, 0, 0], 
                             [0, 0, 0, 0, 0, 1, 1, 0],
                             [0, 0, 0, 0, 0, 0, 1, 1], 
                             [1, 0, 0, 0, 0, 0, 0, 1],])
    
    elif CASE == 2:
        numTeams = 5
        numRobots = 8
        robTeams = np.array([[1, 1, 0, 0, 0], 
                             [1, 0, 0, 1, 0], 
                             [1, 0, 0, 0, 1], 
                             [0, 1, 1, 0, 0], 
                             [0, 1, 0, 0, 1], 
                             [0, 0, 1, 1, 0],
                             [0, 0, 1, 0, 1], 
                             [0, 0, 0, 1, 1],])
    elif CASE == 3:
        numTeams = 4
        numRobots = 4
        robTeams = np.array([[1, 0, 0, 1],
                             [1, 1, 0, 0],
                             [0, 1, 1, 0],
                             [0, 0, 1, 1],])
    
    else:
        exit()

    """Variables"""
    locations = randomStartingPositions(numRobots) #locations or robots



    """--------------------------------------------------------------------------"""    
    """Perform Tests"""
    #scheduler test
    schedule, teams, commPeriod = testScheduler(numRobots, numTeams, robTeams)
    
    #robot test
    robots = testRobot(numRobots, teams, schedule)
    
    # create the initial plans for all periods
    initialTime = 0
    for r in range(0,numRobots):
        robots[r].vnew = locations[r]
        robots[r].totalTime = initialTime
    
    for period in range(0,schedule.shape[1]):
        updatePaths(schedule, teams, robots, numRobots, period)
        for r in range(0,numRobots):
            robots[r].composeGraphs()
    
    
    
    if DEBUG:
        subplot = 1
        for r in teams:
            r = np.asarray(r[0]) -1
            plotMeetingGraphs(robots, r, subplot, len(teams))
            plotMeetingPaths(robots, r, subplot, len(teams))
            subplot += 1        
            
    #TODO write general update control function which moves robots along the paths
    currentTime = initialTime
    
    while currentTime < TOTALTIME:
        currentTime = update(currentTime, robots)
        
    """    
    dataSensorMeasurements, totalMap = update(currentTime, robots, numRobots, locations)
    """

def update(currentTime, robots):
    """Update procedure of intermittent communication"""
    # Input arguments:
    # currentTime = current Time of the execution
    # robots = instances of the robots that are to be moved
    # numRobots = how many robots    
    # locations = starting locations of the robots
    
    currentTime += SENSORPERIOD
    
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

def updatePaths(schedule, teams, robots, numRobots, period):
    """Update procedure of intermittent communication"""
    # Input arguments:
    # schedule = schedule of meeting events
    # teams = which robot belongs to which team
    # robots = instances of the robots that are to be moved
    # numRobots = how many robots    
    # period = how many epochs we have in the schedule
    
    #add node v0 to list of nodes for each robot       
    for r in range(0,numRobots):        
        #make last node equal to the first of new period
        if robots[r].nodeCounter > 0:
            robots[r].nodeCounter -= 1

        robots[r].startNodeCounter = robots[r].nodeCounter
        robots[r].startLocation = robots[r].endLocation
        robots[r].startTotalTime = robots[r].endTotalTime 
        robots[r].nearestNodeIdx = robots[r].endNodeCounter
        
        robots[r].initializeGraph()
        robots[r].addNode(firstTime = True)
    
        
    teamsDone = np.zeros(len(teams))

    #find out which team has a meeting event at period k=0
    for team in schedule[:, period]:
        
                
        if teamsDone[team] or team < 0:
            continue
        
        connected = False
        
        while not connected:    
            #sample new nodes and create path
            distribution = 'uniform'
            rangeSamples = DISCRETIZATION
            
            for sample in range(0,TOTALSAMPLES):
                if sample == RANDOMSAMPLESMAX-1:
                    mean = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
                    stdDev = 4*COMMRANGE*COMMRANGE*np.identity(2)
                    distribution = 'gaussian'
                    rangeSamples = [mean,stdDev]
                
                if sample >= RANDOMSAMPLESMAX:
                    vrand = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
                
                #find which robot is in team and get nearest nodes and new random samples for them
                for r in teams[team][0]:        
                    
                    if distribution == 'uniform':
                        vrand = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
                    
                    robots[r-1].vrand = vrand
                    
                    #find nearest node to random sample
                    nearestNodeIdx = findNearestNode(robots[r-1].graph,vrand)
                    robots[r-1].nearestNodeIdx = nearestNodeIdx
                
                #find new node towards max distance to random sample and incorporate time delay, that is why it is outside of previous loop since we need all the nearest nodes from the other robots
                steer(robots, team, teams, UMAX, EPSILON)
                
                for r in teams[team][0]:
                    # get all nodes close to new node
                    buildSetVnear(robots[r-1], EPSILON, GAMMARRT)
                    
                    extendGraph(robots[r-1], UMAX)
                    
                    robots[r-1].addNode()
                    
                # finding out if vnew should be in goal set
                # TODO should I really just start checking once we sample for meeting locations?
                if sample >= RANDOMSAMPLESMAX: 
                    calculateGoalSet(robots, team, teams, COMMRANGE, TIMEINTERVAL)
                
                rewireGraph(robots, team, teams, UMAX, TIMEINTERVAL, DEBUG)
                
            # check if we have a path
            for r in teams[team][0]: 
                connected = checkGoalSet(robots[r-1].graph)
                
                if not connected:
                    robots[r-1].nodeCounter = robots[r-1].startNodeCounter
                    robots[r-1].vnew = robots[r-1].startLocation
                    robots[r-1].totalTime = robots[r-1].startTotalTime
                    robots[r-1].initializeGraph()
                    robots[r-1].addNode(firstTime = True)
                else:
                    leastCostGoalSet(robots[r-1], DEBUG)
                    robots[r-1].vnew = robots[r-1].endLocation
                    robots[r-1].totalTime = robots[r-1].endTotalTime
                    getPath(robots[r-1])
                
        teamsDone[team] = True
    

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

def sampleTest(rangeSamples, distribution='uniform'):
    """Tests the random sample generation"""
    
    vrand = sampleVrand(DISCRETIZATION, rangeSamples, distribution)

    if DEBUG:
        print('vrand ' + distribution)
        print(vrand)

    return vrand    
    
def testRobot(numRobots, teams, schedule):
    """Test the robot class"""
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

    

 
def testScheduler(numRobots, numTeams, robTeams):
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
    communicationPeriod = np.shape(S)[0]  # Communication schedule repeats infinitely often

    #Print test information
    if DEBUG:
        print('Teams')
        print(*T)
        
        print('Schedule')
        print(S)
        
        print('Period')
        print(communicationPeriod)
    
    return S, T, communicationPeriod



if __name__ == "__main__":
    """Entry in Test Program"""
    
    """Setup"""
    np.random.seed(1994)
    
    clearPlots()
    
    CASE = 3 #case corresponds to which robot structure to use (1 = 8 robots, 8 teams, 2 = 8 robots, 5 teams, 3 = 2 robots 2 teams)
    DEBUG = True #debug to true shows prints
    COMMRANGE = 3 # communication range for robots
    TIMEINTERVAL = 1 # time interval for communication events
    
    DISCRETIZATION = np.array([600, 600]) #grid space
    RANDOMSAMPLESMAX = 30 #how many random samples before trying to converge for communication
    TOTALSAMPLES = 50 #how many samples in total

    SENSORPERIOD = 20 #time between sensor measurement or between updates of data
    EIGENVECPERIOD = 40 #time between POD calculations
    
    TOTALTIME = 1000 #total execution time of program
    
    UMAX = 50 # Max velocity, 30 pixel/second
    EPSILON = DISCRETIZATION[0]/10 # Maximum step size of robots
    GAMMARRT = 100 # constant for rrt* algorithm, can it be calculated?
    
    main()