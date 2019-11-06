#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:48:29 2019

@author: hannes
"""

import numpy as np
from IntermittentComms import Schedule, Robot, sampleVrand, measurement, findNearestNode
from Visualization import plotMatrix

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
    
    #sample test
#    rangeSamples = DISCRETIZATION #only for uniform
#    vrand = sampleTest(rangeSamples,distribution='uniform')
    
#    rangeSamples = [[300,300],[10,10]] #only gaussian
#    vrand = sampleTest(rangeSamples,distribution='gaussian')
    createInitialPaths(schedule, teams, commPeriod, locations, robots, numRobots)
    
#    dataSensorMeasurements, totalMap = update(currentTime, robots, numRobots, locations)

def createInitialPaths(schedule, teams, commPeriod, locations, robots, numRobots):
    #add node v0 to list of nodes for each robot
    for r in range(0,numRobots):
        robots[r].nodes.append(locations[r])
    
    paths = []
    return paths

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
            pos = np.random.randint(0, COMMRANGE, size=2)
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


def update(currentTime, robots, numRobots, locations):
    """Update procedure of intermittent communication"""
    # Input arguments:
    # currentTime = current Time of the execution
    # robots = instances of the robots that are to be moved
    # numRobots = how many robots    
    # locations = starting locations of the robots
    
    currentTime = 0 
    
    # TODO needs massive rework, was wrong approch, need to take robots from teams which have a schedule on k, build their graph, take teams with k+1 and build there graph
    while currentTime < TOTALTIME:
        
        #sample new nodes and create path: Algorithm 1 from paper
        distribution = 'uniform'
        rangeSamples = DISCRETIZATION
        
        for sample in range(0,TOTALSAMPLES):
            if sample == RANDOMSAMPLESMAX:
                mean = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
                stdDev = np.diag(4*COMMRANGE*COMMRANGE*np.identity(2)) #TODO find a more elegant version to put dimension 2 there
                distribution = 'gaussian'
                rangeSamples = [mean,stdDev]
                 
            vrand = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
        
        nearestNodeIdx = findNearestNode(locations, vrand)

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

    
def testRobot(numRobots, teams, schedule):
    """Test the robot class"""
    # Input arguments:
    # numRobots = how many robots
    # teams = team assignments
    # schedule = schedule for meeting events


    robots = []
    for r in range(0, numRobots):
        rob = Robot(r + 1, teams[r][0], schedule[r], DISCRETIZATION)
        robots.append(rob)
        #Print test information
    if DEBUG:
        print('Robot 1 schedule')
        print(robots[0].schedule)
        
        print('Robot 1 team')
        print(robots[0].teams)
        
        print('Robot 1 ID')
        print(robots[0].ID)
    
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
    CASE = 3 #case corresponds to which robot structure to use (1 = 8 robots, 8 teams, 2 = 8 robots, 5 teams, 3 = 2 robots 2 teams)
    DEBUG = True #debug to true shows prints
    COMMRANGE = 3 #communication range for robots
    
    DISCRETIZATION = np.array([600, 600]) #grid space
    RANDOMSAMPLESMAX = 5 #how many random samples before trying to converge for communication
    TOTALSAMPLES = 10 #how many samples in total

    SENSORPERIOD = 20 #time between sensor measurement or between updates of data
    EIGENVECPERIOD = 40 #time between POD calculations
    
    TOTALTIME = 1000 #total execution time of program
    
    main()