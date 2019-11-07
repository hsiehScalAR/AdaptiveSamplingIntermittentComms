#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:48:29 2019

@author: hannes
"""

import numpy as np
from IntermittentComms import Schedule, Robot, sampleVrand, measurement, findNearestNode, steer
from Visualization import plotMatrix, plotMeetingGraphs

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
        robots[r].addNode(locations[r])
    
    lastTeam = -1
    
    #find out which team has a meeting event at period k=0
    for team in schedule[:, 0]:
        if lastTeam == team or team < 0:
            continue
        
        #sample new nodes and create path
        distribution = 'uniform'
        rangeSamples = DISCRETIZATION
        
        for sample in range(0,TOTALSAMPLES):
            if sample == RANDOMSAMPLESMAX:
                mean = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
                stdDev = np.diag(4*COMMRANGE*COMMRANGE*np.identity(2)) #TODO find a more elegant version to put dimension 2 there
                distribution = 'gaussian'
                rangeSamples = [mean,stdDev]
            
            #find which robot is in team
            for r in teams[np.int(team)][0]:                
                vrand = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
                
                #find nearest node to random sample
                nearestNode = findNearestNode(robots[np.int(r-1)].graph,vrand)
                
                #find new node towards max distance to random sample
                # TODO steer has to be outside of this robot loop because I need the information from all robots of the time to compute vnew
                vnew = steer(vrand,robots[np.int(r-1)].graph.nodes[nearestNode], UMAX, EPSILON)
                
                robots[np.int(r-1)].addNode(vnew,nearestNode)
            
        lastTeam = team       
        
    paths = []
    
    if DEBUG:
        print('Graph robot 0')        
        print(robots[0].graph.nodes.data())
        
        print('Graph robot 1')        
        print(robots[1].graph.nodes.data())
        
        plotMeetingGraphs(robots[0].graph,robots[1].graph)
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
        belongsToTeam = []
        for t in range(0,len(teams)):    
#            if r in teams[t]:
            if r+1 in teams[t]:
                print('Robot ', str(r), ' in team ', str(t))
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
#    teams = np.asarray(T).reshape((4, 2))-1
    #creates schedule
    S = scheduleClass.createSchedule()
    #communication period is equall to number of robots
    communicationPeriod = np.shape(S)[0]  # Communication schedule repeats infinitely often

    #Print test information
    if DEBUG:
        print('Teams')
#        print(teams)
        print(*T)
        
        print('Schedule')
        print(S)
        
        print('Period')
        print(communicationPeriod)
    
#    return S, teams, communicationPeriod
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
    
    UMAX = 50 #Max velocity, 30 pixel/second
    EPSILON = 100 #Maximum step size of robots
    main()