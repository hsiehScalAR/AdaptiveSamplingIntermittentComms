#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:48:29 2019

@author: hannes
"""

import numpy as np
from IntermittentComms import Schedule, Robot, sampleVrand, measurement, findNearestNode, steer, buildSetVnear, extendGraph, rewireGraph
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
    
    # create the initial plans for all periods
    initialTime = 0
    for r in range(0,numRobots):
        robots[r].vnew = locations[r]
        robots[r].totalTime = initialTime
    
    for period in range(0,schedule.shape[1]):
        createInitialPaths(schedule, teams, robots, numRobots, period)
        for r in range(0,numRobots):
            robots[r].composeGraphs()
    
    plotMeetingGraphs(robots[0].totalGraph,robots[1].totalGraph)
    plotMeetingGraphs(robots[1].totalGraph,robots[2].totalGraph)
    plotMeetingGraphs(robots[2].totalGraph,robots[3].totalGraph)
    plotMeetingGraphs(robots[0].totalGraph,robots[3].totalGraph)
    
    print('Graph robot 0')        
    print(robots[0].totalGraph.nodes.data())
    
    print('Graph robot 1')        
    print(robots[1].totalGraph.nodes.data())

    print('Graph robot 2')        
    print(robots[2].totalGraph.nodes.data())
    
    print('Graph robot 3')        
    print(robots[3].totalGraph.nodes.data())
        

#    dataSensorMeasurements, totalMap = update(currentTime, robots, numRobots, locations)

def createInitialPaths(schedule, teams, robots, numRobots, period):
    
    #add node v0 to list of nodes for each robot       
    for r in range(0,numRobots):
        robots[r].nearestNodeIdx = robots[r].nodeCounter
        
        #make last node equal to the first of new period
        if robots[r].nodeCounter > 0:
            robots[r].nodeCounter -= 1
        
        robots[r].initializeGraph()
        robots[r].addNode(firstTime = True)
        
    teamsDone = np.zeros(len(teams))

    #find out which team has a meeting event at period k=0
    for team in schedule[:, period]:
        
        if teamsDone[np.int(team)] or team < 0:
            continue
        
        #sample new nodes and create path
        distribution = 'uniform'
        rangeSamples = DISCRETIZATION
        
        for sample in range(0,TOTALSAMPLES):
            if sample == RANDOMSAMPLESMAX:
                mean = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
#                stdDev = np.diag(4*COMMRANGE*COMMRANGE*np.identity(2)) #TODO find a more elegant version to put dimension 2 there, something comming from the 2D or *D inputs
                stdDev = 4*COMMRANGE*COMMRANGE*np.identity(2) #TODO find a more elegant version to put dimension 2 there, something comming from the 2D or *D inputs
                distribution = 'gaussian'
                rangeSamples = [mean,stdDev]
            
            #find which robot is in team and get nearest nodes and new random samples for them
            for r in teams[np.int(team)][0]:                
                vrand = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
                robots[np.int(r-1)].vrand = vrand
                
                #find nearest node to random sample
                nearestNodeIdx = findNearestNode(robots[np.int(r-1)].graph,vrand)
                robots[np.int(r-1)].nearestNodeIdx = nearestNodeIdx
            
            #find new node towards max distance to random sample and incorporate time delay, that is why it is outside of previous loop since we need all the nearest nodes from the other robots
            steer(robots, team, teams, UMAX, EPSILON)
            
            for r in teams[np.int(team)][0]:
                # get all nodes close to new node
                setVnear = buildSetVnear(robots[np.int(r-1)], EPSILON, GAMMARRT)
                
                extendGraph(robots[np.int(r-1)], setVnear, UMAX)
                
                robots[np.int(r-1)].addNode()
                
                rewireGraph(robots[np.int(r-1)], setVnear, UMAX)
            
        teamsDone[np.int(team)] = True     
        
    paths = []
    
    if DEBUG:
        print('Graph robot 0')        
        print(robots[0].graph.nodes.data())
        
        print('Graph robot 1')        
        print(robots[1].graph.nodes.data())
    
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
                print('mean and stddev')
                print(rangeSamples)
                 
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
    
    CASE = 3 #case corresponds to which robot structure to use (1 = 8 robots, 8 teams, 2 = 8 robots, 5 teams, 3 = 2 robots 2 teams)
    DEBUG = False #debug to true shows prints
    COMMRANGE = 3 # TODO: communication range for robots
    
    DISCRETIZATION = np.array([600, 600]) #grid space
    RANDOMSAMPLESMAX = 10 #how many random samples before trying to converge for communication
    TOTALSAMPLES = 20 #how many samples in total

    SENSORPERIOD = 20 #time between sensor measurement or between updates of data
    EIGENVECPERIOD = 40 #time between POD calculations
    
    TOTALTIME = 1000 #total execution time of program
    
    UMAX = 10 # TODO: Max velocity, 30 pixel/second
    EPSILON = 50 # TODO: Maximum step size of robots
    GAMMARRT = 100 # TODO: constant for rrt* algorithm, can it be calculated?
    
    main()