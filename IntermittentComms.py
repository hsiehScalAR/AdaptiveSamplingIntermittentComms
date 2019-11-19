#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:48:29 2019

@author: hannes
"""

import numpy as np
from operator import itemgetter
import networkx as nx

def getPath(robot):
    
    graphReversed = robot.graph.reverse()
    predecessors = nx.dfs_successors(graphReversed,robot.endNodeCounter)
    
    path = []
    for key, value in predecessors.items():
        path.append(key)
        
    path.append(value[0])
    path.reverse()

    robot.path = robot.graph.subgraph(path)

def leastCostGoalSet(robot):
    """Get the goal set node with the least cost"""
    # Input arguments
    # robot = robot whose graph is to be checked
    

    graph = robot.graph
    dictNodes = nx.get_node_attributes(graph,'goalSet')
    goalNodes = list(dictNodes.keys())
    
    goalNodes = np.asarray(goalNodes)
        
    for node in goalNodes:
        times = []
        times.append(graph.nodes[node]['t'])
        
    times = np.asarray(times)

    robot.endLocation = graph.nodes[goalNodes[np.argmin(times)]]['pos']
    robot.endTotalTime = np.min(times)
    robot.endNodeCounter = goalNodes[np.argmin(times)]
    print('leastCostGoalSet')
    print('Robot ID: %d' %robot.ID)
    print('End Node: %d' %robot.endNodeCounter)
    print('End Time: %.2f\n' %robot.endTotalTime)

def checkGoalSet(graph):
    """Check if there are any nodes in the goal set"""
    # Input arguments
    # graph = graph which is to be checked
    
    dictNodes = nx.get_node_attributes(graph,'goalSet')
    goalNodes = list(dictNodes.keys())
    
    if not goalNodes:
        return False

    return True

def updateGoalSet(robots, team, teams, timeInterval):
    """Check if goal set is still valid after rewiring"""
    # Input arguments
    # robots = all robot instances
    # team = which team we currently look at
    # teams = instance of all teams
    # timeInterval = arrival time interval for no delay constraint
    
    r = teams[team][0][0]
    
    dictNodes = nx.get_node_attributes(robots[r-1].graph,'goalSet')
    goalNodes = list(dictNodes.keys())
    
    if not goalNodes:
        pass    
    goalNodes = np.asarray(goalNodes)
        
    for node in goalNodes:
        times = []
        
        for r in teams[team][0]:
            times.append(robots[r-1].graph.nodes[node]['t'])
        
        times = np.asarray(times)
        timeDiff = np.abs(times - times[0])
        
        if not all(y <= timeInterval for y in timeDiff):
            print('Update goal set')
            print(timeDiff)
            print('Deleted goalsetNode')
            print(node)
            
            for r in teams[team][0]: 
                del robots[r-1].graph.nodes[node]['goalSet']                
    
    
def calculateGoalSet(robots, team, teams, commRadius, timeInterval):
    """check if the new nodes are close to each other and if the time is right so that they belong to the goal set"""
    # Input arguments
    # robots = all robot instances
    # team = which team we currently look at
    # teams = instance of all teams
    # commRadius = communication range of the robots
    
    positions = []
    times = []
    for r in teams[team][0]:
        positions.append(robots[r-1].graph.nodes[robots[r-1].vnewIdx]['pos'])
        times.append(robots[r-1].graph.nodes[robots[r-1].vnewIdx]['t'])
    
    positions = np.asarray(positions)
    times = np.asarray(times)
    
    timeDiff = np.abs(times - times[0])
    normDist = np.sqrt(np.sum((positions[0] - positions)**2, axis=1))

    if all(x <= commRadius for x in normDist) and all(y <= timeInterval for y in timeDiff):
        for r in teams[team][0]:
            robots[r-1].graph.nodes[robots[r-1].vnewIdx]['goalSet'] = True

def updateSuccessorNodes(robot, startNode, uMax):
    """Update successor node weight and times as the rewiring changed the graph structure"""
    # Input arguments
    # robot = which robot graph that we are analyzing
    # startNode = which node got changed
    # uMax = maximum velocity of robot for cost computation
    
    allSuccessors = nx.dfs_successors(robot.graph,startNode)

    for key, value in allSuccessors.items():
        for v in value:
            
            startTime = robot.graph.nodes[key]['t']
            startPos = robot.graph.nodes[key]['pos']
            succPos = robot.graph.nodes[v]['pos']
            succTotalCost, succEdgeCost = cost(startTime,startPos,succPos,uMax)
            
            robot.graph[key][v]['weight'] = succEdgeCost
            robot.graph.nodes[v]['t'] = succTotalCost            

def rewireGraph(robots, team, teams, uMax, timeInterval):
    # TODO: Find out what this min t = max t means

    """Check if there is a node in setVnear which has a smaller cost as child from vnew than previous"""
    # Input arguments
    # robot = which robot graph that we are analyzing
    # setVnear = set of close nodes for which we check the cost to  be a child of vnew
    # uMax = maximum velocity of robot for cost computation
    
    rewired = False
    
    for r in teams[team][0]:
        robot = robots[r-1]
        setVnear = robot.setVnear
        timeVnew = robot.totalTime
        posVnew = robot.vnew
            
        nearTotalCost = 0
        nearEdgeCost = 0
        
        for n in range(0,len(setVnear)):
            vnearTime = robot.graph.nodes[setVnear[n]]['t']
            vnearPos = robot.graph.nodes[setVnear[n]]['pos']
            nearTotalCost, nearEdgeCost = cost(timeVnew,posVnew,vnearPos,uMax) 
    
            if nearTotalCost < vnearTime:
                predecessor = list(robot.graph.pred[setVnear[n]])[-1]
                robot.graph.remove_edge(predecessor,setVnear[n])
                robot.graph.add_edge(robot.vnewIdx,setVnear[n], weight = nearEdgeCost)
                robot.graph.nodes[setVnear[n]]['t'] = nearTotalCost
                updateSuccessorNodes(robot, setVnear[n], uMax)
                rewired = True
                
    if rewired:
        updateGoalSet(robots, team, teams, timeInterval)
            
def cost(nearTime, nearPos, newPos, uMax):
    # TODO: add information gain cost

    """Calculate cost between two nodes"""
    # Input arguments
    # nearTime = time at current near node
    # nearPos = position at current near node
    # newPos = position of vnew for which we search a less costly parent node
    # uMax = maximum velocity of robot for cost computation
    
    normDist = np.sqrt(np.sum((nearPos - newPos)**2))
    nearEdgeCost = normDist/uMax
    nearTotalCost = nearTime + nearEdgeCost
    
    return nearTotalCost, nearEdgeCost 

def extendGraph(robot, uMax):
    """Check if there is a node in setVnear which has a smaller cost as parent to vnew than the nearest node"""
    # Input arguments
    # robot = which robot graph that we are analyzing
    # uMax = maximum velocity of robot for cost computation
    
    setVnear = robot.setVnear
    
    vnew = robot.vnew
    vminCost = robot.totalTime
    
    nearTotalCost = 0
    nearEdgeCost = robot.vnewCost
    
    for n in range(0,len(setVnear)):
        time = robot.graph.nodes[setVnear[n]]['t']
        pos = robot.graph.nodes[setVnear[n]]['pos']
        nearTotalCost, nearEdgeCost = cost(time,pos,vnew,uMax) 

        if nearTotalCost < vminCost:
            robot.nearestNodeIdx = setVnear[n]
            vminCost = nearTotalCost     
    
    robot.vnewCost = nearEdgeCost
    robot.totalTime = vminCost

def buildSetVnear(robot, epsilon, gammaRRT):
    """Examine all nodes and build set with nodes close to vnew with a radius of communication range"""
    # Input Arguments
    # robot = current robot with its graph
    # epsilon = maximum allowed distance traveled
    # gammaRRT = parameter for radius of near nodes in RRT star
    
    setVnear = []
    
    vnew = robot.vnew
    cardinality = robot.graph.number_of_nodes()
    dimension = 2 # TODO: Find nicer way to say 2 here
    
    radius = min(gammaRRT*(pow(np.log(cardinality)/cardinality,1/dimension)), epsilon)
    
    dictNodes = nx.get_node_attributes(robot.graph,'pos')

    nodes = list(dictNodes.values())
    nodes = np.asarray(nodes)

    normDist = np.sqrt(np.sum((nodes - vnew)**2, axis=1))
    
    for n in range(0,len(nodes)):
        if normDist[n] < radius:
            setVnear.append(list(dictNodes.keys())[n])
    
    robot.setVnear = setVnear
    
def steer(robots, team, teams, uMax, epsilon):
    """Steer towards vrand but only as much as allowed by the dynamics"""
    # Input Arguments
    # robots = robot classes
    # team = current path planning for idx team
    # teams = contains all teams
    # uMax = maximal velocity in pixel/second
    # epsilon = maximum allowed distance traveled
    
    #find minTimes for nearest nodes of all robots
    minTimes = []
    for r in teams[team][0]:
        nearestNodeIdx = robots[r-1].nearestNodeIdx
        graphDict = robots[r-1].graph.nodes[nearestNodeIdx]
        
        vnearest = list(graphDict.values())
        nearestTime = np.asarray(vnearest[1])
        
        minTimes.append(nearestTime)
    
    #steer towards vrand
    for r in teams[team][0]:    
        nearestNodeIdx = robots[r-1].nearestNodeIdx
        graphDict = robots[r-1].graph.nodes[nearestNodeIdx]
        
        vnearest = list(graphDict.values())
        
        vrand = robots[r-1].vrand
        nearestTime = np.asarray(vnearest[1])
        nearestNode = np.asarray(vnearest[0])
    
    
        dist = vrand - nearestNode 
        normDist = np.sqrt(np.sum((nearestNode - vrand)**2))

        s = min(epsilon,normDist)
        travelTime = s/uMax

        deltaTcost = travelTime - (nearestTime - min(minTimes))
        
        if deltaTcost > 0:          
            vnew = np.around(nearestNode + uMax*deltaTcost*dist/normDist)
            distVnew = np.sqrt(np.sum((nearestNode - vnew)**2))
            travelTimeVnew = distVnew/uMax            
        else: 
            vnew = nearestNode
            travelTimeVnew = 0        
        
        totalTimeVnew = travelTimeVnew + nearestTime
        robots[r-1].vnew = np.around(vnew)
        robots[r-1].vnewCost = travelTimeVnew
        robots[r-1].totalTime = totalTimeVnew
    

def measurement(numRobots, sensorPeriod): 
    #TODO check how we measure stuff, if single value since each robot measure one place or measurement over time for all robots
    """Simulates a measurement for all robots over time and for a single robot at one time instance"""
    # Input Arguments
    # numRobots = how many robots
    # sensorPeriod = period of sensing in ms
    
    allMeasurementsOverTime = np.random.uniform(0,1,(numRobots, sensorPeriod))
    singleMeasurement = np.random.uniform(0,1)
    return allMeasurementsOverTime, singleMeasurement

def findNearestNode(graph, vrand):
    """Return nearest node index"""
    # Input Arguments
    # graph = current graph
    # vrand = new random node
    
    dictNodes = nx.get_node_attributes(graph,'pos')

    nodes = list(dictNodes.values())
    nodes = np.asarray(nodes)

    normDist = np.sum((nodes - vrand)**2, axis=1)
    
    return list(dictNodes.keys())[np.argmin(normDist)]

def sampleVrand(discretization, rangeSamples, distribution = 'uniform'):
    """Sample a new random position"""
    # Input Arguments
    # discretization = grid space needed to check for boundary issues
    # rangeSamples = defines uniform range boundaries or mean and std dev for gaussian
    # distribution = which distribution to use
    
    inBoundary = False
    while inBoundary == False:
        if distribution == 'uniform': #TODO check if we should use 0 for lower bound or something else
            vrand = np.random.uniform([0,0],rangeSamples)
        
        if distribution == 'gaussian':
            vrand = np.random.multivariate_normal(rangeSamples[0],rangeSamples[1])
        if 0 <= vrand[0] <= discretization[0] and 0 <= vrand[1] <= discretization[1]:
            inBoundary = True
  
    return np.around(vrand)


class Robot:
    
    objs = []  # Registrar keeps all attributes of class

    def __init__(self, ID, teams, schedule, discretization):
        """Initializer of robot class"""
        # Input arguments:
        # ID = robot number
        # teams = to which team each robot belongs
        # schedule = schedule of teams
        # discretization = grid space 
        
        self.ID = ID
        self.teams = teams
        self.schedule = schedule
        self.activeLocations = {}  # Store active location as indexed by (timeStart, timeEnd): locations
        self.sensorData = {}  # Store data as (timeStart, timeEnd): data
        self.eigenData = {}
        self.mapping = np.zeros([600,600])
        Robot.objs.append(self)
        Robot.discretization = discretization

        
        # Graph variables
        self.graph = nx.DiGraph()
        self.totalGraph = nx.DiGraph()
        
        self.path = nx.DiGraph()
        self.totalPath = nx.DiGraph()
        
        self.nodeCounter = 0
        self.nearestNodeIdx = 0
        self.vrand = np.array([0, 0])
        
        self.vnew = np.array([0, 0])
        self.vnewIdx = 0
        self.vnewCost = 0
        
        self.totalTime = 0
        
        self.setVnear = []
    
        self.startTotalTime = 0
        self.startNodeCounter = 0
        self.startLocation = np.array([0, 0])
        
        self.endTotalTime = 0
        self.endNodeCounter = 0
        self.endLocation = np.array([0, 0])
        
    def composeGraphs(self):
        self.totalGraph = nx.compose(self.totalGraph,self.graph)
        self.totalPath = nx.compose(self.totalPath,self.path)
    
    def initializeGraph(self):
        self.graph = nx.DiGraph()
    
    def addNode(self, firstTime = False):
        """Add new node with pos and total time attributes and edge with edge travel time cost to graph based on self variables"""
        
        self.graph.add_node(self.nodeCounter, pos = self.vnew, t = self.totalTime)
        self.vnewIdx = self.nodeCounter
        if self.nodeCounter != 0 and firstTime == False:
            self.graph.add_edge(self.nearestNodeIdx,self.nodeCounter, weight = self.vnewCost)
        self.nodeCounter += 1

    def createMap(self,newData,currentLocations):
        """creates a measurement map in the grid space without time reference"""
        # Input arguments:
        # newData = new measurement for single robot
        # currentLocations = sensing location of single robot
        
        self.mapping[currentLocations[0],currentLocations[1]] = newData
    
    @classmethod     
    def getTotalMap(cls):
        """Gives the complete sensor measurements of all robots in the grid space"""
        # Input is class
        
        totalMap = np.zeros(Robot.discretization)
        for obj in cls.objs: 
            totalMap += obj.mapping
        return totalMap
        
    def addNewData(self, newData, currentLocations, timeStart, timeEnd, dataType):
        """Update Robot data either from measurements or eigenvalues"""
        # Input arguments:
        # newData = new measurement or eigenvalue calculations of all robots
        # currentLocations = sensing locations of all robots
        # timeStart = start time of measurements
        # timeEnd = end time of measurements
        # dataType = if sensor or eigenvalue
        
        if dataType == 'sensor':
            self.sensorData[(timeStart, timeEnd)] = newData
            
        else:
            self.eigenData[(timeStart, timeEnd)] = newData

        self.activeLocations[(timeStart, timeEnd)] = currentLocations  # Store active location as indexed by (timeStart, timeEnd): locations

    def getDataRobots(self, newData, dataType):
        """Get Robot data either from measurements or eigenvalues"""
        # Input arguments:
        # newData = new measurement or eigenvalue calculations
        # dataType = if sensor or eigenvalue
        
        if dataType == 'sensor':
            self.sensorData = {**self.sensorData, **newData}
        else:
            self.eigenData = {**self.eigenData, **newData}

    @classmethod 
    def constructDataMatrix(cls):
        #TODO check if I can do that differently and if I even need it
        """Gives the time evolution of the measurements in [gridX,gridY, time]"""
        
        maxTime = 0
        # Find end time of data matrix
        for obj in cls.objs:
            keys = list(obj.sensorData.keys())
            maxKeys = max(keys, key=itemgetter(1))[1]  # Returns largest end time
            if maxTime < maxKeys:
                maxTime = maxKeys

        dataMatrix = np.zeros((Robot.discretization[0], Robot.discretization[1], maxTime))
        
        for obj in cls.objs:  # Fill in data matrix

            # Match sensor data start and end time to active locations
            for key, data in obj.sensorData.items():
                dataStartTime, dataEndTime = key[0], key[1]
                
                dataMatrix[obj.activeLocations[(dataStartTime, dataEndTime)][:, 0], obj.activeLocations[(dataStartTime, dataEndTime)][:, 1], dataStartTime:dataEndTime] = data
        return dataMatrix

    @classmethod
    def estimateLossyMatrix(cls, dataMatrix, eigenvalues, eigenvectors):
        """Estimates missing values of matrix using POD eigenvalues and vectors"""
        
        estimateMatrix = np.zeros_like(dataMatrix)
        return estimateMatrix



class Schedule:
    """scheduler class to create teams and schedules"""
    # Input arguments:
    # numRobots = how many robots
    # numTeams = how many teams
    # robTeams = which robots are in which teams, comes from initial graph design; robot i belongs to team j in matrix
    
    def __init__(self, numRobots, numTeams, robTeams):
        self.numRobots = numRobots
        self.numTeams = numTeams
        self.robTeams = robTeams

    def createTeams(self):
        """Create teams based on number of robots and number of teams"""
        
        T = [[] for x in range(self.numTeams)]

        for i in range(0, self.numTeams):
            T[i] = np.where(self.robTeams[:, i] > 0)
            T[i] += np.ones(np.shape(T[i]),dtype=int)
        return T

    def createSchedule(self):
        """Create schedule based on team compositions"""
        
        T = self.createTeams()
        schedule = np.zeros((self.numRobots, self.numTeams))
        teams = np.where(self.robTeams[0,:] > 0)[0].astype('int')
        teams = teams + np.ones(np.shape(teams))
        teams = teams.astype('int')
        
        schedule[0, 0:np.shape(teams)[0]] = teams

        for j in range(0, self.numRobots):
            teams = np.where(self.robTeams[j, :] > 0)[0].astype('int')
            
            teams = teams + np.ones(np.shape(teams))
            teams = teams.astype('int')

            for t in range(0, np.shape(teams)[0]):
                rule12 = False
                rule3 = False
                team = teams[t]

                for col in range(0, self.numTeams):
                    if team in schedule[:, col]:
                        schedule[j, col] = team
                        rule12 = True
                        break
                if not rule12:
                    col = 0
                    while col <= self.numTeams and not rule3:
                        placedTeams = np.unique(schedule[np.where(schedule[:, col] > 0), col]).astype('int')
                        totalSum = 0
                        for pt in range(0, np.shape(placedTeams)[0]):
                            pteam = placedTeams[pt].astype('int')
                            if np.intersect1d(T[team - 1], T[pteam - 1]).size == 0:
                                totalSum += 1
                        if totalSum == np.shape(placedTeams)[0]:
                            schedule[j, col] = team
                            rule3 = True
                        col += 1

        schedule = schedule[:, ~np.all(schedule == 0, axis=0)]  # Remove columns full of zeros
        
        #Make the indexes go from 0 to numTeams-1 instead of 1 to numTeams
        schedule = schedule -1
        schedule = schedule.astype('int')
        return schedule
