#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:47:51 2019

@author: hannes
"""

#General imports
import numpy as np
import networkx as nx

def calculateGeometricCenter(locations, numRobots):
    geoCenter = np.array([0,0])
    for r in range(0,numRobots):
        geoCenter += locations[r]
    geoCenter = geoCenter/numRobots
    return geoCenter

def getPath(robot):
    """Get the shortest path for the robot to the end location"""
    # Input arguments
    # robot = robot instance whose graph we have to analize
    
    graphReversed = robot.graph.reverse()
    predecessors = nx.dfs_successors(graphReversed,robot.nodeCounter-1)
    
    path = []
    for key, value in predecessors.items():
        path.append(key)
        
    path.append(value[0])
    path.reverse()
    robot.paths.append(path)
    robot.path = robot.graph.subgraph(path)    

def updateSuccessorNodes(robot, startNode):
    """Update successor node weight and times as the rewiring changed the graph structure"""
    # Input arguments
    # robot = which robot graph that we are analyzing
    # startNode = which node got changed

    
    allSuccessors = nx.dfs_successors(robot.graph,startNode)

    for key, value in allSuccessors.items():
        for v in value:
            
            startTime = robot.graph.nodes[key]['t']
            startPos = robot.graph.nodes[key]['pos']
            succPos = robot.graph.nodes[v]['pos']
            succTotalCost, succEdgeCost, information = cost(startTime,startPos,succPos,robot)
            
            robot.graph[key][v]['weight'] = succEdgeCost
            robot.graph.nodes[v]['t'] = succTotalCost
            robot.graph.nodes[v]['informationGain'] = information             

def rewireGraph(robot):
    # TODO: Find out what this min t = max t means

    """Check if there is a node in setVnear which has a smaller cost as child from vnew than previous"""
    # Input arguments
    # robots = which robot graph that we are analyzing
    
    setVnear = robot.setVnear
    timeVnew = robot.totalTime
    posVnew = robot.vnew
        
    nearTotalCost = 0
    nearEdgeCost = 0
    information = 0
    
    for n in range(0,len(setVnear)):
        vnearTime = robot.graph.nodes[setVnear[n]]['t']
        
        vnearInformation = robot.graph.nodes[setVnear[n]]['informationGain']
        vnearUtility = getUtility(vnearTime,vnearInformation)
        
        vnearPos = robot.graph.nodes[setVnear[n]]['pos']
        nearTotalCost, nearEdgeCost, information = cost(timeVnew,posVnew,vnearPos,robot) 
        
        nearUtility = getUtility(nearTotalCost,information)
        
        if (nearTotalCost < vnearTime) or (nearUtility < vnearUtility):
            if list(robot.graph.pred[setVnear[n]]) == []:
                continue
            predecessor = list(robot.graph.pred[setVnear[n]])[-1]
            robot.graph.remove_edge(predecessor,setVnear[n])
            robot.graph.add_edge(robot.vnewIdx,setVnear[n], weight = nearEdgeCost)
            robot.graph.nodes[setVnear[n]]['t'] = nearTotalCost
            robot.graph.nodes[setVnear[n]]['informationGain'] = information
            updateSuccessorNodes(robot, setVnear[n])
            
def cost(nearTime, nearPos, newPos, robot):
    """Calculate cost between two nodes"""
    # Input arguments
    # nearTime = time at current near node
    # nearPos = position at current near node
    # newPos = position of vnew for which we search a less costly parent node
    # robot = robot which is updating cost
    
    normDist = np.sqrt(np.sum((nearPos - newPos)**2))
    nearEdgeCost = normDist/robot.uMax

    information = getInformationGain(robot,newPos)
#    nearEdgeCost = nearEdgeCost/information

    nearTotalCost = nearTime + nearEdgeCost

    return round(nearTotalCost,1), round(nearEdgeCost,1), information 

def extendGraph(robot):
    """Check if there is a node in setVnear which has a smaller cost as parent to vnew than the nearest node"""
    # Input arguments
    # robot = which robot graph that we are analyzing
    # uMax = maximum velocity of robot for cost computation
    
    setVnear = robot.setVnear
    
    vnew = robot.vnew
    vminCost = robot.totalTime

    vminInformation = robot.vnewInformation
    
    vminUtility = getUtility(vminCost,vminInformation)
    
    nearTotalCost = 0
    vminEdgeCost = robot.vnewCost
    
    for n in range(0,len(setVnear)):
        time = robot.graph.nodes[setVnear[n]]['t']
        pos = robot.graph.nodes[setVnear[n]]['pos']
        nearTotalCost, nearEdgeCost, information = cost(time,pos,vnew,robot) 
        
        nearUtility = getUtility(nearTotalCost,information)

        if (nearTotalCost < vminCost) or (nearUtility < vminUtility):        
            robot.nearestNodeIdx = setVnear[n]
            vminCost = nearTotalCost
            vminEdgeCost = nearEdgeCost
            vminUtility = nearUtility
            vminInformation = information
        
    robot.vnewCost = vminEdgeCost
    robot.totalTime = vminCost
    robot.vnewInformation = vminInformation

def buildSetVnear(robot, epsilon, gammaRRT):
    """Examine all nodes and build set with nodes close to vnew with a radius of communication range"""
    # Input Arguments
    # robot = current robot with its graph
    # epsilon = maximum allowed distance traveled
    # gammaRRT = parameter for radius of near nodes in RRT star
    
    setVnear = []
    
    vnew = robot.vnew
    cardinality = robot.graph.number_of_nodes()
    dimension = 2
    
    radius = min(gammaRRT*(pow(np.log(cardinality)/cardinality,1/dimension)), epsilon)
    
    dictNodes = nx.get_node_attributes(robot.graph,'pos')

    nodes = list(dictNodes.values())
    nodes = np.asarray(nodes)

    normDist = np.sqrt(np.sum((nodes - vnew)**2, axis=1))
    
    for n in range(0,len(nodes)):
        if normDist[n] < radius:
            setVnear.append(list(dictNodes.keys())[n])
    
    robot.setVnear = setVnear
    
def steer(robot, epsilon, commRange):
    """Steer towards vrand but only as much as allowed by the dynamics"""
    # Input Arguments
    # robots = robot classes
    # epsilon = maximum allowed distance traveled
    
    # TODO: Add different motion model
    #steer towards vrand   
    nearestNodeIdx = robot.nearestNodeIdx
    graphDict = robot.graph.nodes[nearestNodeIdx]
    
    vnearest = list(graphDict.values())
    
    vrand = robot.vrand
    nearestTime = np.asarray(vnearest[1])
    nearestNode = np.asarray(vnearest[0])


    dist = vrand - nearestNode 
    normDist = np.sqrt(np.sum((nearestNode - vrand)**2))

    s = min(epsilon,normDist)
    travelTime = s/robot.uMax
    if travelTime > 0:    
        vnew = np.around(nearestNode + robot.uMax*travelTime*dist/normDist)
        distVnew = np.sqrt(np.sum((nearestNode - vnew)**2))
        travelTimeVnew = distVnew/robot.uMax            
    else: 
        vnew = nearestNode
        travelTimeVnew = 0   

    totalTimeVnew = travelTimeVnew + nearestTime
    discretization = robot.discretization

    if not (0 + commRange <= vnew[0] < discretization[0] - commRange and 0 + commRange <= vnew[1] < discretization[1] - commRange):
        if vnew[0] >= discretization[0] - commRange:
            vnew[0] = discretization[0]- commRange - 1
        if vnew[1] >= discretization[1] - commRange:
            vnew[1] = discretization[1]-commRange - 1
        if vnew[0] < 0 + commRange:
            vnew[0] = 0 + commRange
        if vnew[1] < 0 + commRange:
            vnew[1] = 0 + commRange

    information = getInformationGain(robot, vnew)
    
    robot.vnew = np.around(vnew)
    robot.vnewCost = round(travelTimeVnew,1)
    robot.totalTime = round(totalTimeVnew,1)
    robot.vnewInformation = information
        
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
        if distribution == 'uniform':
            vrand = np.random.uniform([0,0],rangeSamples)
        
        if distribution == 'gaussian':
            vrand = np.random.multivariate_normal(rangeSamples[0],rangeSamples[1])
        if 0 <= vrand[0] < discretization[0] and 0 <= vrand[1] < discretization[1]:
            inBoundary = True
  
    return np.around(vrand)

def getInformationGain(robot, pos):
    """Get expected variance from GP model, either by inferring or lookup"""
    # Input Arguments
    # robot = robot instance whose GP is to be looked at
    # pos = position to check the expected variance for
    
    if robot.optPath:
        ys = robot.expectedVariance[np.int(pos[0]),np.int(pos[1])]
    elif robot.optPoint:
        ym, ys = robot.GP.inferGP(robot,pos)
    else:
        return 0
    return ys

# TODO: This doesn't seem to do anything, maybe remove it..
def getUtility(time, informationGain):
    """Utility function which calgulates utility by dividing cost by variance with weight"""
    # Input Arguments
    # time = cost of node
    # informationGain = gain of node

    beta = 4
    if informationGain < 1/beta:
        informationGain = 1/beta
    # return time/(beta*informationGain) 
    return 0
    
def sampleNPoints(robot, discretization, rangeSamples, distribution = 'uniform'):
    """Sample N new random positions and check for highest variance"""
    # Input Arguments
    # robot = robot instance to access GP
    # discretization = grid space needed to check for boundary issues
    # rangeSamples = defines uniform range boundaries or mean and std dev for gaussian
    # distribution = which distribution to use
        
    maxVariance = 0
    vrand = np.array([0, 0])
    for _ in range(0,10):
        point = sampleVrand(discretization, rangeSamples, distribution)
        var = getInformationGain(robot, point)
        if var > maxVariance:
            vrand = point
            maxVariance = var
    return vrand

def getInformationGainAlongPath(robot, pos, nearestNodeIdx, epsilon):
    """Calculate information gain along path for each sensor period"""
    # Input Arguments
    # robot = robot instance to access graph
    # pos = vrand from sampling
    # nearestNodeIdx = nearest node to vrand for steering
    # epsilon = maximum step size
    
    positionNotReached = True    
    deltaT = robot.sensorPeriod
    discretization = robot.discretization
    
    graphDict = robot.graph.nodes[nearestNodeIdx]
    vnearest = list(graphDict.values())
    nearestNode = np.asarray(vnearest[0])
    
    measPos = nearestNode
    var = 0
    
    distance = pos - measPos
    normDist = np.sqrt(np.sum((pos - nearestNode)**2))
    s = min(epsilon,normDist)
    if normDist == 0:
        return 0
    
    vnew = np.around(nearestNode + s*distance/normDist)
    pos = vnew
    
    while positionNotReached:
        
        distance = pos - measPos
        normDist = np.sqrt(np.sum((pos - measPos)**2))
    
        travelTime = normDist/robot.uMax
    
        if travelTime <= deltaT:
            measPos = pos
            positionNotReached = False
        else:
            step = np.around(robot.uMax*deltaT*distance/normDist)
            measPos = measPos + step
        
        if not (0 <= measPos[0] < discretization[0] and 0 <= measPos[1] < discretization[1]):
            if measPos[0] >= discretization[0]:
                measPos[0] = discretization[0]-1
            if measPos[1] >= discretization[1]:
                measPos[1] = discretization[1]-1
            if measPos[0] < 0:
                measPos[0] = 0
            if measPos[1] < 0:
                measPos[1] = 0
                
        if robot.optPath:
            ys = robot.expectedVariance[np.int(measPos[0]),np.int(measPos[1])]
        else:
            ym, ys = robot.GP.inferGP(robot,measPos)
        
        var += ys    
    
    return var
















