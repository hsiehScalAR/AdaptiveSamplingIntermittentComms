#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:47:51 2019

@author: hannes
"""

#General imports
import numpy as np
import networkx as nx

def getPath(robot):
    """Get the shortest path for the robot to the end location"""
    # Input arguments
    # robot = robot instance whose graph we have to analize
    
    graphReversed = robot.graph.reverse()
    predecessors = nx.dfs_successors(graphReversed,robot.endNodeCounter)
    
    path = []
    for key, value in predecessors.items():
        path.append(key)
        
    path.append(value[0])
    path.reverse()
    robot.paths.append(path)
    robot.path = robot.graph.subgraph(path)

def leastCostGoalSet(robot, debug):
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
    
    if debug:
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

def updateGoalSet(robots, timeInterval, debug):
    """Check if goal set is still valid after rewiring"""
    # Input arguments
    # robots = all robot instances
    # timeInterval = arrival time interval for no delay constraint
        
    dictNodes = nx.get_node_attributes(robots[0].graph,'goalSet')
    goalNodes = list(dictNodes.keys())
    
    if not goalNodes:
        pass    
    goalNodes = np.asarray(goalNodes)
        
    for node in goalNodes:
        times = []
        
        for r in range(0, len(robots)): 
            times.append(robots[r].graph.nodes[node]['t'])
        
        times = np.asarray(times)
        timeDiff = np.abs(times - times[0])
        
        if not all(y <= timeInterval for y in timeDiff):
            
            if debug:
                print('Update goal set')
                print(timeDiff)
                print('Deleted goalsetNode')
                print(node)
            
            for r in range(0, len(robots)): 
                del robots[r].graph.nodes[node]['goalSet']                
    
    
def calculateGoalSet(robots, commRadius, timeInterval):
    """check if the new nodes are close to each other and if the time is right so that they belong to the goal set"""
    # Input arguments
    # robots = all robot instances
    # commRadius = communication range of the robots
    
    positions = []
    times = []
    for r in range(0, len(robots)): 
        positions.append(robots[r].graph.nodes[robots[r].vnewIdx]['pos'])
        times.append(robots[r].graph.nodes[robots[r].vnewIdx]['t'])
    
    positions = np.asarray(positions)
    times = np.asarray(times)
    
    timeDiff = np.abs(times - times[0])
    normDist = np.sqrt(np.sum((positions[0] - positions)**2, axis=1))

    if all(x <= commRadius for x in normDist) and all(y <= timeInterval for y in timeDiff):
        for r in range(0, len(robots)): 
            robots[r].graph.nodes[robots[r].vnewIdx]['goalSet'] = True

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

def rewireGraph(robots, uMax, timeInterval, debug):
    # TODO: Find out what this min t = max t means

    """Check if there is a node in setVnear which has a smaller cost as child from vnew than previous"""
    # Input arguments
    # robots = which robot graph that we are analyzing
    # uMax = maximum velocity of robot for cost computation
    # timeInterval = interval for meeting time arrival
    # debug = show values
    
    rewired = False
    
    for r in range(0, len(robots)): 
        robot = robots[r]
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
                if list(robot.graph.pred[setVnear[n]]) == []:
                    continue
                predecessor = list(robot.graph.pred[setVnear[n]])[-1]
                robot.graph.remove_edge(predecessor,setVnear[n])
                robot.graph.add_edge(robot.vnewIdx,setVnear[n], weight = nearEdgeCost)
                robot.graph.nodes[setVnear[n]]['t'] = nearTotalCost
                updateSuccessorNodes(robot, setVnear[n], uMax)
                rewired = True
                
    if rewired:
        updateGoalSet(robots, timeInterval, debug)
            
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
    
    return round(nearTotalCost,1), round(nearEdgeCost,1) 

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
    
def steer(robots, uMax, epsilon):
    """Steer towards vrand but only as much as allowed by the dynamics"""
    # Input Arguments
    # robots = robot classes
    # uMax = maximal velocity in pixel/second
    # epsilon = maximum allowed distance traveled
    
    #find minTimes for nearest nodes of all robots
    minTimes = []
    for r in range(0, len(robots)): 
        nearestNodeIdx = robots[r].nearestNodeIdx
        graphDict = robots[r].graph.nodes[nearestNodeIdx]
        
        vnearest = list(graphDict.values())
        nearestTime = np.asarray(vnearest[1])
        
        minTimes.append(nearestTime)
    
    # TODO: Add different motion model
    #steer towards vrand
    for r in range(0, len(robots)):   
        nearestNodeIdx = robots[r].nearestNodeIdx
        graphDict = robots[r].graph.nodes[nearestNodeIdx]
        
        vnearest = list(graphDict.values())
        
        vrand = robots[r].vrand
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
        robots[r].vnew = np.around(vnew)
        robots[r].vnewCost = round(travelTimeVnew,1)
        robots[r].totalTime = round(totalTimeVnew,1)

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