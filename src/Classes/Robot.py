#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:48:29 2019

@author: hannes
"""

#General imports
import numpy as np
import networkx as nx
#TODO see if I need itemgetter
from operator import itemgetter

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

        #Path variables
        self.paths = []
        self.scheduleCounter = 0
        self.atEndLocation = False
        self.currentLocation = np.array([0,0])
        self.pathCounter = 0
        self.trajectory = []
        
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
        
        Robot.objs.append(self)
        Robot.discretization = discretization
        
    def composeGraphs(self):
        self.totalGraph = nx.compose(self.totalGraph,self.graph)
        self.totalPath = nx.compose(self.totalPath,self.path)
    
    def initializeGraph(self):
        self.graph = nx.DiGraph()
    
    def addNode(self, firstTime = False):
        """Add new node with pos and total time attributes and edge with edge travel time cost to graph based on self variables"""
        # Input arguments:
        # FirstTime = bool that decides if we should do an edge or not
        
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




