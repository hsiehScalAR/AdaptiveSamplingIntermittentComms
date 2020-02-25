#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:48:29 2019

@author: hannes
"""

#General imports
import numpy as np
import networkx as nx

#Personal imports
from Classes.GaussianProcess import GaussianProcess
from Classes.ReducedOrderModel import ReducedOrderModel

class Robot:
    
    objs = []  # Registrar keeps all attributes of class

    def __init__(self, ID, teams, schedule, discretization, optPath, optPoint, spatiotemporal, specialKernel, pod, logFile, folder):
        """Initializer of robot class

        Input arguments:
        ID = robot number
        teams = to which team each robot belongs
        schedule = schedule of teams
        discretization = grid space 
        the rest are global variables from TestIntermittent
        """

        self.ID = ID
        self.teams = teams
        self.schedule = schedule
        self.activeLocations = {}  # Store active location as indexed by (timeStart, timeEnd): locations
        self.sensorData = {}  # Store data as (timeStart, timeEnd): data
        self.eigenData = {}
        self.mapping = np.zeros([discretization[0],discretization[1],2])
        self.mappingGroundTruth = np.zeros_like([discretization[0],discretization[1]])
        self.sensingRange = 0
        self.numbMeasurements = 0
        self.measurementRangeX = np.array([self.sensingRange, self.sensingRange])
        self.measurementRangeY = np.array([self.sensingRange, self.sensingRange])
        self.uMax = 0
        self.sensorPeriod = 0.1
        self.deltaT = 0.1
        
        self.optPath = optPath
        self.optPoint = optPoint

        self.expectedMeasurement = np.zeros([discretization[0],discretization[1]])
        self.expectedVariance = np.ones([discretization[0],discretization[1]])
        self.currentTime = 0
        
        #Path variables
        self.paths = []
        self.scheduleCounter = 0
        self.atEndLocation = False
        self.currentLocation = np.array([0,0])
        self.pathCounter = 0
        self.trajectory = []
        self.meetings = []
        
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
        self.vnewInformation = 0
        
        self.totalTime = 0
        
        self.setVnear = []
    
        self.startTotalTime = 0
        self.startNodeCounter = 0
        self.startLocation = np.array([0, 0])
        
        self.endTotalTime = 0
        self.endNodeCounter = 0
        self.endLocation = np.array([0, 0])
        
        # Model variable
        if pod:
            self.model = ReducedOrderModel(spatiotemporal, specialKernel, logFile, folder)
        else:
            self.model = GaussianProcess(spatiotemporal, specialKernel, logFile, folder)

        Robot.objs.append(self)
        Robot.discretization = discretization
        
    def composeGraphs(self):
        """Adds the graphs of different epoch together

        No input arguments
        """

        self.totalGraph = nx.compose(self.totalGraph,self.graph)
        self.totalPath = nx.compose(self.totalPath,self.path)
    
    def initializeGraph(self):
        """Initializer for nx graphs

        No input arguments
        """

        self.graph = nx.DiGraph()
    
    def addNode(self, firstTime = False):
        """Add new node with pos and total time attributes and edge with edge travel time cost to graph based on self variables

        Input arguments:
        FirstTime = bool that decides if we should do an edge or not
        """

        self.graph.add_node(self.nodeCounter, pos = self.vnew, t = self.totalTime, informationGain = self.vnewInformation)
        self.vnewIdx = self.nodeCounter
        if self.nodeCounter != 0 and firstTime == False:
            self.graph.add_edge(self.nearestNodeIdx,self.nodeCounter, weight = self.vnewCost)
        self.nodeCounter += 1

    def createMap(self,newData, newDataTime, currentLocations):
        """creates a measurement map in the grid space without time reference

        Input arguments:
        newData = new measurement for single robot
        newDataTime = time of new measurement for single robot
        currentLocations = sensing location of single robot
        """
        
        x = np.int(self.currentLocation[0])
        y = np.int(self.currentLocation[1])
        if self.sensingRange < 1:
            self.mapping[x, y, 0] = newData
            self.mapping[x, y, 1] = newDataTime
    
        else:
            self.mapping[x-self.measurementRangeX[0]:x+self.measurementRangeX[1], 
                         y-self.measurementRangeY[0]:y+self.measurementRangeY[1], 0] = newData
            self.mapping[x-self.measurementRangeX[0]:x+self.measurementRangeX[1], 
                         y-self.measurementRangeY[0]:y+self.measurementRangeY[1], 1] = newDataTime




