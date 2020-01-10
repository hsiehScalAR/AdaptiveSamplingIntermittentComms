#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:21:38 2019

@author: hannes
"""

#General imports
import numpy as np
import scipy.io as sio

def setupMatlabFileMeasurementData(discetization, invert=True):
    mat = sio.loadmat('Data/FTLEDoubleGyre.mat')
    data = mat['FTLE']
    if invert:
        return (data + 1)*-1 +4
    else:
        return data + 1
    
def loadMeshFiles(sensorPeriod, correctTimeSteps = False):
    pathname_mesh = 'Data/meshfiles/600x600_mesh.mat'
    pathname_node = 'Data/meshfiles/600x600_node_soln_fine.mat'
    pathname_times = 'Data/meshfiles/600x600_node_soln_fine_times.mat'

    mat_contents = sio.loadmat(pathname_mesh)
    meshNodes = mat_contents['MeshNodes']
    meshNodes = np.rint(meshNodes)

    mat_contents = sio.loadmat(pathname_node)
    nodeSoln = mat_contents['NodalSolution']
    
    mat_contents = sio.loadmat(pathname_times)
    timeValues = mat_contents['T'].T.tolist()[1:]
    timeValues = np.around(np.array(timeValues).T,1)
    
    syncIdx = []
    radius = 10
    scaling = 20
    
    measurementGroundTruthList = []

    if correctTimeSteps:
        maxTime = np.int(timeValues[0][-1])
        
        sampling = sensorPeriod

        for _ in range(0,np.int(maxTime/sensorPeriod)):
            idx = np.where(timeValues == sampling)
            syncIdx.append(idx[1][0])
            sampling = round(sampling+sensorPeriod,1)

        for tIdx in syncIdx:
            nodeSol = nodeSoln[:,tIdx]
            data = np.zeros([600,600])
            
            for idx in range(0,meshNodes.shape[1]):
                posx = np.int(meshNodes[:,idx][0])
                posy = np.int(meshNodes[:,idx][1])
                data[posx-radius:posx+radius,posy-radius:posy+radius] = nodeSol[idx]
            data = data/scaling    
            # data = (data - np.min(data)) / (np.max(data) - np.min(data))
            measurementGroundTruthList.append(data)
    else:
        lag = 3
        skip = 2
        maxTime = timeValues.shape[1]
        
        sampling = sensorPeriod

        for tIdx in range(0,np.int(maxTime),skip):
            nodeSol = nodeSoln[:,tIdx]
            data = np.zeros([600,600])
            
            for idx in range(0,meshNodes.shape[1]):
                posx = np.int(meshNodes[:,idx][0])
                posy = np.int(meshNodes[:,idx][1])
                data[posx-radius:posx+radius,posy-radius:posy+radius] = nodeSol[idx]
            data = data/scaling    
            # data = (data - np.min(data)) / (np.max(data) - np.min(data))
            for _ in range(0,lag):
                measurementGroundTruthList.append(data)

            
        maxTime = np.int(timeValues.shape[1]/skip)*sensorPeriod*lag
    return measurementGroundTruthList, maxTime
    
def getSetup(case):
    """returns the setup for the robot teams based on the case"""
    #Input arguments
    # case = which case we are treating
    
    #robot i belongs to team j
    positions = None
    if case == 1:
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
        
        positions = np.array([[0, 0],
                              [0,300],
                              [0, 599],
                              [300, 599],
                              [599, 599],
                              [599, 300],
                              [599, 0],
                              [300, 0],])
    
    elif case == 2:
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
        
        positions = np.array([[0, 0],
                              [0,300],
                              [0, 599],
                              [300, 599],
                              [599, 599],
                              [599, 300],
                              [599, 0],
                              [300, 0],])
    elif case == 3:
        numTeams = 4
        numRobots = 4
        robTeams = np.array([[1, 0, 0, 1],
                             [1, 1, 0, 0],
                             [0, 1, 1, 0],
                             [0, 0, 1, 1],])
    
        positions = np.array([[0, 0],
                             [0, 599],
                             [599, 0],
                             [599, 599],])
    
    else:
        exit()
        
    return numTeams, numRobots, robTeams, positions