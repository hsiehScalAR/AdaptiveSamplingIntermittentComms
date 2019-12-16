#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:21:38 2019

@author: hannes
"""

#General imports
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def setupMatlabFileMeasurementData(discetization, invert=True):
    mat = sio.loadmat('Data/FTLEDoubleGyre.mat')
    data = mat['FTLE']
    if invert:
        return (data + 1)*-1 +4
    else:
        return data + 1
    
def loadMeshFiles():
    pathname_mesh = 'Data/meshfiles/600x600_mesh.mat'
    pathname_node = 'Data/meshfiles/600x600_node_soln_fine.mat'

    mat_contents = sio.loadmat(pathname_mesh)
    meshNodes = mat_contents['MeshNodes']
    meshNodes = np.rint(meshNodes)

    mat_contents = sio.loadmat(pathname_node)
    nodeSoln = mat_contents['NodalSolution']
    
    nodeSol1 = nodeSoln[:,1500]
    data = np.zeros([600,600])
    radius = 10    
    for idx in range(0,meshNodes.shape[1]):
        posx = np.int(meshNodes[:,idx][0])
        posy = np.int(meshNodes[:,idx][1])
        data[posx-radius:posx+radius,posy-radius:posy+radius] = nodeSol1[idx]
        
    return data/20
    
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