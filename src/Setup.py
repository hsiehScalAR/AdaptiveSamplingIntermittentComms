#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:21:38 2019

@author: hannes
"""

#General imports
import numpy as np
import scipy.io as sio

def loadBariumCloud(sensorPeriod):
    """Gets data from the barium cloud

    Input arguments:
    sensorPeriod = sampling time
    """

    loadPath = '/home/hannes/MasterThesisCode/AdaptiveSamplingIntermittentComms/src/Data/BariumCloudImages/Processed/'

    npzFile = np.load(loadPath + 'BariumCloudDataBig.npz')

    data = npzFile['data']

    newData = []
    maxIteration = np.int(np.round(len(data)/sensorPeriod))
    sample = 0
    for t in range(0,maxIteration):
        if (t % ((maxIteration)/len(data))) == 0:
            sample += 1
            if sample == 100:
                sample = 99
        newData.append(data[sample])

    return newData, len(data)

def setupMatlabFileMeasurementData(invert=True):
    """Gets data from an FTLE file

    Input arguments:
    invert = invert the data values
    """

    mat = sio.loadmat('Data/FTLEDoubleGyre.mat')
    data = mat['FTLE']
    if invert:
        return (data + 1)*-1 +4
    else:
        return data + 1
    
def loadMeshFiles(sensorPeriod, correctTimeSteps = False):
    """Gets data from the mesh files

    Input arguments:
    sensorPeriod = sampling time to match data
    correctTimeSteps = if we should use sampling time to get data or just avery frame
    """

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
            for _ in range(0,lag):
                measurementGroundTruthList.append(data)

            
        maxTime = np.int(timeValues.shape[1]/skip)*sensorPeriod*lag
    return measurementGroundTruthList, maxTime

def new_map_practice(maxTime):
    """
    Practice to get a sense for what the ground truth map looked like and how it impacted the program
    
    This is a fixed map size which has a variable number of time steps but is a fixed data stream

    XXX: Right now this code is dependent on a fixed 600x600 grid size 
         Need to work to figure out how much might need to change for this
    """
    measurementGroundTruthList = []
    data = np.zeros([600, 600]) #np.random.rand(600, 600)
    for i in range(100):
        for j in range(100):
            data[200 + i, 200 + j] = 1.0

    skip = 2
    
    for tIdx in range(0, np.int(maxTime), skip):
        
        measurementGroundTruthList.append(data)
    
    return(measurementGroundTruthList, maxTime)

def new_dynamic_process(num_particles, max_time):
    # XXX: Future problem how does the pod stuff deal with obstacles? (How should it be relfelcted in the map?)
    # XXX: Plumb the simple agents into UNITY along with the solved for paths (The entire process can be post-processed

    # XXX: What is the "Behavior" of the particles: e.g. Boids, potential methods etc? 
    
    # Setup the environment (This would be where I would put obstacles of a map) 
    measurementGroundTruthList = []
    
    # This is an nxn discritized grid (we will need a scaling factor to properly convert the agent pos to the grid)
    data = np.zeros([600, 600])
    skip = 1
    dt = 0.1

    # Setup the agents (simpl) 
    pos = np.random.rand(num_particles, 2)
    vel = np.random.rand(num_particles, 2)

    scale = 0.1 # in x and y so uniform grid XXX: This does not necessarily need to be true

    pos_in_grid = np.zeros((num_particles, 2))
    pos_in_grid[:,0] = pos[:,0] // scale
    pos_in_grid[:,1] = pos[:,1] // scale

    
    idx = pos_in_grid.astype(int)
    idx = idx[idx[:,0] < 600]
    idx = idx[idx[:,1] < 600]

    # Put data in grid world coordinates
    data[idx[:,0], idx[:,1]] = 1.0
 
    
    for i in range(0, np.int(max_time), skip):
        
        if i == 0:
            # update the ground truth list 
            measurementGroundTruthList.append(data)
            continue
        
        # Step the agents through an update (single point integrator / something easy)
        new_pos = pos + vel * dt
        
        # Put the robots new pose in the environment
        pos_in_grid[:,0] = new_pos[:,0] // scale
        pos_in_grid[:,1] = new_pos[:,1] // scale

        idx = pos_in_grid.astype(int)
        idx = idx[idx[:,0] < 600]
        idx = idx[idx[:,1] < 600]
        #print(idx)
        data = np.zeros((600, 600))
        data[idx[:,0], idx[:,1]] = 1.0

        # update the ground truth list 
        measurementGroundTruthList.append(data)

        pos = new_pos
        
    
    return(measurementGroundTruthList, max_time)
    

def getSetup(case, pod, heterogeneous, discretization):
    """Returns the setup for the robot teams based on the case

    Input arguments:
    case = which case we are treating
    pod = if we are using pod or not
    heterogeneous = if heterogeneous setup
    discretization = workspace
    """
    
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

        uMax = np.array([80,80,80,80,80,80,80,80])
        sensorPeriod = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
        if pod:
            sensingRange = np.array([20,20,20,20,20,20,20,20])
        else:
            sensingRange = np.array([0,0,0,0,0,0,0,0])
        
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

        uMax = np.array([80,80,80,80,80,80,80,80])
        sensorPeriod = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
        if pod:
            sensingRange = np.array([20,20,20,20,20,20,20,20])
        else:
            sensingRange = np.array([0,0,0,0,0,0,0,0])

    elif case == 3:
        if heterogeneous:

            numTeams = 3
            numRobots = 4
            robTeams = np.array([[1, 1, 1],
                                 [1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1],])
        
            positions = np.array([[0, 0],
                                [0, discretization[1]-1],
                                [discretization[0]-1, 0],
                                [discretization[0]-1, discretization[1]-1],])

            uMax = np.array([160, 80, 80, 80])
            commRange = np.array([100, 3, 3, 3])
            if pod:
                sensingRange = np.array([ 20, 20, 20, 20, 20])
                sensorPeriod = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
            else:
                sensingRange = np.array([0, 0, 0, 0, 0])
                sensorPeriod = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        else:
            numTeams = 4
            numRobots = 4
            robTeams = np.array([[1, 0, 0, 1],
                                 [1, 1, 0, 0],
                                 [0, 1, 1, 0],
                                 [0, 0, 1, 1],])
        
            positions = np.array([[0, 0],
                                [0, discretization[1]-1],
                                [discretization[0]-1, 0],
                                [discretization[0]-1, discretization[1]-1],])
    
            uMax = np.array([80,80,80,80])
            sensorPeriod = np.array([0.1,0.1,0.1,0.1])
            commRange = np.array([3,3,3,3])
            if pod:
                sensingRange = np.array([20,20,20,20])
            else:
                sensingRange = np.array([0,0,0,0])

    elif case == 4:
        if heterogeneous:
            numTeams = 9
            numRobots = 10
            robTeams = np.array([   [0, 0, 0, 0, 1, 0, 1, 0, 1],
                                    [0, 0, 0, 0, 0, 1, 0, 1, 1],
                                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 1, 0, 0],
                                    ])
        
            positions = np.array([  [250, 300],
                                    [350, 300],
                                    [0, 549],
                                    [50, 599],
                                    [549, 599],
                                    [599, 549],
                                    [599, 50],
                                    [549, 0],
                                    [50, 0],
                                    [0, 50],
                                    ])

            uMax = np.array([40, 40, 40, 40, 40, 40, 40, 40, 40, 40])
            commRange = np.array([100, 100, 3, 3, 3, 3, 3, 3, 3, 3])
            if pod:
                sensingRange = np.array([ 5, 5, 20, 20, 20, 20, 20, 20, 20, 20])
                sensorPeriod = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
            else:
                sensingRange = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                sensorPeriod = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        else: 
            print('**********************************************************************\n')
            print('WARNING: Setup for heterogeneous robots chosen but not heterogeneous as keyword, exiting application')
            print('**********************************************************************\n')
            exit()
    
    elif case == 5:
        if heterogeneous:
            numTeams = 4
            numRobots = 5
            robTeams = np.array([   [1, 1, 1, 1],
                                    [1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1],
                                    ])
        
            positions = np.array([  [300, 300],
                                    [0, 0],
                                    [0, 599],
                                    [599, 0],
                                    [599, 599],
                                    ])

            uMax = np.array([160, 80, 80, 80, 80])
            commRange = np.array([100, 3, 3, 3, 3])
            if pod:
                sensingRange = np.array([ 20, 20, 20, 20, 20])
                sensorPeriod = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
            else:
                sensingRange = np.array([0, 0, 0, 0, 0])
                sensorPeriod = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        else: 
            print('**********************************************************************\n')
            print('WARNING: Setup for heterogeneous robots chosen but not heterogeneous as keyword, exiting application')
            print('**********************************************************************\n')
            exit()

    elif case == 6:
    
        numTeams = 5
        numRobots = 5
        robTeams = np.array([   [1, 0, 0, 0, 1],
                                [1, 1, 0, 0, 0],
                                [0, 1, 1, 0, 0],
                                [0, 0, 1, 1, 0],
                                [0, 0, 0, 1, 1],
                                ])
    
        positions = np.array([  [0, 0],
                                [0, 599],
                                [599, 0],
                                [599, 599],
                                [300, 300],
                                ])

        uMax = np.array([80, 80, 80, 80, 80])
        commRange = np.array([3, 3, 3, 3, 3])
        if pod:
            sensingRange = np.array([ 20, 20, 20, 20, 20])
            sensorPeriod = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        else:
            sensingRange = np.array([0, 0, 0, 0, 0])
            sensorPeriod = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

    else:
        print('**********************************************************************\n')
        print('WARNING: Nonvalid setup chosen, exiting application')
        print('**********************************************************************\n')
        exit()
        
    return numTeams, numRobots, robTeams, positions, uMax, sensingRange, sensorPeriod, commRange
