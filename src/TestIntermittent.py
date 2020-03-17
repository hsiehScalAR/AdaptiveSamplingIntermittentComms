#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:48:29 2019

@author: hannes
"""

#General imports
import numpy as np
import matplotlib.pyplot as plt
import os, shutil
from skimage.measure import compare_ssim as ssim
from scipy.spatial import procrustes

#Personal imports
from Classes.Scheduler import Schedule
from Classes.Robot import Robot
from Setup import getSetup, loadMeshFiles, loadBariumCloud

from Utilities.ControllerUtilities import moveAlongPath, communicateToTeam, checkMeetingLocation, measurement
from Utilities.VisualizationUtilities import (plotMeasurement, plotMeetingGraphs, plotMeetingPaths, 
                                              plotTrajectory, plotTrajectoryAnimation,
                                              plotTrajectoryOverlayGroundTruth, plotDye)
from Utilities.PathPlanningUtilities import (sampleVrand, findNearestNode, steer, buildSetVnear, 
                                             extendGraph, rewireGraph, calculateGoalSet, 
                                             checkGoalSet, leastCostGoalSet, getPath, 
                                             getInformationGainAlongPath, sampleNPoints)
from Utilities.LogUtilities import LogFile

def main():
    """
    Main test loop
    
    No inputs 
    """

    """Create Measurement Data"""

    if BARIUM:
        measurementGroundTruthList, maxTime = loadBariumCloud(DELTAT)
    else:
        measurementGroundTruthList, maxTime = loadMeshFiles(DELTAT, CORRECTTIMESTEP)

    plotDye(measurementGroundTruthList[50],measurementGroundTruthList[500],measurementGroundTruthList[999], FOLDER+'/')
    
    print('**********************************************************************\n')
    print('Max allowed time is: %.1f length of data: %.1f\n' %(maxTime,len(measurementGroundTruthList)))
    print('**********************************************************************\n')

    if maxTime < TOTALTIME:
        print('**********************************************************************\n')
        print('WARNING: Given Total Time is bigger than available data (maxTime = %d), terminating application' %maxTime)
        print('**********************************************************************\n')
        exit()

    if STATIONARY:
        measurementGroundTruth = measurementGroundTruthList[np.int(STATIONARYTIME/DELTAT)]
    else:
        measurementGroundTruth = measurementGroundTruthList[0]

           
    """create robot to team correspondence"""
    numTeams, numRobots, robTeams, positions, uMax, sensingRange, sensorPeriod, commRange = getSetup(CASE, POD, HETEROGENEOUS, DISCRETIZATION)
    
    if COMMNOISE != 0:
        commRange = commRange + COMMNOISE*np.random.randn(commRange.shape[0])
        commRange = np.where(commRange < 3, 3, commRange)
        print(commRange)

    if FULLYCONNECTED:
        commRange = np.ones_like(commRange)*COMMRANGE
    """Write parameters to new logfile"""
    parameters = {
                'RANDINT        ': RANDINT,
                'TOTALTIME      ': TOTALTIME,
                'CASE           ': CASE,
                'HETEROGENEOUS  ': HETEROGENEOUS,
                'FULLYCONNECTED ': FULLYCONNECTED,
                'CORRECTTIMESTEP': CORRECTTIMESTEP,
                'POD            ': POD,
                'MODEL          ': MODEL,
                'OPTPATH        ': OPTPATH,
                'OPTPOINT       ': OPTPOINT,
                'SPATIOTEMPORAL ': SPATIOTEMPORAL,
                'SPECIALKERNEL  ': SPECIALKERNEL,
                'STATIONARY     ': STATIONARY,
                'STATIONARYTIME ': STATIONARYTIME,
                'PREDICTIVETIME ': PREDICTIVETIME,
                'SENSINGRANGE   ': sensingRange,
                'COMMRANGE      ': commRange,
                'TIMEINTERVAL   ': TIMEINTERVAL,
                'SENSORPERIOD   ': sensorPeriod,
                'UMAX           ': uMax
                }
    logFile = LogFile(LOGFILE,FOLDER +'/')
    logFile.writeParameters(**parameters)

    """Variables"""
    if isinstance(positions, np.ndarray):
        locations = positions
    else:
        locations = randomStartingPositions(numRobots) #locations of robots

    """Initialize schedules and robots"""
    schedule, teams, commPeriod = initializeScheduler(numRobots, numTeams, robTeams)
    
    print('Schedule: ')
    print(schedule)

    for idx, t in enumerate(teams):
        teams[idx] = [t,True]

    robots = initializeRobots(numRobots, teams, schedule, logFile)

    """create the initial plans for all periods"""
    initialTime = 0
    
    print('*************************************************************')
    print('Initializing Environment\n')

    for r in range(0,numRobots):
        robots[r].vnew = locations[r]
        robots[r].currentLocation = locations[r]
        robots[r].totalTime = initialTime
        robots[r].sensingRange = sensingRange[r]
        robots[r].uMax = uMax[r]
        robots[r].sensorPeriod = sensorPeriod[r]
        robots[r].mappingGroundTruth = measurementGroundTruth
        robots[r].commRange = commRange[r]
        
        """Initialize models"""
        meas, measTime = measurement(robots[r])
        robots[r].createMap(meas, measTime, robots[r].currentLocation)
        if MODEL:
            robots[r].model.initialize(robots[r])

    print('Environment Initialized')
    print('*************************************************************\n')

    print('*************************************************************')
    print('Initializing Paths\n')
    
    teamsInAllEpochs = []

    for period in range(0,schedule.shape[1]):
        teamsInEpoch = []
        teamsDone = np.zeros(len(teams))
        print('Period: %d'%period)
        idx = 0
        
        #find out which team has a meeting event at period k=0
        robNoMeeting = []
        for team in schedule[:, period]:
            if team < 0:
                print('robot without meeting event')
                robNoMeeting.append(robots[idx])
                
            if teamsDone[team] or team < 0:                
                idx += 1
                continue

            robs = []
            commRangeList = []
            for r in teams[team][0][0]:
                robs.append(robots[r-1])
                commRangeList.append(robots[r-1].commRange)

            updatePaths(robs, max(commRangeList), teams[team][1])
            teamsDone[team] = True
            
            idx += 1
            teamsInEpoch.append(teams[team])

        if len(robNoMeeting) != 0:
            print('No meeting planned')
            updatePaths(robNoMeeting, max(commRangeList), meeting=False)
            
            for _, mRob in enumerate(robNoMeeting):
                newTeam = []
                newTeam.append(mRob.ID+1)
                teams.append([np.asarray([newTeam]),False])
            
                teamsInEpoch.append(teams[-1])

        for r in range(0,numRobots):
            robots[r].composeGraphs() 
        teamsInAllEpochs.append(teamsInEpoch)
    
    print('Paths Initialized')    
    print('*************************************************************\n')

    """Control loop"""
    print('*************************************************************')
    print('Starting ControlLoop\n')

    currentTime = initialTime

    modelEstimates = []
    modelEstimates.append([np.zeros([DISCRETIZATION[0],DISCRETIZATION[1]]),0])
    scheduleCheck = np.zeros(numRobots)
    currentEpoch = 0
    epochCounter = 0

    for t in range(0,np.int(TOTALTIME/DELTAT)):
        if not STATIONARY:
            for r in range(0,numRobots):
                robots[r].mappingGroundTruth = measurementGroundTruthList[t]
            
        currentTime = update(currentTime, robots, teamsInAllEpochs[currentEpoch], commPeriod, modelEstimates, schedule, numTeams, scheduleCheck, currentEpoch)
        
        if np.all(scheduleCheck):
            epochCounter += 1
            currentEpoch = epochCounter % commPeriod
            print('-------------------------------------------------------------')
            print('Current Epoch: %d' %currentEpoch)
            print('-------------------------------------------------------------')
            scheduleCheck[:] = 0

            

    print('ControlLoop Finished')
    print('*************************************************************\n')
    
    print('*************************************************************')
    print('Starting Plotting\n')

    if DEBUG:
        plotMeasurement(measurementGroundTruth, 'Ground truth measurement map', FOLDER+'/')
        # subplot = 1
        # team = 0
        # for r in teams:
        #     r = np.asarray(r[0]) -1
        #     plotMeetingGraphs(robots, r, team,  FOLDER+'/',subplot, len(teams))
        #     plotMeetingPaths(robots, r, team,  FOLDER+'/', subplot, len(teams))
        #     subplot += 1
        #     team += 1
        
        plotTrajectory(robots, FOLDER+'/')
        totalMap = robots[0].mapping[:,:,0]
        plotMeasurement(totalMap, 'Measurements of robots after communication events', FOLDER+'/')
   
    if MODEL:
        # plotTrajectoryOverlayGroundTruth(robots,0, FOLDER+'/')

        for r in range(0,numRobots):
            robots[r].model.update(robots[r])
            robots[r].model.plot(robots[r])
            if r == 0:
                if ANIMATION:
                    plotTrajectoryAnimation(robots, measurementGroundTruthList, modelEstimates, FOLDER+'/')
            if PREDICTIVETIME != None:
                
                if PREDICTIVETIME >= maxTime:
                    predictiveTime = maxTime-DELTAT
                else:
                    predictiveTime = np.int(PREDICTIVETIME/DELTAT)
                robots[r].currentTime = predictiveTime*DELTAT
                robots[r].mappingGroundTruth = measurementGroundTruthList[predictiveTime]
                robots[r].model.plot(robots[r], robots[r].currentTime)
        
        errorCalculation(robots, logFile)
    
    print('Plotting Finished')
    print('*************************************************************')

def update(currentTime, robots, teams, commPeriod, modelEstimates, schedule, numTeams, scheduleCheck, currentEpoch):
    """
    Update procedure of intermittent communication

    Input arguments:
    currentTime = current time of the execution
    robots = instances of the robots that are to be moved
    teams = teams of the robots
    commPeriod = how many schedules there are
    modelEstimates = list of model updates and time
    schedule = schedule of meeting events
    numTeams = number of teams
    """

    atEndPoint = np.zeros(len(robots))
    
    for i, rob in enumerate(robots):
        atEndPoint[i] = moveAlongPath(rob, DELTAT)

    currentTime += DELTAT
    
    for idx, team in enumerate(teams):
        meeting = team[1]
        team = team[0].copy()

        # scheduleCounter = []

        # for r in team[0]:
        #     scheduleCounter.append(robots[r-1].scheduleCounter)
        # scheduleCounter = np.asarray(scheduleCounter)
        # sameSchedule = scheduleCounter - scheduleCounter[0]

        # correctTeam = False
        # if not np.any(sameSchedule):
        #     # currentEpoch = scheduleCounter[0] % commPeriod
        #     for t in schedule[:,currentEpoch]:
        #         if t == idx:
        #             correctTeam = True

        # correctEpoch = False
        # if currentEpoch == scheduleCounter[0] % commPeriod:
        #     correctEpoch = True

        # if meeting:
        #     condition = np.all(atEndPoint[team-1]) and not np.any(sameSchedule) and correctTeam and correctEpoch
        # else:
        #     condition = np.all(atEndPoint[team-1]) and idx >= numTeams and not np.any(sameSchedule) and correctEpoch

        meetingHappend = False
        for r in team[0]:
            if scheduleCheck[robots[r-1].ID] == 1:
                meetingHappend = True


        if np.all(atEndPoint[team-1]) and not meetingHappend:         
            
            currentLocations = []
            commRangeList = []
            for r in team[0]: 
                currentLocations.append(robots[r-1].currentLocation)
                commRangeList.append(robots[r-1].commRange)
            currentLocations = np.asarray(currentLocations)
            if meeting:
                if not checkMeetingLocation(currentLocations, max(commRangeList)):
                    continue
            
                robs = []
                i = 0         
                print('Com Meeting:')
                for r in team[0]:
                    print(robots[r-1].ID)   
                    robots[r-1].meetings.append([currentLocations[i], idx])
                    i += 1             
                    robots[r-1].scheduleCounter += 1
                    robots[r-1].atEndLocation = False
                    
                    robots[r-1].endTotalTime  = currentTime
                    robs.append(robots[r-1])
                    scheduleCheck[robots[r-1].ID] = 1

                modelEstimates.append([communicateToTeam(robs, MODEL, POD), currentTime])
                # modelEstimates.append([communicateToTeam(robots, MODEL, POD), currentTime])
            else:
                robs = []
                print('Comm No meeting')
                for r in team[0]:             
                    print(robots[r-1].ID) 
                    robots[r-1].scheduleCounter += 1
                    robots[r-1].atEndLocation = False
                    
                    robots[r-1].endTotalTime  = currentTime
                    robs.append(robots[r-1])
                    scheduleCheck[robots[r-1].ID] = 1
            
            print('Updating Paths')
            updatePaths(robs,max(commRangeList),meeting)
            print('Paths Updated\n')
            
            for r in team[0]:
                robots[r-1].composeGraphs()

    return round(currentTime,1)

def updatePaths(robots, commRange, meeting=True):
    """
    Update procedure of intermittent communication

    Input arguments:
    robots = instances of the robots that are to be moved
    commRange = maximum communication range of robots in team
    """

    #add node v0 to list of nodes for each robot       
    for r in range(0, len(robots)):        
        #make last node equal to the first of new period
        if robots[r].nodeCounter > 0:
            robots[r].nodeCounter -= 1

        robots[r].startNodeCounter = robots[r].nodeCounter
        robots[r].startLocation = robots[r].vnew
        robots[r].startTotalTime = robots[r].endTotalTime 
        robots[r].nearestNodeIdx = robots[r].endNodeCounter
        
        robots[r].initializeGraph()
        robots[r].addNode(firstTime = True)
             
    connected = False
    counter = 0
    while not connected:    
        #sample new nodes and create path
        distribution = 'uniform'
        rangeSamples = DISCRETIZATION
        
        for sample in range(0,TOTALSAMPLES):
            if sample == RANDOMSAMPLESMAX-1:
                mean = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
                stdDev = 4*commRange*commRange*np.identity(DIMENSION)
                distribution = 'gaussian'
                rangeSamples = [mean,stdDev]
            
            if sample >= RANDOMSAMPLESMAX:
                vrand = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
            
            #find which robot is in team and get nearest nodes and new random samples for them
            for r in range(0, len(robots)):       
                
                if distribution == 'uniform':
                    #looking for either optimal position or path based on model modeling
                    if OPTPATH:              
                        maxVariance = 0
                        vrand = np.array([0, 0])
                        nearNIdx = 0
                        
                        for _ in range(0,10):
                            point = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
                            #find nearest node to random sample
                            nearNIdx = findNearestNode(robots[r].graph,point)
                            var = getInformationGainAlongPath(robots[r], point, nearNIdx, EPSILON)
                            
                            if var >= maxVariance:
                                maxVariance = var
                                vrand = point                                                          
                    elif OPTPOINT:
                        vrand = sampleNPoints(robots[r], DISCRETIZATION, rangeSamples, distribution)
                    else: 
                        vrand = sampleVrand(DISCRETIZATION, rangeSamples, distribution)
                        
                    robots[r].vrand = vrand
                    nearestNodeIdx = findNearestNode(robots[r].graph,vrand)
                            
                    robots[r].nearestNodeIdx = nearestNodeIdx
                    
                else:
                    robots[r].vrand = vrand
                    
                    #find nearest node to random sample
                    nearestNodeIdx = findNearestNode(robots[r].graph,vrand)
                    robots[r].nearestNodeIdx = nearestNodeIdx
            
            #find new node towards max distance to random sample and incorporate time delay, 
            #that is why it is outside of previous loop since we need all the nearest nodes from the other robots
            steer(robots, EPSILON)
            
            for r in range(0, len(robots)): 
                # get all nodes close to new node
                buildSetVnear(robots[r], EPSILON, GAMMARRT)
                
                extendGraph(robots[r])
                
                robots[r].addNode()
                
            # finding out if vnew should be in goal set
            if sample >= RANDOMSAMPLESMAX: 
                calculateGoalSet(robots, commRange, TIMEINTERVAL)
            
            rewireGraph(robots, TIMEINTERVAL, commRange, DEBUG)
            
        # check if we have a path
        
        if meeting:  
            connected = checkGoalSet(robots[0].graph)
        else: 
            connected = True
        if not connected:
            for r in range(0, len(robots)):
                robots[r].nodeCounter = robots[r].startNodeCounter
                robots[r].vnew = robots[r].startLocation
                robots[r].totalTime = robots[r].startTotalTime
                robots[r].initializeGraph()
                robots[r].addNode(firstTime = True)
        else:
            leastCostGoalSet(robots, DEBUG, meeting)
            
            for r in range(0, len(robots)):
                robots[r].vnew = robots[r].endLocation
                robots[r].totalTime = robots[r].endTotalTime
                getPath(robots[r])
        counter += 1

        if (counter % 10) == 0:
            print('Still planning: %d tries\r' %counter, end='')

    print('Needed %d retry(-ies) for path planning' %(counter-1))
    
def initializeRobots(numRobots, teams, schedule, logFile):
    """
    Initialize the robot class
    
    Input arguments:
    numRobots = how many robots
    teams = team assignments
    schedule = schedule for meeting events
    logFile = where to save the output
    """
    robots = []
    for r in range(0, numRobots):
        belongsToTeam = []
        for t in range(0,len(teams)):    
            if r+1 in teams[t][0]:
                belongsToTeam.append(t)
        rob = Robot(r, np.asarray(belongsToTeam), schedule[r], DISCRETIZATION, OPTPATH, OPTPOINT, 
                    SPATIOTEMPORAL, SPECIALKERNEL, POD, logFile, FOLDER + '/')
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

def initializeScheduler(numRobots, numTeams, robTeams):
    """
    Initialize schedule and create teams and schedule 
    
    Input arguments:
    numRobots = how many robots
    numTeams = how many teams
    robTeams = which robots are in which teams, comes from initial graph design; robot i belongs to team j in matrix
    """

    #initializer
    scheduleClass = Schedule(numRobots, numTeams, robTeams)
    
    #Assigns robot numbers to teams
    T = scheduleClass.createTeams()
    #creates schedule
    S = scheduleClass.createSchedule()
    #communication period is equal to number of robots
    communicationPeriod = np.shape(S)[1]  # Communication schedule repeats infinitely often

    #Print test information
    if DEBUG:
        print('Teams')
        print(*T)
        
        print('Period')
        print(communicationPeriod)
    
        print('Schedule')
        print(S)

    return S, T, communicationPeriod

def randomStartingPositions(numRobots):
    """
    Ensures that the starting position are exclusive within communication radius
    
    Input arguments:
    numRobots = how many robots
    """

    locations = np.zeros([numRobots, 2])
    pos = np.random.randint(0,2*COMMRANGE, size=2)
    locations[0] = pos
    
    for i in range(1,numRobots):
        equal = True
        while equal:
            pos = np.random.randint(0, 2*COMMRANGE, size=2)
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

def errorCalculation(robots,logFile):
    """
    Error calculation of modelling, computes different errors and writes to file

    Input arguments:
    robots = instance of the robots
    logFile = where to save the output
    """
    
    #TODO: use nrmse next time or fnorm
    for robot in robots:
        rmse = np.sqrt(np.square(robot.mappingGroundTruth - robot.expectedMeasurement).mean())
        logFile.writeError(robot.ID,rmse,robot.currentTime, 'RMSE', endTime=True)

        # nrmse = 100 * rmse/(np.max(robot.mappingGroundTruth)-np.min(robot.mappingGroundTruth))
        # logFile.writeError(robot.ID,nrmse,robot.currentTime, 'NRMSE', endTime=True)
        
        # rmse = np.sqrt(np.sum(np.square(robot.mappingGroundTruth - robot.expectedMeasurement)))
        # fnorm = rmse/(np.sqrt(np.sum(np.square(robot.mappingGroundTruth))))
        # logFile.writeError(robot.ID,fnorm,robot.currentTime, 'FNORM', endTime=True)

        similarity = ssim(robot.mappingGroundTruth,robot.expectedMeasurement, gaussian_weights=False)
        logFile.writeError(robot.ID,similarity,robot.currentTime, 'SSIM', endTime=True)

        _, _, procru = procrustes(robot.mappingGroundTruth,robot.expectedMeasurement)
        logFile.writeError(robot.ID,procru,robot.currentTime, 'Dissim')

if __name__ == "__main__":
    """Entry in Test Program"""
    
    """Setup Variables"""
    
    TOTALTIME = 100 #total execution time of program
    CASE = 3    #case corresponds to which robot structure to use (1 = 8 robots, 8 teams; 2 = 8 robots, 
                #5 teams; 3 = 4 robots 4 teams; 4 = 10 robots, 9 teams)
    CORRECTTIMESTEP = False #If dye time steps should be matched to correct time steps or if each 
                            #time step in dye corresponds to time step here
    
    DEBUG = False #debug to true shows prints
    BARIUM = False # if barium cloud data
    HETEROGENEOUS = False # if we are using heterogeneous robots
    ANIMATION = False #if animation should be done
    POD = True # if we are using POD or GP
    MODEL = True #if model should be calculated
    OPTPATH = MODEL == True #if path optimization should be used, can not be true if optpoint is used
    OPTPOINT = MODEL != OPTPATH == True #if point optimization should be used, can not be true if optpath is used
    
    SPATIOTEMPORAL = True # if spatiotemporal data or not
    STATIONARY = not SPATIOTEMPORAL #if we are using time varying measurement data or not
    SPECIALKERNEL = (False == SPATIOTEMPORAL) != STATIONARY  # if own kernel should be used, only works if spatiotemporal 
    STATIONARYTIME = 5 #which starting time to use for the measurement data, if not STATIONARY, 0 is used for default
    PREDICTIVETIME = None #Time for which to make a prediction at the end, has to be bigger than total time

    FULLYCONNECTED = False #if fully connected without commrange constraint
    
    if FULLYCONNECTED:
        COMMRANGE = 1200 # communication range for robots
        RANDOMSAMPLESMAX = 49 #how many random samples before trying to converge for communication
        TOTALSAMPLES = 50 #how many samples in total
    else:
        COMMRANGE = 3 # communication range for robots
        RANDOMSAMPLESMAX = 30 #how many random samples before trying to converge for communication
        TOTALSAMPLES = 50 #how many samples in total
    
    TIMEINTERVAL = 1 # time interval for communication events
    
    DISCRETIZATION = np.array([600, 600]) #grid space

    DIMENSION = 2 #dimension of robot space
    
    DELTAT = 0.1 #time step of simulation
       
    EPSILON = DISCRETIZATION[0]/10 # Maximum step size of robots
    GAMMARRT = 100 # constant for rrt* algorithm, can it be calculated?
    
    BASEPATH = 'Results/Tmp'
    LOGFILE = 'logFile'

    """Remove Tmp results files"""  
    for filename in os.listdir(BASEPATH):
        filePath = os.path.join(BASEPATH, filename)
        try:
            if os.path.isfile(filePath) or os.path.islink(filePath):
                os.unlink(filePath)
            elif os.path.isdir(filePath):
                shutil.rmtree(filePath)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (filePath, e))
    
    
    """Main function execution for a given number of iterations"""
    
    ITERATIONS = 1 # How many main iterations

    COMMNOISE = 0 # communication range noise parameter

    for i in range(1,ITERATIONS+1):

        # RANDINT = np.random.randint(0,2000)
        RANDINT = 752
        np.random.seed(RANDINT)
        
        if COMMNOISE != 0:
            FOLDER = BASEPATH + '/Noise '+str(COMMNOISE)
        else:
            FOLDER = BASEPATH + '/Test'+str(i)
        
        os.mkdir(FOLDER)
        
        main()
        if COMMNOISE != 0:
            COMMNOISE = COMMNOISE*2
