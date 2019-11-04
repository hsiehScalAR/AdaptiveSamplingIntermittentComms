#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:48:29 2019

@author: hannes
"""

import numpy as np
from IntermittentComms import Schedule, Robot


def main():
    """main test loop"""
    # no inputs 
   
    """Variables"""
    discretization = (600, 600)
    sensorPeriod = 20
    eigenVecPeriod = 40
    num_locations = 20
    locations = np.random.randint(0, 599, size=(num_locations, 2))
    total_time = 1000
    curr_time = 0
    
    """create robot to team correspondence"""
    #robot i belongs to team j
    if CASE == 1:
        num_teams = 8
        num_robots = 8
        rob_in_teams = np.array([[1, 1, 0, 0, 0, 0, 0, 0], 
                                 [0, 1, 1, 0, 0, 0, 0, 0], 
                                 [0, 0, 1, 1, 0, 0, 0, 0], 
                                 [0, 0, 0, 1, 1, 0, 0, 0], 
                                 [0, 0, 0, 0, 1, 1, 0, 0], 
                                 [0, 0, 0, 0, 0, 1, 1, 0],
                                 [0, 0, 0, 0, 0, 0, 1, 1], 
                                 [1, 0, 0, 0, 0, 0, 0, 1],])
    
    elif CASE == 2:
        num_teams = 5
        num_robots = 8
        rob_in_teams = np.array([[1, 1, 0, 0, 0], 
                                 [1, 0, 0, 1, 0], 
                                 [1, 0, 0, 0, 1], 
                                 [0, 1, 1, 0, 0], 
                                 [0, 1, 0, 0, 1], 
                                 [0, 0, 1, 1, 0],
                                 [0, 0, 1, 0, 1], 
                                 [0, 0, 0, 1, 1],])
    
    else:
        exit()


    """--------------------------------------------------------------------------"""    
    """Perform Tests"""
    #scheduler test
    schedule, teams, commPeriod = testScheduler(num_robots, num_teams, rob_in_teams)
    
    #robot test
    robots = testRobot(num_robots, teams, schedule, discretization)
    
    
    
    while curr_time < total_time:
        # Collect and send sensing data
        for r in range(0, num_robots):
            data_val = np.ones((num_locations, sensorPeriod))  # Test data matrix for 20 locations over duration of sensor data
            robots[r].add_new_data(data_val, locations, (curr_time, curr_time + sensorPeriod), 'sensor')  # Set data matrices
        curr_time += sensorPeriod

        locations = np.remainder(locations + 1, 600)  # Test changing locations

        # Create new matrix of data points, fill in sensor values
        #
        # Estimate missing values using communication protocol here
        # Collect and send eigenvector data
        # for r in range(0, num_robots):
        #
        #     robots[r].add_new_data([data_val * 3] * sensorPeriod, (curr_time, curr_time + eigenVecPeriod), 'eigen')
        # curr_time += eigenVecPeriod

        # Use communication protocol here

        data_val += 1

    data_matrix_of_sensor_measurements = Robot.construct_data_matrix()  # Aggregated matrix of estimated values from robots
    
    # Estimate missing values using gappy POD

    
def testRobot(num_robots, teams, schedule, discretization):



    robots = []
    for r in range(0, num_robots):
        rob = Robot(r + 1, teams[r][0], schedule[r], discretization)
        robots.append(rob)
        #Print test information
    if DEBUG:
        print('Robot 1 schedule')
        print(robots[0].schedule)
        
        print('Robot 2 team')
        print(robots[1].teams)
        
        print('Robot 3 ID')
        print(robots[2].ID)
    
    return robots

    

 
def testScheduler(num_robots, num_teams, rob_in_teams):
    """initialize schedule and create teams and schedule"""  
    # Input arguments:
    # num_robots = how many robots
    # num_teams = how many teams
    # rob_in_teams = which robots are in which teams, comes from initial graph design; robot i belongs to team j in matrix
    
    #initializer
    scheduleClass = Schedule(num_robots, num_teams, rob_in_teams)
    #Assigns robot numbers to teams
    T = scheduleClass.create_teams()
    #creates schedule
    S = scheduleClass.create_schedule()
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
    #case corresponds to which robot structure to use (1 = 8 robots, 8 teams, 2 = 8 robots, 5 teams)
    #debug to true shows prints
    
    CASE = 1
    DEBUG = True

    main()