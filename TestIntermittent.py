#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:48:29 2019

@author: hannes
"""

import numpy as np
from IntermittentComms import Schedule, Robot

if __name__ == "__main__":
    num_teams = 5
    num_robots = 8

    #robot i belongs to team j
#    rob_in_teams = np.array([[1, 1, 0, 0, 0, 0, 0, 0], 
#                             [0, 1, 1, 0, 0, 0, 0, 0], 
#                             [0, 0, 1, 1, 0, 0, 0, 0], 
#                             [0, 0, 0, 1, 1, 0, 0, 0], 
#                             [0, 0, 0, 0, 1, 1, 0, 0], 
#                             [0, 0, 0, 0, 0, 1, 1, 0],
#                             [0, 0, 0, 0, 0, 0, 1, 1], 
#                             [1, 0, 0, 0, 0, 0, 0, 1],])

    rob_in_teams = np.array([[1, 1, 0, 0, 0], 
                             [1, 0, 0, 1, 0], 
                             [1, 0, 0, 0, 1], 
                             [0, 1, 1, 0, 0], 
                             [0, 1, 0, 0, 1], 
                             [0, 0, 1, 1, 0],
                             [0, 0, 1, 0, 1], 
                             [0, 0, 0, 1, 1],])
    

    schedule = Schedule(num_robots, num_teams, rob_in_teams)

    T = schedule.create_teams()

    S = schedule.create_schedule()

    communication_period = np.shape(S)[0]  # Communication schedule repeats infinitely often

    sensor_data_period = 20
    eigenvec_data_period = 40

    total_time = 1000

    curr_time = 0

    robots = []
    for r in range(0, num_robots):
        teams = np.where(rob_in_teams[r, :] > 0)[0].astype('int')
        teams = teams + np.ones(np.shape(teams))
        rob = Robot(r + 1, teams, S[r])
        robots.append(rob)

    num_locations = 20
    locations = np.random.randint(0, 599, size=(num_locations, 2))
    while curr_time < total_time:
        # Collect and send sensing data
        for r in range(0, num_robots):
            data_val = np.ones((num_locations, sensor_data_period))  # Test data matrix for 20 locations over duration of sensor data
            robots[r].add_new_data(data_val, locations, (curr_time, curr_time + sensor_data_period), 'sensor')  # Set data matrices
        curr_time += sensor_data_period

        locations = np.remainder(locations + 1, 600)  # Test changing locations

        # Create new matrix of data points, fill in sensor values
        #
        # Estimate missing values using communication protocol here
        # Collect and send eigenvector data
        # for r in range(0, num_robots):
        #
        #     robots[r].add_new_data([data_val * 3] * sensor_data_period, (curr_time, curr_time + eigenvec_data_period), 'eigen')
        # curr_time += eigenvec_data_period

        # Use communication protocol here

        data_val += 1

    data_matrix_of_sensor_measurements = Robot.construct_data_matrix()  # Aggregated matrix of estimated values from robots

    # Testing functionality - TODO remove
    print(*T)
    print(S)
    # Estimate missing values using gappy POD
