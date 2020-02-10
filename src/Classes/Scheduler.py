#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:44:07 2019

@author: hannes
"""

#General imports
import numpy as np

class Schedule:
    
    def __init__(self, numRobots, numTeams, robTeams):
        """scheduler class to create teams and schedules

        Input arguments:
        numRobots = how many robots
        numTeams = how many teams
        robTeams = which robots are in which teams, comes from initial graph design; robot i belongs to team j in matrix
        """

        self.numRobots = numRobots
        self.numTeams = numTeams
        self.robTeams = robTeams

    def createTeams(self):
        """Create teams based on number of robots and number of teams

        No input arguments
        """

        T = [[] for x in range(self.numTeams)]

        for i in range(0, self.numTeams):
            T[i] = np.where(self.robTeams[:, i] > 0)
            T[i] += np.ones(np.shape(T[i]),dtype=int)
        return T

    def createSchedule(self):
        """Create schedule based on team compositions

        No input arguments
        """
        
        T = self.createTeams()
        schedule = np.zeros((self.numRobots, self.numTeams))
        teams = np.where(self.robTeams[0,:] > 0)[0].astype('int')
        teams = teams + np.ones(np.shape(teams))
        teams = teams.astype('int')
        
        schedule[0, 0:np.shape(teams)[0]] = teams

        for j in range(0, self.numRobots):
            teams = np.where(self.robTeams[j, :] > 0)[0].astype('int')
            
            teams = teams + np.ones(np.shape(teams))
            teams = teams.astype('int')

            for t in range(0, np.shape(teams)[0]):
                rule12 = False
                rule3 = False
                team = teams[t]

                for col in range(0, self.numTeams):
                    if team in schedule[:, col]:
                        schedule[j, col] = team
                        rule12 = True
                        break
                if not rule12:
                    col = 0
                    while col <= self.numTeams and not rule3:
                        placedTeams = np.unique(schedule[np.where(schedule[:, col] > 0), col]).astype('int')
                        totalSum = 0
                        for pt in range(0, np.shape(placedTeams)[0]):
                            pteam = placedTeams[pt].astype('int')
                            if np.intersect1d(T[team - 1], T[pteam - 1]).size == 0:
                                totalSum += 1
                        if totalSum == np.shape(placedTeams)[0]:
                            schedule[j, col] = team
                            rule3 = True
                        col += 1

        schedule = schedule[:, ~np.all(schedule == 0, axis=0)]  # Remove columns full of zeros
        
        #Make the indexes go from 0 to numTeams-1 instead of 1 to numTeams
        schedule = schedule -1
        schedule = schedule.astype('int')
        return schedule