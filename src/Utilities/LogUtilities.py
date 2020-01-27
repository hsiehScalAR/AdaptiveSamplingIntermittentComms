#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:15:54 2020

@author: hannes
"""

#General imports
import numpy as np
import datetime

# Personal imports

class LogFile:

    def __init__(self, name, path):
        """Initialises parameters of logfile"""
        # Input arguments:
        # name = name of the logfile
        # path = location of logfile

        self.name = path + name + '.txt'
        self.createLogFile(name, path)
        self.firstTimeParam = 0
        self.firstTimeInter = True
        self.firstTimeEnd = True

    def createLogFile(self, name, path):
        """Creates logfile"""
        # Input arguments:
        # name = name of the logfile
        # path = location of logfile
        
        with open(self.name, 'w') as writer:
            writer.write('# This is a LogFile for TestIntermittent.py\n\n')
            writer.write('Hannes Rovina\n')
            writer.write(str(datetime.datetime.now()) + '\n')

    def writeParameters(self, **parameters):
        """Writes parameters to logfile"""
        # Input arguments:
        # parameters = dict of parameters which have to be written to logfile

        if self.firstTimeParam < 2:
            with open(self.name,'a') as writer:
                if self.firstTimeParam == 0:
                    writer.write('\n# Parameters:\n\n')
                else:
                     writer.write('\n')
                for key, value in parameters.items():
                    writer.write(key + '\t\t' + str(value) + '\n')
                self.firstTimeParam += 1

    def writeError(self, robotID, error, time, errorName, endTime=False):
        """Writes the errors to the logfile"""
        # Input arguments:
        # robotID = which robot 
        # error = modelling error
        # time = time of error
        # errorName = name of the error
        # endTime = bool if we are at end of execution

        with open(self.name,'a') as writer:
            if self.firstTimeInter:
                writer.write('\n# Intermediate Error Results:\n\n')
                writer.write('Metric:\t\tID:\t\tError:\t\tTime:\n\n')
                self.firstTimeInter = False
            elif endTime and self.firstTimeEnd:
                writer.write('\n# Final Error Results:\n\n')
                writer.write('Metric:\t\tID:\t\tError:\t\tTime:\n\n')
                self.firstTimeEnd = False
                
            writer.write('{:7s}\t\t{:3d}\t\t{:6.2f}\t\t{:6.1f} \n'.format(errorName,robotID,error,time))


            

