#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  15 14:17:32 2020

@author: hannes
"""

#General imports
import numpy as np
import matplotlib.pyplot as plt
import os

def readAllFiles(path):
    data = []
    for dirpath, dirnames, files in os.walk(path):
        for file_name in files:
            if file_name == 'logFile.txt':
                with open(dirpath + '/' + file_name, "r+") as logFileData:

                    rmse = []
                    ssim = []
                    dissim = []                    

                    for line in logFileData:
                        if line.startswith('\n'):
                            continue
                        if line.startswith('#'):
                            header = line
                            continue
                        if header.startswith('# Intermediate'):
                            if line.startswith('Metric'):
                                continue
                            else:
                                columns = line.split()
                                if columns[1] == '0':
                                    if columns[0] == 'RMSE':
                                        rmse.append([float(columns[2]), float(columns[3])])
                                    elif columns[0] == 'SSIM':
                                        ssim.append([float(columns[2]), float(columns[3])])
                                    elif columns[0] == 'Dissim':
                                        dissim.append([float(columns[2]), float(columns[3])])
                    data.append([rmse,ssim,dissim])
    return data

def plotError(data, metric, saveLoc, testrun=None):
    if metric == 'RMSE':
        idx = 0
        bottom = 0
        top = 3
    elif metric == 'SSIM':
        idx = 1
        bottom = 0
        top = 1
    elif metric == 'DISSIM':
        idx = 2
        bottom = 0
        top = 1
        
    if testrun != None:
        
        plt.figure()
        error, t = zip(*data[testrun][idx])
        plt.plot(t,error, '-', label=metric)
        plt.ylim([bottom, top])
        plt.legend()
        plt.savefig(saveLoc + metric + '_Testrun_%d'%testrun + '.png' )
        plt.close()


if __name__ == "__main__":

    path = '/home/hannes/MasterThesisCode/AdaptiveSamplingIntermittentComms/src/Results/Tests/IntermediateResults/SpatioTemporal/'
    saveLoc = '/home/hannes/MasterThesisCode/AdaptiveSamplingIntermittentComms/src/Results/Tests/IntermediateResults/Figures/'
    data = readAllFiles(path)
    
    plotError(data,'DISSIM',saveLoc,0)


