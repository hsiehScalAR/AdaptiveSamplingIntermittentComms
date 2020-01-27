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
    """Function which opens every subdir and reads the logFiles"""
    # Input arguments:
    # path = path to main parent directory

    data = []
    for dirpath, _, files in os.walk(path):
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
                                if int(columns[1]) == ROBOTID:
                                    if columns[0] == 'RMSE':
                                        rmse.append([float(columns[2]), float(columns[3])])
                                    elif columns[0] == 'SSIM':
                                        ssim.append([float(columns[2]), float(columns[3])])
                                    elif columns[0] == 'Dissim':
                                        dissim.append([float(columns[2]), float(columns[3])])
                    data.append([rmse,ssim,dissim])
    return data

def plotError(data, metric, saveLoc, testrun=None):
    """Plots the error in a lineplot"""
    # Input arguments:
    # data = error data which is to be analysed
    # metric = which error metric is being used 
    # saveLoc = where to save the image
    # testrun = which test run to plot or if None all testruns

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
        plt.plot(t,error, '-', label=metric + '_run_%d' %testrun)
        plt.ylim([bottom, top])
        plt.legend()
        plt.savefig(saveLoc + metric + '_Testrun_%d'%testrun + '.png' )
        plt.close()
    else:
        plt.figure()
        plt.title(metric + '_All_' + CASE)
        for i in range(0,np.shape(data)[0]):
            error, t = zip(*data[i][idx])
            plt.plot(t,error, '-', label=metric + '_run_%d' %i)
        plt.ylim([bottom, top])
        plt.savefig(saveLoc + metric + '_All_' + CASE + '.png' )
        plt.close()

def statistics(totalData, saveLoc, stp):
    """Plots the error in a combined box plot"""
    # Input arguments:
    # totalData = error data which is to be analysed
    # saveLoc = where to save the image
    # stp = stationary or spatiotemporal case

    _, ax = plt.subplots()

    if stp == 0:
        name = 'Stationary Results (n=10)'
    else:
        name = 'Spatiotemporal Results (n=10)'
    ax.set_title(name)

    mean = np.zeros([10,6])
    std = np.zeros([10,6])

    for idx, data in enumerate(totalData):
        for i in range(0,10):
            for e in range(0,3):
                error,_ = zip(*data[i][e])
                error = np.array(error)
                mean[i,e + 3*idx] = np.mean(error)
                std[i,e + 3*idx] = np.std(error)
        
    bp1 = ax.boxplot([mean[:,0],mean[:,2]],positions=[1,3], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C0"))
    bp2 = ax.boxplot([mean[:,3],mean[:,5]], positions=[2,4], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C2"))
    ax.set_ylim(0,2.5)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Intermittent', 'Full'], loc='upper right')

    plt.xticks([1.5, 3.5], ['RMSE', 'DISSIM'])
    plt.savefig(saveLoc + 'Boxplot_' + SETUP[stp] + '.png' )
    plt.close()


if __name__ == "__main__":
    """Entry in Error Analysis Program"""

    basePath = '/home/hannes/MasterThesisCode/AdaptiveSamplingIntermittentComms/src/Results/Tests/IntermediateResults/'
    saveLoc = '/home/hannes/MasterThesisCode/AdaptiveSamplingIntermittentComms/src/Results/Tests/IntermediateResults/Figures/'
    CASE = ['Intermittent','Full']

    SETUP = ['Spatial','SpatioTemporal']    
    
    for stp, _ in enumerate(SETUP):
        totalData = []
        for idx,_ in enumerate(CASE):
            if CASE[idx] == 'Intermittent':
                path = basePath + SETUP[stp] + '/IntermittentCommunication'
                ROBOTID = 0
            elif CASE[idx] == 'Full':
                path = basePath + SETUP[stp] + '/AllTimeCommunication'
                ROBOTID = 4
        
            data = readAllFiles(path)
            totalData.append(data)
        statistics(totalData, saveLoc, stp)
        # plotError(data,'DISSIM',saveLoc)


