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
    """Function which opens every subdir and reads the logFiles

    Input arguments:
    path = path to main parent directory
    """
    
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
    """Plots the error in a lineplot

    Input arguments:
    data = error data which is to be analysed
    metric = which error metric is being used 
    saveLoc = where to save the image
    testrun = which test run to plot or if None all testruns
    """

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

def individualStatistics(totalData, saveLoc, stp):
    """Plots the error in an individual box plot

    Input arguments:
    totalData = error data which is to be analysed
    saveLoc = where to save the image
    stp = stationary or spatiotemporal case
    """

    _, ax = plt.subplots()

    if stp == 0:
        name = 'Stationary Results GP (n=10)'
    elif stp == 1:
        name = 'Stationary Results POD (n=10)'
    elif stp == 2:
        name = 'Spatiotemporal Results GP (n=10)'
    else:
        name = 'Spatiotemporal Results POD (n=10)'
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
    ax.set_ylabel('RMSE')
    plt.xticks([1.5, 3.5], ['RMSE', 'DISSIM'])
    plt.savefig(saveLoc + 'Boxplot_' + SETUP[stp] + '.png' )
    plt.close()

    print('\n' + SETUP[stp])

    print('intermittent, RMSE mean:    %.2f   std:  %.2f' %(np.mean(mean[:,0]), np.mean(std[:,0])))
    print('intermittent, DISSIM mean:  %.2f   std:  %.2f' %(np.mean(mean[:,2]), np.mean(std[:,2])))
    print('full, RMSE mean:            %.2f   std:  %.2f' %(np.mean(mean[:,3]), np.mean(std[:,3])))
    print('full, DISSIM mean:          %.2f   std:  %.2f' %(np.mean(mean[:,5]), np.mean(std[:,5])))

def totalStatistics(totalData, saveLoc):
    """Plots the error in a combined box plot

    Input arguments:
    totalData = error data which is to be analysed
    saveLoc = where to save the image
    """

    mean = np.zeros([10,24])
    std = np.zeros([10,24])

    index = 0

    for _, dataSet in enumerate(totalData):
        for _, data in enumerate(dataSet):
            for i in range(0,10):
                for e in range(0,3):
                    error,_ = zip(*data[i][e])
                    error = np.array(error)
                    mean[i,e + index] = np.mean(error)
                    std[i,e + index] = np.std(error)
            index += 3

    _, ax = plt.subplots(figsize=(8, 6))
        
    bp1 = ax.boxplot([mean[:,0],mean[:,6],mean[:,12],mean[:,18]],positions=[1,3,5,7], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C0"))
    bp2 = ax.boxplot([mean[:,3],mean[:,9],mean[:,15],mean[:,21]], positions=[2,4,6,8], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C2"))
    ax.set_ylim(0,2.5)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Intermittent', 'Full'], loc='lower right')
    ax.set_ylabel('RMSE')
    
    plt.xticks([1.5,2.5,3.5,5.5,6.5,7.5],['GP','\nSpatial','POD','GP','\nSpatiotemporal','POD'])

    ax.xaxis.set_tick_params(length=0)
    ax.axvline(4.5,ymin=0, ymax=1, color='grey',linestyle='--', lw=1)

    plt.savefig(saveLoc + 'Boxplot_Comparison_RMSE.png')
    plt.close()


    _, ax = plt.subplots(figsize=(8, 6))

    bp3 = ax.boxplot([mean[:,2],mean[:,8],mean[:,14],mean[:,20]],positions=[1,3,5,7], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C0"))
    bp4 = ax.boxplot([mean[:,5],mean[:,11],mean[:,17],mean[:,23]], positions=[2,4,6,8], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C2"))
    ax.set_ylim(0,1)
    ax.legend([bp3["boxes"][0], bp4["boxes"][0]], ['Intermittent', 'Full'], loc='lower right')
    ax.set_ylabel('DISSIM')
    
    plt.xticks([1.5,2.5,3.5,5.5,6.5,7.5],['GP','\nSpatial','POD','GP','\nSpatiotemporal','POD'])

    ax.xaxis.set_tick_params(length=0)
    ax.axvline(4.5,ymin=0, ymax=1, color='grey',linestyle='--', lw=1)

    plt.savefig(saveLoc + 'Boxplot_Comparison_DISSIM.png')
    plt.close()

    return mean, std

if __name__ == "__main__":
    """Entry in Error Analysis Program"""

    basePath = '/home/hannes/MasterThesisCode/AdaptiveSamplingIntermittentComms/src/Results/Tests/IntermediateResults/'
    saveLoc = '/home/hannes/MasterThesisCode/AdaptiveSamplingIntermittentComms/src/Results/Tests/IntermediateResults/Figures/'
    CASE = ['Intermittent','Full']

    SETUP = ['SpatialGP','SpatialPOD','SpatioTemporalGP','SpatioTemporalPOD']    
    
    totalData = []
    for stp, _ in enumerate(SETUP):
        individualData = []
        for idx,_ in enumerate(CASE):
            if CASE[idx] == 'Intermittent':
                path = basePath + SETUP[stp] + '/IntermittentCommunication'
                ROBOTID = 0
            elif CASE[idx] == 'Full':
                path = basePath + SETUP[stp] + '/AllTimeCommunication'
                ROBOTID = 4
        
            data = readAllFiles(path)
            individualData.append(data)
        totalData.append(individualData)
        individualStatistics(individualData, saveLoc, stp)

    mean, std = totalStatistics(totalData, saveLoc)

    # print(np.mean(mean[0:-1:3],axis=0))
    # print(np.std(mean[0:-1:3],axis=0))
    

