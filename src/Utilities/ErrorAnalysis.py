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

    mean = np.zeros([10,9])
    std = np.zeros([10,9])

    for idx, data in enumerate(totalData):
        for i in range(0,10):
            for e in range(0,3):
                error,_ = zip(*data[i][e])
                error = np.array(error)
                mean[i,e + 3*idx] = np.mean(error)
                std[i,e + 3*idx] = np.std(error)
        
    bp1 = ax.boxplot([mean[:,0],mean[:,2]],positions=[1,4], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C0"))
    bp2 = ax.boxplot([mean[:,3],mean[:,5]], positions=[2,5], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C2"))
    bp3 = ax.boxplot([mean[:,6],mean[:,8]], positions=[3,6], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C1"))
    
    ax.set_ylim(0,2.5)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ['Intermittent', 'AllTime','Full'], loc='upper right')
    ax.set_ylabel('RMSE')
    plt.xticks([2, 5], ['RMSE', 'DISSIM'])
    plt.savefig(saveLoc + 'Boxplot_' + SETUP[stp] + '.png' )
    plt.close()

    print('\n' + SETUP[stp])

    print('intermittent, RMSE mean:    %.2f   std:  %.2f' %(np.mean(mean[:,0]), np.mean(std[:,0])))
    print('intermittent, DISSIM mean:  %.2f   std:  %.2f' %(np.mean(mean[:,2]), np.mean(std[:,2])))
    print('all-time, RMSE mean:        %.2f   std:  %.2f' %(np.mean(mean[:,3]), np.mean(std[:,3])))
    print('all-time, DISSIM mean:      %.2f   std:  %.2f' %(np.mean(mean[:,5]), np.mean(std[:,5])))
    print('full, RMSE mean:            %.2f   std:  %.2f' %(np.mean(mean[:,6]), np.mean(std[:,6])))
    print('full, DISSIM mean:          %.2f   std:  %.2f' %(np.mean(mean[:,8]), np.mean(std[:,8])))

def totalStatistics(totalData, saveLoc):
    """Plots the error in a combined box plot

    Input arguments:
    totalData = error data which is to be analysed
    saveLoc = where to save the image
    """

    mean = np.zeros([10,36])
    std = np.zeros([10,36])

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

    _, ax = plt.subplots(figsize=(8, 4))
        
    bp1 = ax.boxplot([mean[:,0],mean[:,9],mean[:,18],mean[:,27]],positions=[1,4,7,10], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C0"))
    bp2 = ax.boxplot([mean[:,3],mean[:,12],mean[:,21],mean[:,30]], positions=[2,5,8,11], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C2"))
    bp3 = ax.boxplot([mean[:,6],mean[:,15],mean[:,24],mean[:,33]], positions=[3,6,9,12], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C1"))
    
    ax.set_ylim(0,2.5)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ['Intermittent', 'AllTime','Full'], loc='lower right')
    ax.set_ylabel('RMSE')
    
    plt.xticks([2,3.5,5,8,9.5,11],['GP','\nSpatial','POD','GP','\nSpatiotemporal','POD'])

    ax.xaxis.set_tick_params(length=0)
    ax.axvline(6.5,ymin=0, ymax=1, color='grey',linestyle='--', lw=1)

    plt.savefig(saveLoc + 'Boxplot_Comparison_RMSE.png')
    plt.close()


    _, ax = plt.subplots(figsize=(8, 4))

    bp3 = ax.boxplot([mean[:,2],mean[:,11],mean[:,20],mean[:,29]],positions=[1,4,7,10], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C0"))
    bp4 = ax.boxplot([mean[:,5],mean[:,14],mean[:,23],mean[:,32]], positions=[2,5,8,11], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C2"))
    bp5 = ax.boxplot([mean[:,8],mean[:,17],mean[:,26],mean[:,35]], positions=[3,6,9,12], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C1"))
    
    ax.set_ylim(0,1)
    ax.legend([bp3["boxes"][0], bp4["boxes"][0], bp5["boxes"][0]], ['Intermittent', 'AllTime','Full'], loc='lower right')
    ax.set_ylabel('DISSIM')
    
    plt.xticks([2,3.5,5,8,9.5,11],['GP','\nSpatial','POD','GP','\nSpatiotemporal','POD'])

    ax.xaxis.set_tick_params(length=0)
    ax.axvline(6.5,ymin=0, ymax=1, color='grey',linestyle='--', lw=1)

    plt.savefig(saveLoc + 'Boxplot_Comparison_DISSIM.png')
    plt.close()

    return mean, std

def individualStatisticsHeterogeneous(totalData, saveLoc, stp):
    """Plots the error in an individual box plot

    Input arguments:
    totalData = error data which is to be analysed
    saveLoc = where to save the image
    stp = stationary or spatiotemporal case
    """

    _, ax = plt.subplots()

    if stp == 0:
        name = 'Homogeneous Results GP (n=10)'
    elif stp == 2:
        name = 'Homogeneous Results POD (n=10)'
    elif stp == 1:
        name = 'Heterogeneous Results GP (n=10)'
    else:
        name = 'Heterogeneous Results POD (n=10)'
    ax.set_title(name)

    mean = np.zeros([10,3])
    std = np.zeros([10,3])

    for i in range(0,10):
        for e in range(0,3):
            error,_ = zip(*totalData[i][e])
            error = np.array(error)
            mean[i,e] = np.mean(error)
            std[i,e] = np.std(error)
    
    bp = ax.boxplot([mean[:,0],mean[:,2]],positions=[1,2], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C0"))
    
    ax.set_ylim(0,2.5)
    ax.legend([bp["boxes"][0]], ['Intermittent'], loc='lower right')
    ax.set_ylabel('RMSE')
    plt.xticks([1, 2], ['RMSE', 'DISSIM'])
    plt.savefig(saveLoc + 'Boxplot_' + SETUP[stp] + '.png' )
    plt.close()

    print('\n' + SETUP[stp])

    print('RMSE mean:    %.2f   std:  %.2f' %(np.mean(mean[:,0]), np.mean(std[:,0])))
    print('DISSIM mean:  %.2f   std:  %.2f' %(np.mean(mean[:,2]), np.mean(std[:,2])))
    
def totalStatisticsHeterogeneous(totalData, saveLoc):
    """Plots the error in a combined box plot

    Input arguments:
    totalData = error data which is to be analysed
    saveLoc = where to save the image
    """

    mean = np.zeros([10,12])
    std = np.zeros([10,12])

    index = 0

    for _, dataSet in enumerate(totalData):
        for i in range(0,10):
            for e in range(0,3):
                error,_ = zip(*dataSet[i][e])
                error = np.array(error)
                mean[i,e + index] = np.mean(error)
                std[i,e + index] = np.std(error)
        index += 3

    _, ax = plt.subplots(figsize=(8, 4))
        
    bp = ax.boxplot([mean[:,0],mean[:,3],mean[:,6],mean[:,9]],positions=[1,3,2,4], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C0"))
    
    ax.set_ylim(0,2.5)
    ax.legend([bp["boxes"][0]], ['Intermittent'], loc='lower right')
    ax.set_ylabel('RMSE')
    
    plt.xticks([1, 1.5, 2, 3, 3.5, 4],['GP','\nHomogeneous','POD','GP','\nHeterogeneous','POD'])

    ax.xaxis.set_tick_params(length=0)
    ax.axvline(2.5,ymin=0, ymax=1, color='grey',linestyle='--', lw=1)

    plt.savefig(saveLoc + 'Boxplot_Comparison_Heterogeneous_RMSE.png')
    plt.close()


    _, ax = plt.subplots(figsize=(8, 4))

    bp1 = ax.boxplot([mean[:,2],mean[:,5],mean[:,8],mean[:,11]],positions=[1,3,2,4], notch=True, widths=0.35, patch_artist=True, boxprops=dict(facecolor="C0"))
    
    ax.set_ylim(0,1)
    ax.legend([bp1["boxes"][0]], ['Intermittent'], loc='lower right')
    ax.set_ylabel('DISSIM')
    
    plt.xticks([1, 1.5, 2, 3, 3.5, 4],['GP','\nHomogeneous','POD','GP','\nHeterogeneous','POD'])

    ax.xaxis.set_tick_params(length=0)
    ax.axvline(2.5,ymin=0, ymax=1, color='grey',linestyle='--', lw=1)

    plt.savefig(saveLoc + 'Boxplot_Comparison_Heterogeneous_DISSIM.png')
    plt.close()

    return mean, std

if __name__ == "__main__":
    """Entry in Error Analysis Program"""

    HETEROGENEOUS = True

    if HETEROGENEOUS:
        basePath = '/home/hannes/MasterThesisCode/AdaptiveSamplingIntermittentComms/src/Results/Tests/IntermediateResults/Heterogeneous/'
        saveLoc = '/home/hannes/MasterThesisCode/AdaptiveSamplingIntermittentComms/src/Results/Tests/IntermediateResults/Figures/'
        
        SETUP = ['GPHomogeneous','GPHeterogeneous','PODHomogeneous','PODHeterogeneous']    
        
        totalData = []
        for stp, _ in enumerate(SETUP):
            path = basePath + SETUP[stp]
            ROBOTID = 0
            
            data = readAllFiles(path)
            totalData.append(data)
            individualStatisticsHeterogeneous(data, saveLoc, stp)

        mean, std = totalStatisticsHeterogeneous(totalData, saveLoc)

    else:
        basePath = '/home/hannes/MasterThesisCode/AdaptiveSamplingIntermittentComms/src/Results/Tests/IntermediateResults/'
        saveLoc = '/home/hannes/MasterThesisCode/AdaptiveSamplingIntermittentComms/src/Results/Tests/IntermediateResults/Figures/'
        CASE = ['Intermittent','AllTime','Full']

        SETUP = ['SpatialGP','SpatialPOD','SpatioTemporalGP','SpatioTemporalPOD']    
        
        totalData = []
        for stp, _ in enumerate(SETUP):
            individualData = []
            for idx,_ in enumerate(CASE):
                if CASE[idx] == 'Intermittent':
                    path = basePath + SETUP[stp] + '/IntermittentCommunication'
                    ROBOTID = 0
                elif CASE[idx] == 'AllTime':
                    path = basePath + SETUP[stp] + '/AllTimeCommunication'
                    ROBOTID = 4
                elif CASE[idx] == 'Full':
                    path = basePath + SETUP[stp] + '/FullCommunication'
                    ROBOTID = 0
            
                data = readAllFiles(path)
                individualData.append(data)
            totalData.append(individualData)
            individualStatistics(individualData, saveLoc, stp)

        mean, std = totalStatistics(totalData, saveLoc)
    

