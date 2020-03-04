#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 4 10:08:24 2020

@author: hannes
"""

#General imports
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import os
import numpy as np
import skimage as skimage

"""
NOTE:

In order to generate the image sequence we can run following command in a terminal:

ffmpeg -i "barium_cloud_1_movie.mp4" -f image2 "video-frame%05d.png"

"""


def readAllFiles(path):
    """Function which opens every subdir and reads the logFiles

    Input arguments:
    path = path to main data directory
    """
    images = os.listdir(path)
    sortedImages = sorted(images)
    cropBox = [65,200,160,610]
    scaling = 15
    dataSmall = []
    dataBig = []

    for entry in sortedImages:
        if os.path.isfile(os.path.join(path, entry)):
            image = mpimg.imread(path + entry)
            image = image[cropBox[0]:cropBox[1],cropBox[2]:cropBox[3],0]

            image = (image + 1)*(-1) + 2.4
            image = np.where(image < 1.4, image, 0)
            image = np.where(image > 1, 0, image)

            image = image*scaling
            image = np.pad(image,((50,50),(0,0)))
            image = skimage.filters.gaussian(image,2)

            newImage = skimage.transform.resize(image,[600,600])
            
            plt.imshow(newImage, origin='lower',vmin=-1, vmax=15)
            plt.colorbar()
            plt.savefig(savePath + entry)
            plt.close()
            
            dataBig.append(newImage)
            dataSmall.append(image)

    return dataSmall, dataBig

if __name__ == "__main__":
    """Entry in barium cloud preprocessing Program"""

    basePath = '/home/hannes/MasterThesisCode/AdaptiveSamplingIntermittentComms/src/Data/BariumCloudImages/Raw/'
    savePath = '/home/hannes/MasterThesisCode/AdaptiveSamplingIntermittentComms/src/Data/BariumCloudImages/Processed/'

    dataSmall, dataBig = readAllFiles(basePath)
    np.savez(savePath + 'BariumCloudDataSmall', data=dataSmall)
    np.savez(savePath + 'BariumCloudDataBig', data=dataBig)

    # npzFile = np.load(savePath + 'BariumCloudDataSmall.npz')
    # newData = npzFile['data']

    # plt.imshow(newData[-1], origin='lower',vmin=-1, vmax=15)
    # plt.colorbar()
    # plt.savefig(savePath + 'Frame-1' + '.png')
    # plt.close()
    # print(len(podData))

