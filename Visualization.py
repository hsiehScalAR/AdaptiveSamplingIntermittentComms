#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:45:18 2019

@author: hannes
"""

import matplotlib.pyplot as plt

def plotMatrix(data):
    plt.figure()
    plt.imshow(data);
    plt.colorbar()
    plt.show()