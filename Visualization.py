#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:45:18 2019

@author: hannes
"""

import matplotlib.pyplot as plt
import networkx as nx

def clearPlots():
    plt.close('all')
    
def plotMatrix(data):
    plt.figure()
    plt.imshow(data);
    plt.colorbar()
    plt.show()
    
def plotMeetingGraphs(robots, index):
    plt.figure('robots %.d and %.d' %(index[0],index[1]))
    graph1 = robots[index[0]].totalGraph
    graph2 = robots[index[1]].totalGraph
    nx.draw(graph1, label='robot %.d' %(index[0]), pos=nx.get_node_attributes(graph1, 'pos'), node_color='r',node_size=400,edge_color='r',with_labels = True,font_color='w')
    nx.draw(graph2, label='robot %.d' %(index[1]), pos=nx.get_node_attributes(graph2, 'pos'), node_color='b',node_size=400,edge_color='b',with_labels = True,font_color='w')
    
    plt.legend()
    plt.show()