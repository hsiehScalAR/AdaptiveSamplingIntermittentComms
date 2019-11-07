#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:45:18 2019

@author: hannes
"""

import matplotlib.pyplot as plt
import networkx as nx

def plotMatrix(data):
    plt.figure()
    plt.imshow(data);
    plt.colorbar()
    plt.show()
    
def plotMeetingGraphs(graph1, graph2):
    plt.figure()
    nx.draw(graph1, pos=nx.get_node_attributes(graph1, 'pos'), node_color='r',node_size=50,edge_color='r')
    nx.draw(graph2, pos=nx.get_node_attributes(graph2, 'pos'), node_color='b',node_size=50,edge_color='b')
    plt.show()