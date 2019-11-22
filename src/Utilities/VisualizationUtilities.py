#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:45:18 2019

@author: hannes
"""

#General imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.animation as animation

def clearPlots():
    plt.close('all')

def plotTrajectoryAnimation(robots, save=False):
    
    fig = plt.figure()
    ax1 = plt.axes(xlim=(0, 600), ylim=(0,600))
    graph, = ax1.plot([], [], '.-')
    
    graphs = []
    for r in range(len(robots)):
        graphobj = ax1.plot([], [], '-', label='Robot %d'%r)[0]
        graphs.append(graphobj)
    
    
    def init():
        for graph in graphs:
            graph.set_data([],[])
        return graphs
    
    xlist = [i for i in range(len(robots))]
    ylist = [i for i in range(len(robots))]
    
    def animate(i):
        for r in range(0,len(robots)):
            x,y = zip(*robots[r].trajectory)
            xlist[r] = x[:i]
            ylist[r] = y[:i]

        for gnum,graph in enumerate(graphs):
            
            graph.set_data(xlist[gnum], ylist[gnum]) # set data for each line separately.     
        return graphs

    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(robots[-1].trajectory), interval=200)
    plt.legend()
    plt.show()
    if save:
        ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    
def plotTrajectory(robots):
    plt.figure()
    for r in range(0,len(robots)):
        x,y = zip(*robots[r].trajectory)
        plt.plot(x,y, '-', label='Robot %d'%r)
    plt.legend()
    plt.show()

def plotMatrix(data):
    plt.figure()
    plt.title('Measurements of robots after communication events')
    plt.imshow(data);
    plt.colorbar()
    plt.show()
    
def plotMeetingGraphs(robots, index, subplot=None, length=0):
    #TODO change so that any number of robots in same team can be plotted
    if subplot != None:
        plt.figure('RRT* Graphs')
        plt.subplot(np.ceil(length/2),2,subplot)
    else:
        plt.figure()
   
    plt.title('robots %.d and %.d' %(index[0],index[1]))
    
    graph1 = robots[index[0]].totalGraph
    graph2 = robots[index[1]].totalGraph
    nx.draw(graph1, label='robot %.d' %(index[0]), pos=nx.get_node_attributes(graph1, 'pos'), node_color='r',node_size=100,edge_color='r',with_labels = True,font_color='w',font_size=8)
    nx.draw(graph2, label='robot %.d' %(index[1]), pos=nx.get_node_attributes(graph2, 'pos'), node_color='b',node_size=100,edge_color='b',with_labels = True,font_color='w',font_size=8)
    
    plt.legend()
    plt.show()
    

def plotMeetingPaths(robots, index, subplot=None, length=0):
    #TODO change so that any number of robots in same team can be plotted
    if subplot != None:
        plt.figure('RRT* Paths')
        plt.subplot(np.ceil(length/2),2,subplot)
    else:
        plt.figure()
    
    plt.title('robots %.d and %.d' %(index[0],index[1]))
    
    graph1 = robots[index[0]].totalPath
    graph2 = robots[index[1]].totalPath
    nx.draw(graph1, label='robot %.d' %(index[0]), pos=nx.get_node_attributes(graph1, 'pos'), node_color='r',node_size=100,edge_color='r',with_labels = True,font_color='w',font_size=8)
    nx.draw(graph2, label='robot %.d' %(index[1]), pos=nx.get_node_attributes(graph2, 'pos'), node_color='b',node_size=100,edge_color='b',with_labels = True,font_color='w',font_size=8)
    
    plt.legend()
    plt.show()