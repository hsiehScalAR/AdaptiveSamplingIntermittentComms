#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:45:18 2019

@author: hannes
"""

#General imports
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import networkx as nx
import numpy as np
import matplotlib.animation as animation

from scipy.stats import multivariate_normal

def plotMultivariateNormal():
    x, y = np.mgrid[0:600:1, 0:600:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    rv = []
    plt.figure()
    totalrv = 0
    rv.append(multivariate_normal([30, 30], [[200.0, 100], [100, 200]]).pdf(pos))
    rv.append(multivariate_normal([30, 60], [[200.0, 100], [100, 200]]).pdf(pos))
    rv.append(multivariate_normal([60, 30], [[200.0, 100], [100, 200]]).pdf(pos))
    rv.append(multivariate_normal([60, 60], [[800.0, 100], [100, 800]]).pdf(pos))
    
    for i in range(0,4):
        totalrv += rv[i]

    plt.contourf(x, y, totalrv)
    plt.colorbar()

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

def plotMeasurement(data, title):
    
    fig, ax = plt.subplots()

    ax.set_title(title)
    plt.imshow(data, origin='lower');
    plt.colorbar()

    plt.show()
    
def plotMeetingGraphs(robots, index, team, subplot=None, length=0):

    if subplot != None:
        plt.figure('RRT* Graphs')
        plt.subplot(np.ceil(length/2),2,subplot)
    else:
        plt.figure()
   
    plt.title('Team %.d' %team)
    cmap = plt.get_cmap('hsv')
    for i in range(0,len(index)):
        graph = robots[index[i]].totalGraph
        node_color = colors.to_hex(cmap(i/len(index)))
        nx.draw(graph, label='robot %.d' %(index[i]), pos=nx.get_node_attributes(graph, 'pos'),
                node_color=node_color,node_size=100,with_labels = True,font_color='w',font_size=8)
    
    plt.legend()
    plt.show()
    

def plotMeetingPaths(robots, index, team, subplot=None, length=0):

    if subplot != None:
        plt.figure('RRT* Paths')
        plt.subplot(np.ceil(length/2),2,subplot)
    else:
        plt.figure()
    
    plt.title('Team %.d' %team)
    cmap = plt.get_cmap('hsv')
    for i in range(0,len(index)):
        graph = robots[index[i]].totalPath
        node_color = colors.to_hex(cmap(i/len(index)))
        nx.draw(graph, label='robot %.d' %(index[i]), pos=nx.get_node_attributes(graph, 'pos'),
                node_color=node_color,node_size=100,with_labels = True,font_color='w',font_size=8)
    
    plt.legend()
    plt.show()