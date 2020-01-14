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

PATH = 'Results/Tmp/'


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

    plt.savefig(PATH + 'MultivariateNormal' + '.png')
    plt.close()

def clearPlots():
    plt.close('all')

def plotTrajectoryAnimation(robots, measurementGroundTruthList):
    
    fig = plt.figure()
    ax1 = plt.axes(xlim=(0, 600), ylim=(0,600))
    
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
            
            graph.set_data(ylist[gnum], xlist[gnum]) # set data for each line separately. 
        ax1.imshow(measurementGroundTruthList[i], origin='lower')
        return graphs

    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(robots[-1].trajectory), interval=200)
    plt.legend()
    
    ani.save(PATH + 'basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    
def plotTrajectory(robots):
    plt.figure()
    for r in range(0,len(robots)):
        x,y = zip(*robots[r].trajectory)
        plt.plot(y,x, '-', label='Robot %d'%r)
    plt.legend()
    plt.savefig(PATH + 'Trajectory' + '.png')
    plt.close()

def plotMeasurement(data, title):
    
    _, ax = plt.subplots()

    ax.set_title(title)
    plt.imshow(data, origin='lower')
    plt.colorbar()
    plt.savefig(PATH + 'Measurements' + '.png')
    plt.close()
    
    
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
    plt.savefig(PATH + 'RRT* Graphs' + '.png')
    plt.close()
    

def plotMeetingPaths(robots, index, team, subplot=None, length=0):
    #TODO: add savefig in main loop
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
    plt.savefig(PATH + 'RRT* Paths' + '.png')
    plt.close()
        
    
def plotTrajectoryOverlayGroundTruth(robots, index):
    fig, ax = plt.subplots()
    meetingList = []
    for r in range(0,len(robots)):
        x,y = zip(*robots[r].trajectory)
        ax.plot(y,x, '-', label='Robot %d'%r)
        
        posMeeting, _  = zip(*robots[r].meetings)
        posMeeting = np.array(posMeeting)
        meetingList.append(posMeeting)

    meetingList = np.vstack(meetingList)
    ax.plot(meetingList[:, 1],meetingList[:, 0], 'o', color='r',label='Meetings',markersize=6)
    
    data = robots[index].mappingGroundTruth
        
    im = ax.imshow(data, origin='lower', extent=[0, 600, 0, 600])
    
    ax.set_title('Ground Truth and Trajectories')       
    fig.colorbar(im, ax=ax)
    plt.legend()
    plt.savefig(PATH + 'Ground Truth and Trajectories' + '.png')
    plt.close(fig)
    
def plot3Images(robot, image3, image4):

    fig, ax = plt.subplots(1,4,figsize=(18, 6))
    # fig.subplots_adjust(left=0.02, bottom=0.06, right=0.8, top=0.94, wspace=0.12,hspace=0.1)

    dispTime = robot.currentTime

    title = 'Procrustes: Robot %d, Time %.1f' %(robot.ID,dispTime)
    fig.suptitle(title)

    ax[0].set_title('Expected Measurement')  
    im = ax[0].imshow(robot.expectedMeasurement, origin='lower')

    ax[1].set_title('Procrustes1')  
    im = ax[1].imshow(image3, origin='lower')   

    ax[2].set_title('Procrustes2')  
    im = ax[2].imshow(image4, origin='lower')      

    ax[3].set_title('GroundTruth') 

    im = ax[3].imshow(robot.mappingGroundTruth, origin='lower')

    fig.savefig(PATH + title + '.png')
    
    plt.close(fig)