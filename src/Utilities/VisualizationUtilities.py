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

# def plotTrajectoryAnimation(robots, measurementGroundTruthList, modelEstimates):
#     """Make an animation of the robots journee

#     Input arguments:
#     robots = robots which measured and moved around
#     measurementGroundTruthList = measurement data which changes over time
#     modelEstimates = list of model updates and time
#     """

#     fig = plt.figure()
#     ax1 = plt.axes(xlim=(0, 600), ylim=(0,600))
    
#     graphs = []
#     for r in range(len(robots)):
#         graphobj = ax1.plot([], [], '-', label='Robot %d'%r)[0]
#         graphs.append(graphobj)
    
    
#     def init():
#         for graph in graphs:
#             graph.set_data([],[])
#         return graphs
    
#     xlist = [i for i in range(len(robots))]
#     ylist = [i for i in range(len(robots))]
    
#     def animate(i):
#         for r in range(0,len(robots)):
#             x,y = zip(*robots[r].trajectory)
#             xlist[r] = x[:i]
#             ylist[r] = y[:i]

#         for gnum,graph in enumerate(graphs):
            
#             graph.set_data(ylist[gnum], xlist[gnum]) # set data for each line separately. 
#         ax1.imshow(measurementGroundTruthList[i], origin='lower')
#         return graphs

#     ani = animation.FuncAnimation(fig, animate, init_func=init,
#                                   frames=len(robots[-1].trajectory), interval=200)
#     plt.legend(loc='lower right')
    
#     ani.save(PATH + 'basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

def plotTrajectoryAnimation(robots, measurementGroundTruthList, modelEstimates):
    """Make an animation of the robots journee

    Input arguments:
    robots = robots which measured and moved around
    measurementGroundTruthList = measurement data which changes over time
    modelEstimates = list of model updates and time
    """
    colors = ['b','m','g','r']

    fig, ax = plt.subplots(1,2,figsize=(12, 6))
    fig.subplots_adjust(bottom=0.06, right=0.8, top=0.94, wspace=0.2,hspace=0.1)
    fig.suptitle('Reduced-Order Modeling of a Dynamic Process \n under Intermittent Connectivity with Distributed Robot Teams')

    ax[0].set_xlim(0,600)
    ax[0].set_ylim(0,600)
    ax[0].set_title('Ground Truth')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    
    ax[1].set_xlim(0,600)
    ax[1].set_ylim(0,600)
    ax[1].set_title('Model')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    
    graphs = []
    arrows = []
    for r in range(len(robots)):
        graphobj = ax[0].plot([], [], '-', color=colors[r], label='Robot %d'%r)[0]
        arrow, = ax[0].plot([], [],'o', color=colors[r])
        graphs.append(graphobj)
        arrows.append(arrow)
    
    
    def init():
        for graph in graphs:
            graph.set_data([],[])
        timeText = ax[1].text(-0.2, 1.1,'', transform=ax[1].transAxes)   
        return graphs, timeText
    
    xlist = [i for i in range(len(robots))]
    ylist = [i for i in range(len(robots))]

    updatedModel, updatedTime = zip(*modelEstimates)
    updatedTime = np.around(updatedTime, decimals=1)

    im = ax[1].imshow(updatedModel[0], origin='lower', vmin=-1, vmax=15)

    timeText = ax[1].text(-0.2, 1.1,'', transform=ax[1].transAxes)  
    
    totalTime = len(robots[-1].trajectory)

    def animate(i):
        for r in range(0,len(robots)):
            x,y = zip(*robots[r].trajectory)
            xlist[r] = x[:i]
            ylist[r] = y[:i]

        for gnum,graph in enumerate(graphs):
            graph.set_data(ylist[gnum], xlist[gnum])
        if len(xlist[0]) > 1:
        # if len(xlist[0]) > 10:
            for arrowNb,arrow in enumerate(arrows):
                # print(ylist[arrowNb][-1], xlist[arrowNb][-1])
                # ax[0].arrow(ylist[arrowNb][-1],xlist[arrowNb][-1],ylist[arrowNb][-1]-ylist[arrowNb][-10],xlist[arrowNb][-1]-xlist[arrowNb][-10], shape='full', lw=0, length_includes_head=True, head_width=.05)
                arrow.set_data(ylist[arrowNb][-1], xlist[arrowNb][-1])
            
        ax[0].imshow(measurementGroundTruthList[i], origin='lower', vmin=-1, vmax=15)

        idx = np.where(updatedTime == np.round(i*0.1,decimals=1))
        if len(idx) > 0 and len(idx[0]) > 0:
            ax[1].imshow(updatedModel[idx[0][0]], origin='lower', vmin=-1, vmax=15)
        
        textstr = 'Time: ' + str(np.round(i*0.1,decimals=1))

        timeText.set_text(textstr)

        print('Progress: %.1f percent\r' %(i/totalTime*100), end='')

        return graphs, timeText, arrows

    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(robots[-1].trajectory), interval=500)
    
    cbar_ax = fig.add_axes([0.83, 0.2, 0.01, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    im.set_clim(-1, 15)
    cbar.ax.set_ylabel('Dye Concentration', rotation=270, labelpad=15)
    
    ax[0].legend(loc='lower left', bbox_to_anchor=(-0.17, -0.24), ncol=4)
            
    ani.save(PATH + 'video.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
    
def plotTrajectory(robots):
    """Plot the trajectories of all robots

    Input arguments:
    robots = robots which measured and moved around
    """

    plt.figure()
    for r in range(0,len(robots)):
        x,y = zip(*robots[r].trajectory)
        plt.plot(y,x, '-', label='Robot %d'%r)
    plt.legend()
    plt.savefig(PATH + 'Trajectory' + '.png')
    plt.close()

def plotMeasurement(data, title):
    """Plot the measurement data

    Input arguments:
    data = measurement data
    title = title of the figure
    """

    _, ax = plt.subplots()

    ax.set_title(title)
    plt.imshow(data, origin='lower')
    plt.colorbar()
    plt.savefig(PATH + 'Measurements' + '.png')
    plt.close()
    
    
def plotMeetingGraphs(robots, index, team, subplot=None, length=0):
    """Plot the trajectories of all robots

    Input arguments:
    robots = robots which measured and moved around
    index = which robots should be plotted
    team = team of the robots
    subplot = if we want to plot all teams
    length = how many subplots we need
    """

    #TODO: add savefig in main loop
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
    """Plot the trajectories of all robots

    Input arguments:
    robots = robots which measured and moved around
    index = which robots should be plotted
    team = team of the robots
    subplot = if we want to plot all teams
    length = how many subplots we need
    """

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
    """Plot the trajectories over the ground truth

    Input arguments:
    robots = robots which measured and moved around
    index = which robots ground truth to use
    """

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
    
def plotProcrustes(robot, image1, image2):
    """Plot the procrustes analysis results

    Input arguments:
    robot = robot instance
    image1 = result from procrustes
    image2 = result from procrustes
    """

    fig, ax = plt.subplots(1,4,figsize=(18, 6))

    dispTime = robot.currentTime

    title = 'Procrustes: Robot %d, Time %.1f' %(robot.ID,dispTime)
    fig.suptitle(title)

    ax[0].set_title('Expected Measurement')  
    im = ax[0].imshow(robot.expectedMeasurement, origin='lower')

    ax[1].set_title('Procrustes1')  
    im = ax[1].imshow(image1, origin='lower')   

    ax[2].set_title('Procrustes2')  
    im = ax[2].imshow(image2, origin='lower')      

    ax[3].set_title('GroundTruth') 

    im = ax[3].imshow(robot.mappingGroundTruth, origin='lower')

    fig.savefig(PATH + title + '.png')
    
    plt.close(fig)

def plotDye(image1, image2, image3):
    """Plot the dye at three different time steps

    Input arguments:
    image1 = dye at specific timestep
    image2 = dye at specific timestep
    image3 = dye at specific timestep
    """

    fig, ax = plt.subplots(1,3,figsize=(18, 6))

    title = 'DyePlot'

    im = ax[0].imshow(image1, origin='lower')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')

    im = ax[1].imshow(image2, origin='lower')   
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')

    im = ax[2].imshow(image3, origin='lower')      
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')
    fig.savefig(PATH + title + '.png')
    
    plt.close(fig)