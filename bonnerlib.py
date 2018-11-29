#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Plot the contours of the decision function (of classifier clf)
# and highlight the decision boundary in solid black.
# ax is the axes on which the contour plot is to be drawn.

def dfContour(clf,ax):
    
    # The extent of xy space
    x_min,x_max = ax.get_xlim()
    y_min,y_max = ax.get_ylim()
     
    # form a mesh/grid over xy space
    h = 0.02    # mesh granularity
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(),yy.ravel()]
    
    # evaluate the decision functrion at the grid points
    Z = clf.predict_proba(mesh)[:,1]
    
    # plot the contours of the decision function
    Z = Z.reshape(xx.shape)
    mylevels=np.linspace(0.0,1.0,11)
    ax.contourf(xx, yy, Z, levels=mylevels, cmap=cm.RdBu, alpha=0.5)
    
    # draw the decision boundary in solid black
    ax.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='solid')

# Plot the decision function of classifier clf in 3D.
# if Cflag=1 (the default), draw a contour plot of the decision function
# beneath the 3D plot.
# ax is the axes on which the function is to be drawn.

def df3D(clf,ax,cFlag=1):    
    
    # the extent of xy space
    x_min,x_max = ax.get_xlim()
    y_min,y_max = ax.get_ylim()
    
    # form a mesh/grid over xy space
    h = 0.01    # mesh granularity
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(),yy.ravel()]
    
    # evaluate the decision functrion at the grid points
    Z = clf.predict_proba(mesh)[:,1]
    
    # plot the contours of the decision function
    Z = Z.reshape(xx.shape)
    ax.plot_surface(xx, yy, Z, cmap=cm.RdBu, linewidth=0, rcount=75, ccount=75)
    mylevels=np.linspace(0.0,1.0,11)
    ax.contour(xx,yy,Z,levels=mylevels,linestyles='solid',linewidths=3,cmap=cm.RdBu)
    
    # limits of the vertical axis
    z_min = 0.0
    z_max = 1.0
    
    if cFlag == 1:
        # display a contour plot of the decision function
        z_min = -0.5
        ax.contourf(xx, yy, Z, levels=mylevels, cmap=cm.RdBu, offset=z_min)
    
    ax.set_zlim(z_min,z_max)
