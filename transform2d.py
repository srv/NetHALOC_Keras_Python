# -*- coding: utf-8 -*-

###############################################################################
# Name        : transform2d
# Description : Some auxiliary functions related to 2D transformations
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

import numpy as np
import math

# Motion that minimizes sum of squared distances
# Given two 2D point clouds (Sref and Scur) it computes the 2D motion so that
# when applied to Scur the sum of squared distances between points in minimum.
# Sref and Scur must have the same number of points, and the i-th point
# in Scur must correspond to the i-th point in Sref.
# Sref and Scur have 2 rows (x,y) and N columns (the points).
def least_squares_cartesian(Sref,Scur):
    mx=np.mean(Scur[0,:])
    my=np.mean(Scur[1,:])
    mx2=np.mean(Sref[0,:])
    my2=np.mean(Sref[1,:])
    Sxx=np.sum((Scur[0,:]-mx)*(Sref[0,:]-mx2))
    Syy=np.sum((Scur[1,:]-my)*(Sref[1,:]-my2))
    Sxy=np.sum((Scur[0,:]-mx)*(Sref[1,:]-my2))
    Syx=np.sum((Scur[1,:]-my)*(Sref[0,:]-mx2))  
    o=math.atan2(Sxy-Syx,Sxx+Syy)
    x=mx2-(mx*math.cos(o)-my*math.sin(o))
    y=my2-(mx*math.sin(o)+my*math.cos(o))
    return np.array([[x],[y],[o]])
    
# Point composition
# Given a 2D transformation X1=[x,y,theta], this function applies it to
# all the points in X2.
# X2 has 2 rows (x,y) and N columns (the points)
def compose_point(X1,X2):
    s=math.sin(X1[2])
    c=math.cos(X1[2])
    return np.vstack((X1[0]+np.matmul([c,-s],X2),X1[1]+np.matmul([s,c],X2)))