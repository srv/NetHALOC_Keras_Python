# -*- coding: utf-8 -*-

###############################################################################
# Name        : example_motionestimator
# Description : A basic example on how to use the MotionEstimator class
# Notes       : Just run the script.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

from motionestimator import MotionEstimator
from transform2d import least_squares_cartesian,compose_point
import numpy as np
import matplotlib.pyplot as plt

# Define some parameters
nPoints=100
minVal=-10
maxVal=10
pctNoise=.25

# Build Sref randomly. Translate and rotate it to build Scur. In this way,
# we ensure that each point in Scur is associated with the point in the
# same index in Sref, and also that the motion between the two point clouds
# is known (since we apply it). Also, some points are synthetically changed
# to simulate wrong correspondences.
print('[[ BUILDING RANDOM POINT CLOUDS ]]')
Sref=np.random.random((2,nPoints))*(maxVal-minVal)+minVal
syntheticMotion=np.array([2,2,np.pi/4])
Scur=compose_point(syntheticMotion,Sref)
npToChange=int(pctNoise*nPoints)
Sref[:,:npToChange]=np.random.random((2,npToChange))*(maxVal-minVal)+minVal
Scur[:,:npToChange]=np.random.random((2,npToChange))*(maxVal-minVal)+minVal
print('[[ POINT CLOUDS BUILT ]]')

# Let's plot them
print('[[ PLOTTING THE POINT CLOUDS ]]')
plt.figure()
plt.plot(Sref[0,:],Sref[1,:],'ro')
plt.plot(Scur[0,:],Scur[1,:],'k.')
theX=np.array([Sref[0,npToChange:],Scur[0,npToChange:]])
theY=np.array([Sref[1,npToChange:],Scur[1,npToChange:]])
plt.plot(theX,theY,'k')
plt.legend(['Sref','Scur'])
plt.show()
print('[[ POINT CLOUDS PLOTTED ]]')

# Let's estimate the motion with just least squares, transform Scur according
# to the estimation and plot it.
print('[[ COMPUTING AND PLOTTING LEAST SQUARES CARTESIAN ]]')
lsMotion=least_squares_cartesian(Sref,Scur)
nScur=compose_point(lsMotion,Scur)
plt.figure()
plt.plot(Sref[0,:],Sref[1,:],'ro')
plt.plot(nScur[0,:],nScur[1,:],'k.')
theX=np.array([Sref[0,npToChange:],nScur[0,npToChange:]])
theY=np.array([Sref[1,npToChange:],nScur[1,npToChange:]])
plt.plot(theX,theY,'k')
plt.legend(['Sref','nScur (with least squares)'])
plt.show()
print('[[ LEAST SQUARES CARTESIAN COMPUTED AND PLOTTED ]]')

# Same as before, but now using RANSAC
print('[[ COMPUTING AND PLOTTING RANSAC ]]')
motionEstimator=MotionEstimator()
ransacMotion,hasFailed,bestConsensusSet=motionEstimator.estimate(Sref,Scur)
if hasFailed:
    print('[ OOPS... SOMETHING WENT WRONG. RANSAC DID NOT FOUND A SOLUTION ]')
nScur=compose_point(ransacMotion,Scur)
plt.figure()
plt.plot(Sref[0,:],Sref[1,:],'ro')
plt.plot(nScur[0,:],nScur[1,:],'k.')
theX=np.array([Sref[0,npToChange:],nScur[0,npToChange:]])
theY=np.array([Sref[1,npToChange:],nScur[1,npToChange:]])
plt.plot(theX,theY,'k')
plt.legend(['Sref','nScur (with RANSAC)'])
plt.show()
print('[[ RANSAC COMPUTED AND PLOTTED ]]')