# -*- coding: utf-8 -*-
###############################################################################
# Name        : ImageMatcher
# Description : Determines if two images overlap
# Notes       : See example_imagematcher.py for an usage example
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
from motionestimator import MotionEstimator

class ImageMatcher:
    # Constructor.
    # Input  : matchThreshold - Distance relationship between the first and
    #                           the second descriptor to consider the first
    #                           a good match.
    def __init__(self,matchThreshold=.75):
        self.matchThreshold=matchThreshold
        self._reset_()

    # Reset matcher state        
    def _reset_(self):
        self.iRef=None
        self.iCur=None
        self.kpRef=None
        self.kpCur=None
        self.dRef=None
        self.dCur=None
        self.theMatches=None
        self.theMotion=None
        self.Sref=None
        self.Scur=None
        self.hasFailed=True

    # Obtain SIFT features        
    def _get_sift_(self,theImage):
        gsImage=cv2.cvtColor(theImage,cv2.COLOR_RGB2GRAY)
        theSIFT=cv2.xfeatures2d.SIFT_create()
        return theSIFT.detectAndCompute(gsImage,None)
        
    # Get SIFT matches
    def _get_matches_(self,dRef,dCur):
        bf=cv2.BFMatcher()
        matches=bf.knnMatch(dRef,dCur,k=2)
        # Apply ratio test
        good=[]
        for m,n in matches:
            if m.distance < self.matchThreshold*n.distance:
                good.append(m)
        return np.array([[x.queryIdx,x.trainIdx] for x in good])

    # Image setter
    def define_images(self,iRef,iCur):
        self._reset_()
        self.iRef=iRef.astype('uint8')
        self.iCur=iCur.astype('uint8')
      
    # Do the estimation. The output is true (image overlap) or false (images
    # do not overlap). The estimated motion is self.theMotion
    def estimate(self):
        self.kpRef,self.dRef=self._get_sift_(self.iRef)
        self.kpCur,self.dCur=self._get_sift_(self.iCur)
        self.theMatches=self._get_matches_(self.dRef,self.dCur)
        if self.theMatches.size<(min(len(self.kpRef),len(self.kpCur))*0.1):
            self.hasFailed=True
        else:
            self.Sref=np.array([self.kpRef[i].pt for i in self.theMatches[:,0]]).transpose()
            self.Scur=np.array([self.kpCur[i].pt for i in self.theMatches[:,1]]).transpose()
            theEstimator=MotionEstimator()
            self.theMotion,self.hasFailed,self.bestConsensusSet=theEstimator.estimate(self.Sref,self.Scur)
            self.bestConsensusSet=self.bestConsensusSet.astype('int')
            if self.bestConsensusSet.size<self.theMatches.size*0.25:
                self.hasFailed=True
        return not self.hasFailed
        
    # Plot the consensus set
    def plot_consensus_set(self):
        allImages=np.hstack((self.iRef,self.iCur))
        colOffset=self.iRef.shape[1]
        theX=np.array([[self.kpRef[m[0]].pt[0],self.kpCur[m[1]].pt[0]+colOffset] for m in self.theMatches[self.bestConsensusSet,]]).transpose()
        theY=np.array([[self.kpRef[m[0]].pt[1],self.kpCur[m[1]].pt[1]] for m in self.theMatches[self.bestConsensusSet,]]).transpose()
        plt.figure()
        plt.imshow(allImages)
        plt.plot(theX,theY)
        plt.show()
        
    # Plot the initial matches
    def plot_matches(self):
        allImages=np.hstack((self.iRef,self.iCur))
        colOffset=self.iRef.shape[1]
        theX=np.array([[self.kpRef[m[0]].pt[0],self.kpCur[m[1]].pt[0]+colOffset] for m in self.theMatches[:,]]).transpose()
        theY=np.array([[self.kpRef[m[0]].pt[1],self.kpCur[m[1]].pt[1]] for m in self.theMatches[:,]]).transpose()
        plt.figure()
        plt.imshow(allImages)
        plt.plot(theX,theY)
        plt.show()