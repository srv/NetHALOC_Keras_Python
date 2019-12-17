# -*- coding: utf-8 -*-

###############################################################################
# Name        : MotionEstimator
# Description : Performs RANSAC 2D motion estimation on point clouds.
# Note        : Check example_motionestimator.py for an usage example.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################
import numpy as np
from transform2d import least_squares_cartesian,compose_point

class MotionEstimator:
    # Constructor.
    # Input  : ransacIterations - Number of RANSAC iterations
    #          numRandomSamples - Number of point pairs to be randomly selected
    #                             to build the initial model.
    #          maxAllowableSampleError - Maximum sample error to be considered
    #                                    inlier.
    #          minPointsInConsensusMultiplier - Percentage of samples that
    #                                    have to be in the consensus set to
    #                                    use it to build a candidate model.
    def __init__(self,ransacIterations=1000,numRandomSamples=3,maxAllowableSampleError=20,minPointsInConsensusMultiplier=.6):
        self.set_parameters(ransacIterations,numRandomSamples,maxAllowableSampleError,minPointsInConsensusMultiplier)
        self.Sref=None
        self.Scur=None

    # Static method. Measures the error introduced by theModel.
    # Input  : Sref - Reference point cloud
    #          Scur - Current point cloud
    #          theModel - Model to test
    #          theIndexes - What points in both point clouds use
    # Output : The error
    @staticmethod
    def _measure_error_(Sref,Scur,theModel,theIndexes):
        SScur=compose_point(theModel,Scur[:,theIndexes])
        tmp=SScur-Sref[:,theIndexes]
        return np.sqrt(np.sum(tmp*tmp,axis=0))

    # Parameter setter.
    # Input  : ransacIterations - Number of RANSAC iterations
    #          numRandomSamples - Number of point pairs to be randomly selected
    #                             to build the initial model.
    #          maxAllowableSampleError - Maximum sample error to be considered
    #                                    inlier.
    #          minPointsInConsensusMultiplier - Percentage of samples that
    #                                    have to be in the consensus set to
    #                                    use it to build a candidate model.
    def set_parameters(self,ransacIterations,numRandomSamples,maxAllowableSampleError,minPointsInConsensusMultiplier):
        self.ransacIterations=ransacIterations
        self.numRandomSamples=numRandomSamples
        self.maxAllowableSampleError=maxAllowableSampleError
        self.minPointsInConsensusMultiplier=minPointsInConsensusMultiplier
        
    # Do the RANSAC estimation
    # Input  : Sref - Reference point cloud. First row is X, second row is Y,
    #                 each column is a point.
    #          Scur - Current point cloud. Same format that Sref. Each point 
    #                 Scur[:,i] in Scur is associated with Sref[:,i]
    def estimate(self,Sref,Scur):
        bestError=np.Inf
        bestModel=np.zeros((3,1))
        bestConsensusSet=np.array([])
        allIndexes=np.array(range(Sref.shape[1]))
        for i in range(self.ransacIterations):
            # maybe_inliers := n randomly selected values from data
            np.random.shuffle(allIndexes)
            maybeInliers=allIndexes[:self.numRandomSamples]
            # maybe_model := model parameters fitted to maybe_inliers
            maybeModel=least_squares_cartesian(Sref[:,maybeInliers],Scur[:,maybeInliers])
            # consensus_set := maybe_inliers
            consensusSet=maybeInliers
            # for every point in data not in maybe_inliers 
            #   if point fits maybe_model with an error smaller than t
            #       add point to consensus_set
            notInMaybeInliers=allIndexes[self.numRandomSamples:]
            theError=MotionEstimator._measure_error_(Sref,Scur,maybeModel,notInMaybeInliers)
            consensusSet=np.append(consensusSet,notInMaybeInliers[np.where(theError<self.maxAllowableSampleError)])
            # if the number of elements in consensus_set is > d 
            #         (this implies that we may have found a good model,
            #         now test how good it is)      
            if consensusSet.shape[0]>Sref.shape[1]*self.minPointsInConsensusMultiplier:
                # this_model := model parameters fitted to all points in consensus_set          
                thisModel=least_squares_cartesian(Sref[:,consensusSet],Scur[:,consensusSet])
                # this_error := a measure of how well this_model fits these points
                thisError=np.sum(MotionEstimator._measure_error_(Sref,Scur,thisModel,consensusSet))
                # if this_error < best_error
                #             (we have found a model which is better than any of the previous ones,
                #             keep it until a better one is found)
                #             best_model := this_model
                #             best_consensus_set := consensus_set
                #             best_error := this_error        
                if thisError<bestError:
                    bestModel=thisModel
                    bestConsensusSet=consensusSet
                    bestError=thisError
        hasFailed=bestConsensusSet.size==0
        return bestModel,hasFailed,bestConsensusSet