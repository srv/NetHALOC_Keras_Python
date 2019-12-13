# -*- coding: utf-8 -*-
###############################################################################
# Name        : Tester
# Description : Tests a trained model
# Notes       : Please refer to example_tester.py for an usage example
# Author      : Antoni Burguera (antoni.burguera@uib.es)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

from skimage.transform import resize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
from imagematcher import ImageMatcher
import time

class Tester:
    
    # Constructor.
    # Input  : modelWrapper - The trained model
    #          testDataSet - DataSet used to perform test
    #          batchSize - Size of batches to test
    def __init__(self,modelWrapper=None,testDataSet=None,batchSize=5):
        self.modelWrapper=modelWrapper
        self.testDataSet=testDataSet
        self.batchSize=batchSize
        self.theMatcher=ImageMatcher()
        self.__reset__()
        
    # Resets the testers        
    def __reset__(self):
        self.densePredictions=None
        self.cnnPredictions=None
        self.theHR=None
        self.theAUC=None
        self.theStats=None
        self.theTimes=None
        
    # Setter for the model and the test dataset
    def set_params(self,modelWrapper,testDataSet):
        self.__reset__()
        self.modelWrapper=modelWrapper
        self.testDataSet=testDataSet

    # Makes the model the predict numImages obtained from the dataset using
    # the specified imageGetter. If useCNN=False, the output of the dense
    # layer is provided. It useCNN=True, the output of the las conv. layer is
    # provided.
    def predict_images(self,numImages,imgGetter,useCNN=True):
        inputShape=self.modelWrapper.inputShape
        nBatches=int(np.ceil(float(numImages)/float(self.batchSize)))
        if useCNN:
            theModel=self.modelWrapper.cnnModel
        else:
            theModel=self.modelWrapper.theModel
        thePredictions=np.zeros((numImages,theModel.layers[-1].output_shape[1]))
                                   
        for iBatch in range(nBatches):
            iMin=iBatch*self.batchSize
            iMax=min((iBatch+1)*self.batchSize,numImages)
            theImages=np.zeros((iMax-iMin,inputShape[0],inputShape[1],3))
            for iImage in range(iMin,iMax):
                theImages[iImage-iMin,]=resize(imgGetter(iImage),inputShape)
                thePredictions[iMin:iMax,]=self.modelWrapper.predict(theImages,useCNN)
        return thePredictions

    # Computes the hit ratio evolution, from pct=0 (just for completeness) to
    # pct=99%. For each pct, the best pct*numDBImages are selected. If a loop
    # is there, it is considered a hit.
    # Input : useCNN use conv layer (True) or dense output (False)
    # Output : The hit ratio evolution and the area under the hit ratio curve.
    def compute_hitratio_evolution(self,useCNN=True,distanceType='euclidean'):
        # fbf sortida CNN per totes les imatges de la dB
        dbPred=self.predict_images(self.testDataSet.numDBImages,self.testDataSet.get_dbimage,useCNN)
        # fbf sortida CNN per totes les imatges query
        qPred=self.predict_images(self.testDataSet.numQImages,self.testDataSet.get_qimage,useCNN)
        # fbf dbPred i qPred ha de donar el HALOC de la imatge
        theDistances=cdist(qPred,dbPred,distanceType)
        self.theHR=[]
        for pct in range(100):
            nItems=int(pct*self.testDataSet.numDBImages/100)
            self.theHR.append(self.__compute_hitratio__(theDistances,nItems))
        self.theAUC=np.sum(self.theHR)/10000
        return self.theHR,self.theAUC

    # Helper to compute individual hit ratio
    def __compute_hitratio__(self,theDistances,nItems):
        nFound=0
        for qIndex in range(self.testDataSet.numQImages):
            dbLoopCandidates=np.argsort(theDistances[qIndex,:])[:nItems]
            dbActualLoops=self.testDataSet.get_qloop(qIndex)
            foundLoop=(any(i in dbActualLoops for i in dbLoopCandidates))
            if foundLoop:
                nFound+=1
        theHitRatio=nFound*100/self.testDataSet.numQImages
        return theHitRatio

    # Plots the obtained hit ratio evolution.
    def plot_hitratio_evolution(self):
        if self.theHR==None:
            print('[ERROR] BEFORE PLOTTING, PLEASE CALL compute_hitratio_evolution')
            return
        plt.figure()
        plt.plot(self.theHR)
        plt.title('AUC='+str(int(self.theAUC*100)))
        plt.ylabel('Hit ratio (%)')
        plt.xlabel('Percentage of database')
        plt.show()

    # Computes full stats. Roughly speaking, selects the searchRatio best loop
    # candidates according to the NN and then searches loops there using
    # RANSAC. In this way, tp, fp, tn and fn (as well as processing times) are
    # obtained and returned.
    def compute_fullstats(self,useCNN=True,searchRatio=0.1,distanceType='euclidean'):
        print('[COMPUTING FULLSTATS]')
        print('  * COMPUTING DESCRIPTORS')
        tstart=time.time()
        # fbf canviar dbPRed i qPred pel calcul de HALOC
        dbPred=self.predict_images(self.testDataSet.numDBImages,self.testDataSet.get_dbimage,useCNN)
        qPred=self.predict_images(self.testDataSet.numQImages,self.testDataSet.get_qimage,useCNN)
        print('  * COMPUTING DISTANCES BETWEEN DESCRIPTORS')
        theDistances=cdist(qPred,dbPred,distanceType)
        tdist=time.time()-tstart
        #nItems=int(searchRatio*self.testDataSet.numDBImages) # the percentage of the images of the DB
        nItems=5
        print('  * THE CLOSEST '+str(nItems)+' DATABASE IMAGES WILL BE SEARCHED')
        print('  * STARTING TESTS')
        tp=tn=fp=fn=0
        tstart=time.time()
        for qIndex in range(self.testDataSet.numQImages): #per totes les imatges Query 
            dbLoopCandidates=np.argsort(theDistances[qIndex,:])[:nItems] # agafa els nItems amb menys distancia a la query 
            qImage=self.testDataSet.get_qimage(qIndex) # recupera la imatge query 
            dbActualLoops=self.testDataSet.get_qloop(qIndex) # busca totes les imatges de la dB que tanquen bucle amb la query qIndex
            for dbIndex in range(self.testDataSet.numDBImages): # per totes les imatges de la dB
                isLoop=dbIndex in dbActualLoops #mira si el dBIndex es dins del conjunt d'imatges que tanquen llaç
                foundLoop=False
                if dbIndex in dbLoopCandidates: # si la imatge dbIndex està dins dels 5 millors
                    dbImage=self.testDataSet.get_dbimage(dbIndex)
                    self.theMatcher.define_images(qImage,dbImage)
                    foundLoop=self.theMatcher.estimate() # hi ha loop closing RANSAC entre query i dB
                    del dbImage
                if foundLoop and isLoop:
                    tp+=1
                elif foundLoop and (not isLoop):
                    fp+=1
                elif (not foundLoop) and isLoop:
                    fn+=1
                elif (not foundLoop) and (not isLoop):
                    tn+=1
            print('    + COMPLETED '+str(qIndex+1)+' OF '+str(self.testDataSet.numQImages)+' QUERIES')
            del qImage
        tloops=time.time()-tstart
        print('  * TESTS FINISHED')
        self.theStats=(tp,fp,tn,fn)
        self.theTimes=(tdist,tloops)
        theAccuracy=(tp+tn)/(tp+tn+fp+fn)
        theTPR=tp/(tp+fn)
        theFPR=fp/(fp+tn)
        print('[FULLSTATS COMPUTED '+str(theAccuracy)+' '+str(theTPR)+' '+str(theFPR)+' '+str(tloops+tdist)+']')
        return tp,fp,tn,fn,tdist,tloops