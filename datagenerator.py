# -*- coding: utf-8 -*-
###############################################################################
# Name        : DataGenerator
# Description : Data generators useable by Keras. It contains several loop
#               closing data generators. Each one is explained in its own
#               comments.
# Note        : See example_datagenerator.py to understand how they work.
#               Nevertheless, the data generators are prepared to be used
#               by fit_generator in Keras, not to be used standalone.
# Note        : Yes, inheritance and polymorphism and all kind of OO stuff
#               could have been used. I know.
# Author      : Antoni Burguera (antoni.burguera@uib.es) and Francisco Bonin-Font (francisco.bonin@uib.es)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

from keras.utils import Sequence
import numpy as np  
from skimage.transform import resize
from skimage.transform import SimilarityTransform
from skimage.transform import warp
import cv2
from skimage.feature import hog
from sklearn.cluster import KMeans
from os.path import exists

###############################################################################
# Data generator that builds synthetic loops from the images in a datase and
# extracts HOG or HALOC features.
# Operation : For each image in the dataset, it builds another image that
#             synthetically closes a loop. This is achieved by translating,
#             rotating and scaling randomly the image.
#             One of the images (original or modified) is randomly selected
#             and returned as it is (only scaled to a specific size).
#             The other image is processed and HOG/HALOC descriptors are obtained.
#             The output of the generator is a batch where the data info is
#             the image and the HOG/HALOC descriptors.
# Note      : The size of the HOG descriptors depends on the image size and
#             the specific HOG configuration. If needed to define a Keras model
#             it has to be determined experimentally (i.e. the object does not
#             provide any method to compute it).
###############################################################################
class DataGeneratorHOGImages(Sequence):
    def __init__(self,dataSet,batchSize=20,imgSize=(240,320),scaleMin=0.9,scaleMax=1.1,rotateMin=-90,rotateMax=90,
                 txMin=-0.25,txMax=0.25,tyMin=-0.25,tyMax=0.25):
        self.dataSet=dataSet
        self.batchSize=batchSize
        self.imgSize=imgSize
        self.scaleMin=scaleMin
        self.scaleMax=scaleMax
        self.rotateMin=rotateMin
        self.rotateMax=rotateMax
        self.txMin=txMin
        self.txMax=txMax
        self.tyMin=tyMin
        self.tyMax=tyMax

    def __len__(self):
        return int(np.ceil(self.dataSet.numImages/float(self.batchSize)))

    def __getitem__(self,theIndex):
        X=[]
        y=[]
        bStart=max(theIndex*self.batchSize,0)
        bEnd=min((theIndex+1)*self.batchSize,self.dataSet.numImages)
        for i in range(bStart,bEnd):
            firstImage=self.dataSet.get_image(i)
            secondImage=self.__random_transform__(firstImage)
            theImage,theFeatures=self.__gettuple__([firstImage,secondImage])
            X.append(theImage)
            y.append(theFeatures)
        return np.array(X),np.array(y)
    
    def __gettuple__(self,theLoop):
        theImages=(resize(theLoop[0],self.imgSize),resize(theLoop[1],self.imgSize))
        idxHog=int(np.round(np.random.uniform()))
        idxImage=1-idxHog
        hogFeatures=hog(theImages[idxHog],orientations=8,pixels_per_cell=(10,10),cells_per_block=(1, 1),multichannel=True)
        return theImages[idxImage],hogFeatures
    
    def __image_transform__(self,theImage,scaleFactor,rotationAngle,txFactor,tyFactor):
        centerY,centerX=np.array(theImage.shape[:2])/2.
        theRotation=SimilarityTransform(rotation=np.deg2rad(rotationAngle))
        theZoom=SimilarityTransform(scale=scaleFactor)
        theShift=SimilarityTransform(translation=[-centerX,-centerY])
        theShiftInv=SimilarityTransform(translation=[centerX,centerY])
        theTranslation=SimilarityTransform(translation=[txFactor*2*centerX,tyFactor*2*centerY])
        return warp(theImage, (theShift+(theRotation+theShiftInv))+(theShift+(theZoom+theShiftInv))+theTranslation, mode='reflect')

    def __random_transform__(self,theImage):
        scaleFactor=np.random.uniform(self.scaleMin,self.scaleMax)
        rotationAngle=np.random.uniform(self.rotateMin,self.rotateMax)
        txFactor=np.random.uniform(self.txMin,self.txMax)
        tyFactor=np.random.uniform(self.tyMin,self.tyMax)
        return self.__image_transform__(theImage,scaleFactor,rotationAngle,txFactor,tyFactor)

###############################################################################
# Data generator that uses the existing loops in the database and uses
# HOG descriptors.
# Operation : Same as DataGeneratorHOGImages except that loops are not built
#             synthetically but they come from the dataset itself.
###############################################################################
class DataGeneratorHOGLoops(Sequence):
    def __init__(self,dataSet,batchSize=20,imgSize=(240,320)):
        self.dataSet=dataSet
        self.batchSize=batchSize
        self.imgSize=imgSize

    def __len__(self):
        return int(np.ceil(self.dataSet.numLoops/float(self.batchSize)))

    def __getitem__(self,theIndex):
        X=[]
        y=[]
        bStart=max(theIndex*self.batchSize,0)
        bEnd=min((theIndex+1)*self.batchSize,self.dataSet.numLoops)
        imagePairs=[self.dataSet.get_loop(i) for i in range(bStart,bEnd)]
        for curPair in imagePairs:
            theImage,theFeatures=self.__gettuple__(curPair)
            X.append(theImage)
            y.append(theFeatures)
        return np.array(X),np.array(y)
    
    def __gettuple__(self,theLoop):
        theImages=(resize(theLoop[0],self.imgSize),resize(theLoop[1],self.imgSize))
        idxHog=int(np.round(np.random.uniform()))
        idxImage=1-idxHog
        hogFeatures=hog(theImages[idxHog],orientations=8,pixels_per_cell=(10,10),cells_per_block=(1, 1),multichannel=True)
        return theImages[idxImage],hogFeatures

    
###############################################################################
# Data generator that builds synthetic loops from the images in a datase and
# extracts HALOC features.
# Operation : For each image in the dataset, it builds another image that
#             synthetically closes a loop. This is achieved by translating,
#             rotating and scaling randomly the image.
#             One of the images (original or modified) is randomly selected
#             and returned as it is (only scaled to a specific size).
#             The other image is processed and HALOC descriptors are obtained.
#             The output of the generator is a batch where the data info is
#             the image and the HALOC descriptors.
# Note      : The size of the HALOC descriptors depends on the image size and
#             the specific HALOC configuration. If needed to define a Keras model
#             it has to be determined experimentally (i.e. the object does not
#             provide any method to compute it).
###############################################################################
    
class DataGeneratorHALOCImages(Sequence):
    def __init__(self,dataSet,batchSize=20,imgSize=(240,320),scaleMin=0.9,scaleMax=1.1,rotateMin=-90,rotateMax=90,
                 txMin=-0.25,txMax=0.25,tyMin=-0.25,tyMax=0.25,numDesc=128):
        vector1,vector2,vector3=self.__calculatevectors__(nmaxf=100)
        self.dataSet=dataSet
        self.batchSize=batchSize
        self.imgSize=imgSize
        self.scaleMin=scaleMin
        self.scaleMax=scaleMax
        self.rotateMin=rotateMin
        self.rotateMax=rotateMax
        self.txMin=txMin
        self.txMax=txMax
        self.tyMin=tyMin
        self.tyMax=tyMax
        self.numDesc=numDesc
        self.vector1=vector1
        self.vector2=vector2
        self.vector3=vector3
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.dataSet.numImages/float(self.batchSize)))

    def __getitem__(self,theIndex):
        X=[]
        y=[]
        bStart=max(theIndex*self.batchSize,0)
        bEnd=min((theIndex+1)*self.batchSize,self.dataSet.numImages)
        for i in range(bStart,bEnd):
            firstImage=self.dataSet.get_image(i)
            secondImage=self.__random_transform__(firstImage)
            theImage,theFeatures=self.__gettuple__([firstImage,secondImage])
            X.append(theImage)
            y.append(theFeatures)
        return np.array(X),np.array(y)
       
    def __gettuple__(self,theLoop):
        theImages=(resize(theLoop[0],self.imgSize),resize(theLoop[1],self.imgSize)) # fa un resize de les dues imatges rebudes
        idxHALOC=int(np.round(np.random.uniform()))
        idxImage=1-idxHALOC
        halocFeatures=self.__get_descriptors__(theImages[idxHALOC])
        return theImages[idxImage],halocFeatures
 
    def __image_transform__(self,theImage,scaleFactor,rotationAngle,txFactor,tyFactor):
        centerY,centerX=np.array(theImage.shape[:2])/2.
        theRotation=SimilarityTransform(rotation=np.deg2rad(rotationAngle))
        theZoom=SimilarityTransform(scale=scaleFactor)
        theShift=SimilarityTransform(translation=[-centerX,-centerY])
        theShiftInv=SimilarityTransform(translation=[centerX,centerY])
        theTranslation=SimilarityTransform(translation=[txFactor*2*centerX,tyFactor*2*centerY])
        return warp(theImage, (theShift+(theRotation+theShiftInv))+(theShift+(theZoom+theShiftInv))+theTranslation, mode='reflect')

    def __random_transform__(self,theImage):
        scaleFactor=np.random.uniform(self.scaleMin,self.scaleMax)
        rotationAngle=np.random.uniform(self.rotateMin,self.rotateMax)
        txFactor=np.random.uniform(self.txMin,self.txMax)
        tyFactor=np.random.uniform(self.tyMin,self.tyMax)
        outImage=self.__image_transform__(theImage,scaleFactor,rotationAngle,txFactor,tyFactor)
        return outImage

    def __get_descriptors__(self,theImage): # theImage ja es un objecte imatge en format opencv
        hash=np.zeros((384))
        num_max_fea=100
        # This norm. is just to prevent slightly larger than one values
        theImage/=np.max(theImage)
        ubImage=(theImage*255).astype('uint8')
        gsImage=cv2.cvtColor(ubImage,cv2.COLOR_RGB2GRAY) # convert to gray scale before computing SIFT
        theSIFT=cv2.xfeatures2d.SIFT_create(((num_max_fea)-3)) # crea un objecte tipus SIFT que penja de xfeatures2d
        keyPoints,theDescriptors=theSIFT.detectAndCompute(gsImage,None) # detecta els keypoints i els descriptors, sense mascara
        nbr_of_keypoints=len(keyPoints)
        if nbr_of_keypoints==0:
            print("ERROR: descriptor Matrix is Empty")
            return 
        if nbr_of_keypoints>len(self.vector1):
            print("ERROR:  The number of descriptors is larger than the size of the projection vector. This should not happen.")
            return
        num_of_descriptors=theDescriptors.shape[0] #--> 100
        num_of_components=theDescriptors.shape[1] # --> 128
 #   print (num_of_descriptors)
        hash=[]
        dot = 0
        dot_normalized=0
        suma = 0
        for i in range(num_of_components):
            suma=0
            for j in range(num_of_descriptors):
                dot = theDescriptors[j,i]*self.vector1[j] # for a fixed component, (a fixed column) vary the descriptor (row): dot product 
    #between the matrix column and the vector
                dot_normalized = (dot + 1.0) / 2.0
                suma = suma + dot_normalized

            hash=np.append(hash, (suma/num_of_descriptors))   
        for i in range(num_of_components):
            suma=0
            for j in range(num_of_descriptors):
                dot = theDescriptors[j,i]*self.vector2[j] # for a fixed component, (a fixed column) vary the descriptor (row): dot product 
    #between the matrix column and the vector
                dot_normalized = (dot + 1.0) / 2.0
                suma = suma + dot_normalized

            hash=np.append(hash, (suma/num_of_descriptors))   
        for i in range(num_of_components):
            suma=0
            for j in range(num_of_descriptors):
                dot = theDescriptors[j,i]*self.vector3[j] # for a fixed component, (a fixed column) vary the descriptor (row): dot product 
    #between the matrix column and the vector
                dot_normalized = (dot + 1.0) / 2.0
                suma = suma + dot_normalized

            hash=np.append(hash, (suma/num_of_descriptors))   
        return hash
        
    def __calculatevectors__(self,nmaxf):
        num_max_features=nmaxf
         # get the 3 orthogonal unitary vectors
        vector1=np.random.uniform(0,1,num_max_features); # crea un vector de nombres aleatoris, entre 0 i 1
        vector1 /= np.linalg.norm(vector1) # normalitzo vector1
        # ara vull crear dos vectors mes, que siguin ortogonals al vector1 i unitaris
        #vector2  = np.random.uniform(0,1,num_max_features); # second random vector
        #vector2 /= np.linalg.norm(vector2) # normalitzo vector2
        vector2 = np.random.uniform(0,1,(num_max_features-1)); # second random vector, una component menys
        const1=0
        long=num_max_features-1
        for i in range(long): # dot product between vector2 and vector 1 for the num_max_features-1 components
            const1=const1+(vector1[i]*vector2[i])

        xn=-const1/vector1[num_max_features-1] # the last component of vector2 i the one that makes vector1·vector2=0
        vector2=np.append(vector2, xn) # add the last component to vector2. Now, vector 1 and vector 2 are orthogonals

        #vector2 -= vector2.dot(vector1) * vector1  # faig vector2 ortogonal a vector 1
        vector2 /= np.linalg.norm(vector2) # normalitzo vector2 otra vez
        vector3 = np.random.uniform(0,1,(num_max_features-2)); # create another vector , random and unitary
        #len(vector2), len(vector1), len(vector3)

        # trec un vector 3 ortogonal a vector1 i a vector2, forçant totes les 
        # components aleatories excepte les dues darreres, que seran el resultat de resoldre un sistema de 2 equacions amb 
        # 2 incògnites on el producte escalar ha de ser 0 amb vector1 i vector2. 

        const1=0
        const2=0
        long=num_max_features-2
        for i in range(long): # dot product between vector3 and the num_max_features-2 components of vector1 and vector2
            const1=const1+(vector1[i]*vector3[i])
            const2=const2+(vector2[i]*vector3[i])

        # force the last two elements of vector3 to be orthogonal to vector1 and vector2. Solve a linear system of 
        # equations Ax=B, where A --> the last two components of vector1 and vector2, in the form of
        # two rows of A, row 1 = vector1, row 2 = vector2. B is the constant components, taken from the 
        # dot product between the first num_max_features-2 components of vector1 and the num_max_features-2 components of vector2, 
        # with all the random components of vector3. And X are the last two components of vector3, in such a way that
        # vector1 · vector3=0 and vector2 · vector3=0. 
        A = np.array([[vector1[num_max_features-2],vector1[num_max_features-1]], [vector2[num_max_features-2],vector2[num_max_features-1]]])
        B = np.array([-const1,-const2])
        X = np.linalg.solve(A, B) # solve the linear system. X[0] is the penultimate element of vector3 ,X[1] is the last element of vector3
        #np.allclose(np.dot(A, X), B) # true if Ax=B

        vector3=np.append(vector3, X[0]) ## append the last two elements to vector3
        vector3=np.append(vector3, X[1])
        vector3 /= np.linalg.norm(vector3) # normalitzo vector3

          #  print ("lengh of vector3: "+str(len(vector3)))

        print(np.linalg.norm(vector1), np.linalg.norm(vector2), np.linalg.norm(vector3))
        print(np.dot(vector1, vector2) , np.dot(vector3, vector2) , np.dot(vector1, vector3))
        return vector1, vector2, vector3
    
    
    def on_epoch_end(self):
        np.random.seed(0)

###############################################################################
# Data generator that uses the existing loops in the database and uses
# the hash HALOC as the descriptor.
# Operation : Same as DataGeneratorHALOCImages except that loops are not built
#             synthetically but they come from the dataset itself.
###############################################################################
class DataGeneratorHALOCLoops(Sequence):
    def __init__(self,dataSet,batchSize=20,imgSize=(240,320),numDesc=128):
        #nmaxf: nombre màxim de features a tenir en compte pel càlcul de HALOC
        vector1,vector2,vector3=self.__calculatevectors__(nmaxf=100)
        self.dataSet=dataSet
        self.batchSize=batchSize
        self.imgSize=imgSize
        self.numDesc=numDesc
        self.vector1=vector1
        self.vector2=vector2
        self.vector3=vector3
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.dataSet.numLoops/float(self.batchSize)))

    def __getitem__(self,theIndex):
        X=[]
        y=[]
        bStart=max(theIndex*self.batchSize,0)
        bEnd=min((theIndex+1)*self.batchSize,self.dataSet.numLoops)
        imagePairs=[[self.dataSet.get_loop(i),i] for i in range(bStart,bEnd)]
        for curPair in imagePairs:
            theImage,theFeatures=self.__gettuple__(curPair)
            X.append(theImage)
            y.append(theFeatures)
        return np.array(X),np.array(y)
    
    
    def __gettuple__(self,theLoop):
        theImages=(resize(theLoop[0][0],self.imgSize),resize(theLoop[0][1],self.imgSize)) 
        idxHALOC=int(np.round(np.random.uniform()))
        idxImage=1-idxHALOC
        halocFeatures=self.__get_descriptors__(theImages[idxHALOC])
        return theImages[idxImage],halocFeatures
    
   
    def __get_descriptors__(self,theImage): 
        hash=np.zeros((384))
        num_max_fea=100
        # This norm. is just to prevent slightly larger than one values
        theImage/=np.max(theImage)
        ubImage=(theImage*255).astype('uint8')
        gsImage=cv2.cvtColor(ubImage,cv2.COLOR_RGB2GRAY) # convert to gray scale before computing SIFT
        theSIFT=cv2.xfeatures2d.SIFT_create(((num_max_fea)-3)) 
        keyPoints,theDescriptors=theSIFT.detectAndCompute(gsImage,None) 
        nbr_of_keypoints=len(keyPoints)
        if nbr_of_keypoints==0:
            print("ERROR: descriptor Matrix is Empty")
            return 
        if nbr_of_keypoints>len(self.vector1):
            print("ERROR:  The number of descriptors is larger than the size of the projection vector. This should not happen.")
            return
        num_of_descriptors=theDescriptors.shape[0] #--> 100
        num_of_components=theDescriptors.shape[1] # --> 128
        hash=[]
        dot = 0
        dot_normalized=0
        suma = 0
        for i in range(num_of_components):
            suma=0
            for j in range(num_of_descriptors):
                dot = theDescriptors[j,i]*self.vector1[j] # for a fixed component, (a fixed column) vary the descriptor (row): dot product 
    #between the matrix column and the vector
                dot_normalized = (dot + 1.0) / 2.0
                suma = suma + dot_normalized

            hash=np.append(hash, (suma/num_of_descriptors))   
        for i in range(num_of_components):
            suma=0
            for j in range(num_of_descriptors):
                dot = theDescriptors[j,i]*self.vector2[j] # for a fixed component, (a fixed column) vary the descriptor (row): dot product 
    #between the matrix column and the vector
                dot_normalized = (dot + 1.0) / 2.0
                suma = suma + dot_normalized

            hash=np.append(hash, (suma/num_of_descriptors))   
        for i in range(num_of_components):
            suma=0
            for j in range(num_of_descriptors):
                dot = theDescriptors[j,i]*self.vector3[j] # for a fixed component, (a fixed column) vary the descriptor (row): dot product 
    #between the matrix column and the vector
                dot_normalized = (dot + 1.0) / 2.0
                suma = suma + dot_normalized

            hash=np.append(hash, (suma/num_of_descriptors))   
        return hash
        
    def __calculatevectors__(self,nmaxf):
        num_max_features=nmaxf
         # get the 3 orthogonal unitary vectors
        vector1=np.random.uniform(0,1,num_max_features); # creates a vector of random numbers between 0 and 1
        vector1 /= np.linalg.norm(vector1) # normalize vector1
        vector2 = np.random.uniform(0,1,(num_max_features-1)); # second random vector, one component less
        const1=0
        long=num_max_features-1
        for i in range(long): # dot product between vector2 and vector 1 for the num_max_features-1 components
            const1=const1+(vector1[i]*vector2[i])

        xn=-const1/vector1[num_max_features-1] # the last component of vector2 i the one that makes vector1·vector2=0
        vector2=np.append(vector2, xn) # add the last component to vector2. Now, vector 1 and vector 2 are orthogonals

        vector2 /= np.linalg.norm(vector2) # normalize vector2 
        vector3 = np.random.uniform(0,1,(num_max_features-2)); # create another vector , random and unitary
        #len(vector2), len(vector1), len(vector3)

      
        const1=0
        const2=0
        long=num_max_features-2
        for i in range(long): # dot product between vector3 and the num_max_features-2 components of vector1 and vector2
            const1=const1+(vector1[i]*vector3[i])
            const2=const2+(vector2[i]*vector3[i])

        # force the last two elements of vector3 to be orthogonal to vector1 and vector2. Solve a linear system of 
        # equations Ax=B, where A --> the last two components of vector1 and vector2, in the form of
        # two rows of A, row 1 = vector1, row 2 = vector2. B is the constant components, taken from the 
        # dot product between the first num_max_features-2 components of vector1 and the num_max_features-2 components of vector2, 
        # with all the random components of vector3. And X are the last two components of vector3, in such a way that
        # vector1 · vector3=0 and vector2 · vector3=0. 
        A = np.array([[vector1[num_max_features-2],vector1[num_max_features-1]], [vector2[num_max_features-2],vector2[num_max_features-1]]])
        B = np.array([-const1,-const2])
        X = np.linalg.solve(A, B) # solve the linear system. X[0] is the penultimate element of vector3 ,X[1] is the last element of vector3
        

        vector3=np.append(vector3, X[0]) ## append the last two elements to vector3
        vector3=np.append(vector3, X[1])
        vector3 /= np.linalg.norm(vector3) # normalize vector3

        print(np.linalg.norm(vector1), np.linalg.norm(vector2), np.linalg.norm(vector3))
        print(np.dot(vector1, vector2) , np.dot(vector3, vector2) , np.dot(vector1, vector3))
        return vector1, vector2, vector3
    
    def on_epoch_end(self):
        np.random.seed(0)
        
###############################################################################
# Data generator that builds synthetic loops from the images in a datase and
# extracts SIFT features.
# Operation : Same as DataGeneratorHOGImages except that SIFT features are
#             used instead of HOG. As for SIFT, the obtained SIFT descriptors
#             are clustered (KMeans) into numDesc groups. The resulting
#             centroids are sorted by the number of associated descriptors in
#             descending order and the first numDesc are selected. If there
#             are not numDesc descriptors, the non-existing are set to zero.
###############################################################################
class DataGeneratorSIFTImages(Sequence): 
    def __init__(self,dataSet,batchSize=20,imgSize=(240,320),scaleMin=0.9,scaleMax=1.1,rotateMin=-90,rotateMax=90,
                 txMin=-0.25,txMax=0.25,tyMin=-0.25,tyMax=0.25,numDesc=128):
        self.dataSet=dataSet
        self.batchSize=batchSize
        self.imgSize=imgSize
        self.scaleMin=scaleMin
        self.scaleMax=scaleMax
        self.rotateMin=rotateMin
        self.rotateMax=rotateMax
        self.txMin=txMin
        self.txMax=txMax
        self.tyMin=tyMin
        self.tyMax=tyMax
        self.numDesc=numDesc
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.dataSet.numImages/float(self.batchSize)))

    def __getitem__(self,theIndex):
        X=[]
        y=[]
        bStart=max(theIndex*self.batchSize,0)
        bEnd=min((theIndex+1)*self.batchSize,self.dataSet.numImages)
        for i in range(bStart,bEnd):
            firstImage=self.dataSet.get_image(i)
            secondImage=self.__random_transform__(firstImage)
            theImage,theFeatures=self.__gettuple__([firstImage,secondImage])
            X.append(theImage)
            y.append(theFeatures)
        return np.array(X),np.array(y)
       
    def __gettuple__(self,theLoop):
        theImages=(resize(theLoop[0],self.imgSize),resize(theLoop[1],self.imgSize))
        idxSIFT=int(np.round(np.random.uniform()))
        idxImage=1-idxSIFT
        siftFeatures=self.__get_descriptors__(theImages[idxSIFT])
        return theImages[idxImage],siftFeatures
    
    
    def __image_transform__(self,theImage,scaleFactor,rotationAngle,txFactor,tyFactor):
        centerY,centerX=np.array(theImage.shape[:2])/2.
        theRotation=SimilarityTransform(rotation=np.deg2rad(rotationAngle))
        theZoom=SimilarityTransform(scale=scaleFactor)
        theShift=SimilarityTransform(translation=[-centerX,-centerY])
        theShiftInv=SimilarityTransform(translation=[centerX,centerY])
        theTranslation=SimilarityTransform(translation=[txFactor*2*centerX,tyFactor*2*centerY])
        return warp(theImage, (theShift+(theRotation+theShiftInv))+(theShift+(theZoom+theShiftInv))+theTranslation, mode='reflect')

    def __random_transform__(self,theImage):
        scaleFactor=np.random.uniform(self.scaleMin,self.scaleMax)
        rotationAngle=np.random.uniform(self.rotateMin,self.rotateMax)
        txFactor=np.random.uniform(self.txMin,self.txMax)
        tyFactor=np.random.uniform(self.tyMin,self.tyMax)
        outImage=self.__image_transform__(theImage,scaleFactor,rotationAngle,txFactor,tyFactor)
        return outImage

    def __get_descriptors__(self,theImage):
        outData=np.zeros((self.numDesc,128))
        # This norm. is just to prevent slightly larger than one values
        theImage/=np.max(theImage)
        ubImage=(theImage*255).astype('uint8')
        gsImage=cv2.cvtColor(ubImage,cv2.COLOR_RGB2GRAY)
        theSIFT=cv2.xfeatures2d.SIFT_create()
        keyPoints,theDescriptors=theSIFT.detectAndCompute(gsImage,None)
        
        nClust=min(self.numDesc,theDescriptors.shape[0])
        k=KMeans(n_clusters=nClust,random_state=0).fit(theDescriptors)
        idxSort=np.argsort(np.histogram(k.labels_,nClust)[0])[::-1]        
        theDescriptors=k.cluster_centers_[idxSort,:]
           
        dMin=theDescriptors.min(axis=0)
        dMax=theDescriptors.max(axis=0)
        theDescriptors=(theDescriptors-dMin)/(dMax-dMin)
        outData[:theDescriptors.shape[0],:]=theDescriptors
        outData=outData.ravel()
            
        return outData
        
    def on_epoch_end(self):
        np.random.seed(0)

###############################################################################
# Data generator that uses the existing loops in the database and uses
# SIFT descriptors.
# Operation : Same as DataGeneratorSIFTImages except that loops are not built
#             synthetically but they come from the dataset itself.
###############################################################################
class DataGeneratorSIFTLoops(Sequence):
    def __init__(self,dataSet,batchSize=20,imgSize=(240,320),numDesc=128):
        self.dataSet=dataSet
        self.batchSize=batchSize
        self.imgSize=imgSize
        self.numDesc=numDesc
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.dataSet.numLoops/float(self.batchSize)))

    def __getitem__(self,theIndex):
        X=[]
        y=[]
        bStart=max(theIndex*self.batchSize,0)
        bEnd=min((theIndex+1)*self.batchSize,self.dataSet.numLoops)
        imagePairs=[[self.dataSet.get_loop(i),i] for i in range(bStart,bEnd)]
        for curPair in imagePairs:
            theImage,theFeatures=self.__gettuple__(curPair)
            X.append(theImage)
            y.append(theFeatures)
        return np.array(X),np.array(y)
    
    def __gettuple__(self,theLoop):
        theImages=(resize(theLoop[0][0],self.imgSize),resize(theLoop[0][1],self.imgSize))
        idxSIFT=int(np.round(np.random.uniform()))
        idxImage=1-idxSIFT
        siftFeatures=self.__get_descriptors__(theImages[idxSIFT])
        return theImages[idxImage],siftFeatures
   
    def __get_descriptors__(self,theImage):
        outData=np.zeros((self.numDesc,128))
        # This norm. is just to prevent slightly larger than one values
        theImage/=np.max(theImage)
        ubImage=(theImage*255).astype('uint8')
        gsImage=cv2.cvtColor(ubImage,cv2.COLOR_RGB2GRAY)
        theSIFT=cv2.xfeatures2d.SIFT_create()
        keyPoints,theDescriptors=theSIFT.detectAndCompute(gsImage,None)
        
        nClust=min(self.numDesc,theDescriptors.shape[0])
        k=KMeans(n_clusters=nClust,random_state=0).fit(theDescriptors)
        idxSort=np.argsort(np.histogram(k.labels_,nClust)[0])[::-1]        
        theDescriptors=k.cluster_centers_[idxSort,:]
               
        dMin=theDescriptors.min(axis=0)
        dMax=theDescriptors.max(axis=0)
        theDescriptors=(theDescriptors-dMin)/(dMax-dMin)
        outData[:theDescriptors.shape[0],:]=theDescriptors
        outData=outData.ravel()
            
        return outData

    
    def on_epoch_end(self):
        np.random.seed(0)