# -*- coding: utf-8 -*-
###############################################################################
# Name        : example_candidates
# Description : A basic example on how to obtain loop candidates given an
#               image.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

from dataset import DataSet
from modelwrapper import ModelWrapper
from skimage.transform import resize
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

# Prepares an image to be useable by the Neural Network. This involves:
# * Resizing it to the correct resolution
# * Since the NN expects batches, convert the image into a batch of size
#   one. That is, convert from (height,width,numChannels) to 
#   (1,height,width,numChannels)
def prepare_image(theImage,imgSize):
    return resize(theImage,imgSize).reshape((1,imgSize[0],imgSize[1],3))

# Define some parameters
queryToUse=9

# Load a dataset
dataSet=DataSet('DATASETS/DATASET1.TXT')

# Load a trained model.
theModel=ModelWrapper()
theModel.load('TRAINED_MODELS/SIFTIMAGES_T2_V3_EPOCHS15')

# Get the image size (NN input shape) and the descriptor size (NN output shape)
imgSize=theModel.cnnModel.layers[0].input_shape[1:3]
descSize=theModel.cnnModel.layers[-1].output_shape[1]

# Select a query image
queryImage=prepare_image(dataSet.get_qimage(queryToUse),imgSize)

# Let's predict the descriptor for that query image using only the CNN layer.
queryDescriptor=theModel.predict(queryImage,useCNN=True)

# Let's predict the descriptors for all the database images. Please note that
# this implementation is far from optimal: it would be faster to group the
# images in batches and do the predictions together. Check predict_images
# in tester.py to see how to do it.
dbDescriptors=[]
for dbIndex in range(dataSet.numDBImages):
    dbImage=prepare_image(dataSet.get_dbimage(dbIndex),imgSize)
    dbDesc=theModel.predict(dbImage,useCNN=True)
    dbDescriptors.append(dbDesc[0])

dbDescriptors=np.array(dbDescriptors)

# Compute the distances between the query descriptor and all the database
# descriptors.    
theDistances=cdist(queryDescriptor,dbDescriptors,'euclidean')

# Select the 5 images that most likely close loop with the query.
loopCandidates=np.argsort(theDistances[0,:])[:5]

# Get the images that actually close loop with the used query.
actualLoops=dataSet.get_qloop(queryToUse)

# Plot the query and the 5 images. For each of these 5 images, state if it is
# an actual loop or not.
plt.close('all')
plt.figure()
plt.subplot(2,3,1)
# Plot the query
plt.imshow(dataSet.get_qimage(queryToUse))
plt.title('QUERY')
# Plot each of the selected database images
for i in range(5):
    plt.subplot(2,3,i+2)
    plt.imshow(dataSet.get_dbimage(loopCandidates[i]))
    # If the image is an actual loop (a true positive), show the message
    # "ACTUAL LOOP".
    if loopCandidates[i] in actualLoops:
        plt.title('ACTUAL LOOP')