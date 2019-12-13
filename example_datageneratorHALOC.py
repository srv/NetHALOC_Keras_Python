# -*- coding: utf-8 -*-
###############################################################################
# Name        : example_datagenerator
# Description : A basic example on how the DataGenerator class works.
# Notes       : Just run the script. Be sure that DATASET1.TXT is within
#               the DATASETS folder and that the paths specified in that file
#               for database and query images are correct.
# Notes       : Please note that this is just an example to understand how
#               the data generators work. They should not be directly used,
#               but used to feed Keras fit_generator. 
# Author      : Antoni Burguera (antoni.burguera@uib.es) and Francisco Bonin Font (francisco.bonin@uib.es)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################
from dataset import DataSet
from datagenerator import DataGeneratorHALOCImages
import matplotlib.pyplot as plt
import numpy as np # math operations 

# Load one dataset
print('[[ LOADING DATASET1 ]]')
dataSet=DataSet('DATASETS/DATASET1.TXT')
print('[[DATASETS LOADED ]]\n\n')

# Let's create a data generator
print('[[ CREATING DATA GENERATOS ]]')
dataGenerator=DataGeneratorHALOCImages(dataSet,batchSize=4)
print('[[ DATA GENERATOR CREATED ]]\n\n')

# Let's print some info
print('[[ PRINTING DATA GENERATOR INFO ]]')
print('  * NUMBER OF BATCHES : '+str(dataGenerator.__len__()))
print('  * BATCH SIZE : '+str(dataGenerator.batchSize))
print('[[ BASIC INFO PRINTED ]]\n\n')

# Let's extract the first batch
print('[[ EXTRACTING FIRST BATCH ]]')
firstBatch=dataGenerator.__getitem__(0)
print('[[ FIRST BATCH EXTRACTED ]]\n\n')

# Let's plot and print the batch data
print('[[ PLOTTING THE BATCH IMAGES AND PRINTING DESCRIPTORS ]]')
plt.figure()
for theIndex in range(len(firstBatch[0])):
    plt.subplot(2,2,theIndex+1)
    plt.imshow(firstBatch[0][theIndex])
    print('  * THE DESCRIPTOR ASSOCIATED WITH THE IMAGE SHOWN IS: ')
    print(firstBatch[1][theIndex])
plt.show()
print('[[ PRINTING AND PLOTTING COMPLETE ]]')