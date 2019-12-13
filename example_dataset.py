# -*- coding: utf-8 -*-
###############################################################################
# Name        : example_dataset
# Description : A basic example on how to use the DataSet class
# Notes       : Just run the script. Be sure that DATASET1.TXT, DATASET2.TXT
#               and DATASET3.TXT are within the DATASETS folder and that
#               the paths specified in these files for database and query
#               images folders are correct.
# Author      : Antoni Burguera (antoni.burguera@uib.es)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

from dataset import DataSet
import matplotlib.pyplot as plt
import sys

# Load three datasets
print('[[ LOADING DATASETS ]]')
dataSet1=DataSet('DATASETS/DATASET1.TXT')
dataSet2=DataSet('DATASETS/DATASET2.TXT')
dataSet3=DataSet('DATASETS/DATASET3.TXT')
print('[[DATASETS LOADED ]]\n\n')

# Let's print the dataSet1 info
print('[[ PRINTINT DATASET1 INFO ]]')
dataSet1.print()
print('[[ DATASET1 PRINTED ]]\n\n')

# Let's compare dataSet1 to itself
print('[[ COMPARING DATASET1 TO ITSELF ]]')
dataSet1.compare(dataSet1)
print('[[ COMPARED ]]\n\n')

# Let's compare dataSet1 to dataSet3
print('[[ COMPARING DATASET1 TO DATASET2 ]]')
dataSet1.compare(dataSet2)
print('[[ COMPARED ]]\n\n')

print('[[ PLOTTING ONE IMAGE WITH 3 LOOPS ]]')
# Let's search a query in dataset3 with 3 or more loop closures
for queryIndex in range(dataSet3.numQImages):
    theLoops=dataSet3.get_qloop(queryIndex)
    numLoops=len(theLoops)
    if numLoops>=3:
        break

# If loop not found, just exit
if numLoops<3:
    sys.exit('[[ ERROR: UNABLE TO FOUND 3 LOOP CLOSURES IN DATASET 3]]')
    
# Otherwise, plot the query and three loops
plt.figure()
plt.subplot(2,2,1)
plt.imshow(dataSet3.get_qimage(queryIndex))
    
for i in range(3):
    plt.subplot(2,2,i+2)
    plt.imshow(dataSet3.get_dbimage(theLoops[i]))
    
plt.show()
print('[[ PLOT DONE ]]')