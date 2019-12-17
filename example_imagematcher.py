# -*- coding: utf-8 -*-
###############################################################################
# Name        : example_imagematcher
# Description : A basic example on how to use the ImageMatcher class
# Notes       : Just run the script. Be sure that dataset.py is accessible,
#               the DATASET1.TXT file is accessible, and that the
#               paths specified within DATASET1.TXT are correct.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

from dataset import DataSet
from imagematcher import ImageMatcher

# Define some parameters
loopToUse=0
nonLoopQuery=0

# Load the dataset and get one loop and one non-loop
print('[[ LOADING AND PREPARING DATASET]]')
dataSet=DataSet('DATASETS/DATASET1.TXT')
loop0,loop1=dataSet.get_loop(loopToUse)
nonLoop0=dataSet.get_qimage(nonLoopQuery)
theLoops=dataSet.get_qloop(nonLoopQuery)
for i in range(dataSet.numDBImages):
    if not i in theLoops:
        break
nonLoop1=dataSet.get_dbimage(i)
print('[[ DATASETS LOADED AND IMAGES PREPARED ]]')

# Match loop images
print('[[ MATCHING LOOP IMAGES AND PLOTTING RESULTS ]]')
theMatcher=ImageMatcher()
theMatcher.define_images(loop0,loop1)
doMatch=theMatcher.estimate()
print('  * THE MATCHER STATES THAT IMAGES MATCH? '+str(doMatch))
theMatcher.plot_matches()
theMatcher.plot_consensus_set()
print('[[ LOOP IMAGES MATCHED AND PLOTTED ]]')

# Match non loop images
print('[[ MATCHING NON LOOP IMAGES AND PLOTTING RESULTS ]]')
theMatcher=ImageMatcher()
theMatcher.define_images(nonLoop0,nonLoop1)
doMatch=theMatcher.estimate()
print('  * THE MATCHER STATES THAT IMAGES MATCH? '+str(doMatch))
theMatcher.plot_matches()
print('[[ NON LOOP IMAGES MATCHED AND PLOTTED ]]')
