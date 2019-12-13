# -*- coding: utf-8 -*-
###############################################################################
# Name        : DataSet
# Description : Dataset management class focused on queries.
# Notes       : A dataset is composed of a set of database images, a set of
#               query images and a set of loops between images (meaning that
#               images belonging to a loop partially overlap). Each query image
#               closes at least one loop with one or more database images. 
#               The preferred way to use specify the dataset is by loading it
#               from a file. This can be done directly from the constructor:
#               >> dataSet=DataSet('DATASET1.TXT')
#               The dataset file format is a set of fields separated by the
#               char "#". The fields are:
#               * Relative path to the folder with the database images
#               * Relative path to the folder with the query images
#               * File names of the database images, separated by commas
#               * File names of the query images, separated by commas
#               * Loop specs as a set of indexes separated by commas.
#                 The meaning is as follows: databaseImage[index[2*i]] closes
#                 a loop with queryImage[index[2*i+1]], where databaseImage
#                 and queryImage are the images as they apper in the
#                 previous fields. It is important that ALL the existing loops
#                 are specified in this way
# Notes       : See example_dataset.py for an usage example.
# Author      : Antoni Burguera (antoni.burguera@uib.es)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

import os
from skimage.io import imread

class DataSet:
    
    # Constructor.
    # Input  : fileName - If provided, loads the dataset from file and neglects
    #                     the remaining parameters
    #          dbImagesPath - Path to the database images
    #          qImagesPath  - Path to the query images
    #          dbImagesFns  - List of dabatase images file names
    #          qImageFns    - List of query images file names
    #          theLoops     - List of loops. Each loop is a list with two ints.
    #                         First int indexes dbImageFns and second int
    #                         indexes qImageFns
    # Note   : It is advisable to write the dataset in an external file with
    #          a text editor (format specified in this file's header) and load
    #          it.
    def __init__(self,fileName='',dbImagesPath='',qImagesPath='',dbImageFns=[],qImageFns=[],theLoops=[]):
        if fileName!='':
            self.load(fileName)
        else:
            self.dbImagesPath=dbImagesPath
            self.qImagesPath=qImagesPath
            self.dbImageFns=dbImageFns
            self.qImageFns=qImageFns
            self.theLoops=theLoops
            self._update_counters_()

    # Private method. Updates the number of loops, database images and query
    # images after creating or loading a dataset.
    def _update_counters_(self):
        self.numDBImages=len(self.dbImageFns)
        self.numQImages=len(self.qImageFns)
        self.numImages=self.numDBImages+self.numQImages
        self.numLoops=len(self.theLoops)

    # Saves the dataset to a file
    # Input  : fileName - Name of the file where to save the dataset.
    def save(self,fileName):
        dbImagesFnsString=",".join(self.dbImageFns)
        qImagesFnsString=",".join(self.qImageFns)
        theLoopsString=",".join([str(j) for sub in self.theLoops for j in sub])
        allStrings=self.dbImagesPath+'#'+self.qImagesPath+'#'+dbImagesFnsString+'#'+qImagesFnsString+'#'+theLoopsString
        with open(fileName,'wt') as outFile:
            outFile.write(allStrings)

    # Loads a dataset from a file
    # Input  : fileName - Name of the file with the dataset to load
    def load(self,fileName):
        with open(fileName,'rt') as inFile:
            allStrings=inFile.read()
        dataFields=allStrings.split('#')
        self.dbImagesPath=dataFields[0]
        self.qImagesPath=dataFields[1]
        self.dbImageFns=dataFields[2].split(',')
        self.qImageFns=dataFields[3].split(',')
        self.theLoops=dataFields[4].split(',')
        self.theLoops=[[int(self.theLoops[i]),int(self.theLoops[i+1])] for i in range(0,len(self.theLoops),2)]
        self.fileName=fileName
        self._update_counters_()

    # Prints the dataset info
    def print(self):
        print('[ PRINTING DATASET ]')
        print('DATABASE PATH:')
        print('  - '+self.dbImagesPath)
        print('QUERY PATH:')
        print('  - '+self.qImagesPath)
        print('DATABASE FILES:')
        for curFile in self.dbImageFns:
            print('  - '+curFile)
        print('QUERY FILES:')
        for curFile in self.qImageFns:
            print('  - '+curFile)
        print('LOOPS:')
        for curLoop in self.theLoops:
            print('  - DB: '+self.dbImageFns[curLoop[0]]+' Q: '+self.qImageFns[curLoop[1]])
        print('[ DATASET PRINTED ]\n\n')

    # Compares the dataset with another one and prints the results of the
    # comparison.
    # Input  : ds2 - The dataset to which the current one is compared.
    def compare(self,ds2):
        dbPathEqual=(self.dbImagesPath==ds2.dbImagesPath)
        qPathEqual=(self.qImagesPath==ds2.qImagesPath)
        dbFnsEqual=(self.dbImageFns==ds2.dbImageFns)
        qFnsEqual=(self.qImageFns==ds2.qImageFns)
        loopsEqual=(self.theLoops==ds2.theLoops)

        print('[ COMPARISON REPORT ]')
        if dbPathEqual and qPathEqual and dbFnsEqual and qFnsEqual and loopsEqual:
            print('  - DATASETS ARE IDENTICAL.')
        else:
            if not dbPathEqual:
                print('  - DATABASE PATHS ARE DIFFERENT.')
            if not qPathEqual:
                print('  - QUERY PATHS ARE DIFFERENT.')
            if not dbFnsEqual:
                print('  - DATABASE FILES ARE NOT THE SAME.')
            if not qFnsEqual:
                print('  - QUERY FILES ARE NOT THE SAME.')
            if not loopsEqual:
                print('  - LOOPS ARE NOT THE SAME.')
        print('[ END OF COMPARISON REPORT ]')

    # Outputs one image of the dataset.
    # Input  : theIndex - Index of the image to retrieve. All images are acces-
    #                     sible. Lower indexes correspond to database images
    #                     and larger indexes correspond to query images. The
    #                     index is wrapped, so after the last image the first
    #                     one will be returned.
    # Output : image    - Image in skimage format (height,width,channels)
    #                     being channels in RGB order.
    def get_image(self,theIndex):
        theIndex=theIndex%self.numImages
        if theIndex<self.numDBImages:
            fileName=os.path.join(self.dbImagesPath,self.dbImageFns[theIndex])
        else:
            fileName=os.path.join(self.qImagesPath,self.qImageFns[theIndex-self.numDBImages])
        return imread(fileName)

    # Outputs the two images belonging to the specified loop. The first retur-
    # ned image is a database image and the second one is a query image.
    # Input  : theIndex - Index of the loop to retrieve. The index is wrapped,
    #                     so that after the last loop, the first one will be
    #                     returned.
    # Output : dbImage  - Database image in skimage format (height,width,chan)
    #                     being chan in RGB order.
    # Output : qImage   - Query image in skimage format (height,width,chan)
    #                     being chan in RGB order.
    def get_loop(self,theIndex):
        theIndex=theIndex%self.numLoops
        dbFileName=os.path.join(self.dbImagesPath,self.dbImageFns[self.theLoops[theIndex][0]])
        qFileName=os.path.join(self.qImagesPath,self.qImageFns[self.theLoops[theIndex][1]])
        return imread(dbFileName),imread(qFileName)

    # Outputs the specified database image.
    # Input  : theIndex - Index of the database image to retrieve. The index is
    #                     wrapped so that after the last database image,
    #                     the first one will be returned.
    # Output : dbImage  - Database image in skimage format (height,width,chan)
    #                     being chan in RGB order.
    def get_dbimage(self,theIndex):
        theIndex=theIndex%self.numDBImages
        dbFileName=os.path.join(self.dbImagesPath,self.dbImageFns[theIndex])
        return imread(dbFileName)

    # Outputs the specified query image.
    # Input  : theIndex - Index of the query image to retrieve. The index is
    #                     wrapped so that after the last query image,
    #                     the first one will be returned.
    # Output : dbImage  - Query image in skimage format (height,width,chan)
    #                     being chan in RGB order.   
    def get_qimage(self,theIndex):
        theIndex=theIndex%self.numDBImages
        qFileName=os.path.join(self.qImagesPath,self.qImageFns[theIndex])
        return imread(qFileName)

    # Outputs the indexes of the database images that close a loop with
    # the specified query image.
    # Input  : theIndex - Index of the query image. The index is wrapped, so
    #                     that after the last query image the first one is
    #                     considered.
    # Output : indexes  - List of database images indexes that close loop with
    #                     the specified query.
    def get_qloop(self,theIndex):
        theIndex=theIndex%self.numQImages
        dbIndexes=[]
        for curLoop in self.theLoops:
            if curLoop[1]==theIndex:
                dbIndexes.append(curLoop[0])
        return dbIndexes