# -*- coding: utf-8 -*-
###############################################################################
# Name        : ModelWrapper
# Description : Simple wrapper to ease the acces to one specific Keras model.
# Notes       : See example_modelwrapper.py for an unsage example.
# Author      : Antoni Burguera (antoni.burguera@uib.es)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

from keras import layers
from keras import models
from pickle import dump,load
from os.path import splitext
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.models import Model

class ModelWrapper:
    # Constructor. Please note that it does NOT create the model
    # Input  : inputShape - The size of input images.
    #          outputSize - Size of the model output (dense part of network)
    def __init__(self,inputShape=(240,320,3),outputSize=128*3): # CUIDADO 128*3 porque es la longitud del HALOC
        self.theModel=None
        self.cnnModel=None
        self.trainHistory=None
        self.inputShape=inputShape
        self.outputSize=outputSize

    # Creates the model (theModel) and a second model (cnnModel) neglecting the
    # dense layers.        
    def create(self):
        self.theModel=models.Sequential()
        self.theModel.add(layers.Conv2D(128,(3,3),strides=(2,2),activation='sigmoid',input_shape=self.inputShape))
        self.theModel.add(layers.MaxPool2D((3,3),strides=(2,2)))
        self.theModel.add(layers.Conv2D(64,(3,3),strides=(1,1),activation='sigmoid'))
        self.theModel.add(layers.MaxPool2D((3,3),strides=(2,2)))
        self.theModel.add(layers.Conv2D(4,(3,3),strides=(1,1),activation='sigmoid'))
        self.theModel.add(layers.Flatten())
        self.theModel.add(layers.Dense(512, activation='sigmoid'))
        self.theModel.add(layers.Dense(1024, activation='sigmoid'))
      #  self.theModel.add(layers.Dense(1024, activation='sigmoid')) # fbf 3/07/2019, since the last size for HALOC must be 384 (3*128), lets try with the dencent of the layers from 1024 to 384
      #  self.theModel.add(layers.Dense(512, activation='sigmoid'))
        self.theModel.add(layers.Dense(self.outputSize, activation='sigmoid'))
        self.theModel.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        self.__define_cnnmodel__()
      
    # Private method to define the aforementioned cnnModel
    def __define_cnnmodel__(self):
        self.cnnModel=Model(inputs=self.theModel.input,outputs=self.theModel.layers[5].output)

    # Just a helper to build filenames
    def __build_filenames__(self,fileName):
        baseName,theExtension=splitext(fileName)
        modelFName=baseName+'.h5'
        histFName=baseName+'_HISTORY.pkl'
        return modelFName,histFName
        
    # Saves the model (as a .h5 file) and the training history (by means of
    # pickle)
    def save(self,fileName):
        modelFName,histFName=self.__build_filenames__(fileName)
        self.theModel.save(modelFName)
        with open(histFName,'wb') as histFile:
            dump(self.trainHistory,histFile)
        
    # Loads the model and the training history
    def load(self,fileName):
        modelFName,histFName=self.__build_filenames__(fileName)
        self.theModel=load_model(modelFName)        
        with open(histFName,'rb') as histFile:
            self.trainHistory=load(histFile)
        self.inputShape=self.theModel.layers[0].input_shape[1:]
        self.outputSize=self.theModel.layers[-1].output_shape
        self.__define_cnnmodel__()
        
    # Plots the training history
    def plot_training_history(self,plotTitle='TRAINING EVOLUTION'):
        plt.plot(self.trainHistory['loss'])
        plt.plot(self.trainHistory['val_loss'])
        plt.title(plotTitle)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','validation'], loc='upper left')
        plt.show()

    # Trains the model. Only useable with data generators. Please use those
    # defined in datagenerators.py. Also note that the outputSize has to
    # coincide with the one provided by the data generators.
    def train(self,trainGenerator,valGenerator,nEpochs=100):
        self.trainHistory=self.theModel.fit_generator(trainGenerator,epochs=nEpochs,validation_data=valGenerator).history
        
    # Evaluate the model
    def evaluate(self,testGenerator):
        return self.theModel.evaluate_generator(testGenerator)
        
    # Output the model predictions. Use the parameter useCNN to select the
    # output of the dense layers (useCNN=False) or the output of the convolu-
    # tional layers (useCNN=True).
    def predict(self,theImages,useCNN=True):
        if useCNN:
            return self.cnnModel.predict(theImages)
        return self.theModel.predict(theImages)
