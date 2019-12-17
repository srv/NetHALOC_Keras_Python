# -*- coding: utf-8 -*-
###############################################################################
# Name        : main_trainer
# Description : Performs training, validation and testing using the desired
#               data generator. All combinations of three datasets are used.
# Notes       : To properly use it, proceed as follows:
#               * Consistently define outputSize,theDataGenerator and
#                 savePrefixName
#               * Use a sufficiently large number of epochs (fourth value
#                 in each of the theIndexes items). For example, 100.
#               * Cross your fingers and run the script.
#               * Depending on the number of epochs and the generator, go to:
#                 - Take a cup of coffee.
#                 - Go out and have a beer.
#                 - Go to sleep and check results tomorrow.
#                 - Go on holyday for a couple of days.
#               * Once finished, load the models one by one and plot the
#                 training histories (plot_training_history in ModelWrapper).
#               * By observing the plots, decide the optimal number of epochs.
#               * Change the number of epochs in theIndexes for each case
#                 to the optimal number of epochs observed in previous step.
#               * Change the name of the file where global results are
#                 stored (the one ending with AUC.pkl) to another one. For
#                 example, make it end with AUC2.pkl
#               * Execute the script again, and wait (coffee, sleep, holydays)
#               * When finished, load both the first global results file
#                 (the one ending with AUC.pkl) and the second one (AUC2.pkl)
#               * Compare the values in both files. If values in xxxAUC2.pkl
#                 are all similar or larger than those in xxxAUC.pkl, that's
#                 it: you have good trained models. If some values in xxxAUC2
#                 are significantly smaller than those in xxxAUC... increase
#                 the number of epochs in these cases, and repeat the process
#                 only for those cases (so, just rewrite theIndexes to repre-
#                 sent only the cases you want to modify).
#               * Repeat the process (test, increase epochs...) until
#                 all values in xxxAUC2 are similar or larger than those
#                 in xxxAUC.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # only if ROS is installed
cd /home/xesc/RECERCA/SUMMUM/CNN/python/NNLOOP # cd where you have your DATASETS folder

from dataset import DataSet
from datagenerator import DataGeneratorHALOCImages
from modelwrapper import ModelWrapper
from tester import Tester

print('[[ LOADING DATASETS ]]')
dataSet1=DataSet('DATASETS/DATASET1.TXT')
dataSet2=DataSet('DATASETS/DATASET2.TXT')
print('[[DATASETS LOADED ]]\n\n')

# Create three data generators
print('[[ CREATING DATA GENERATORS ]]')
dataGenerator1=DataGeneratorHALOCImages(dataSet1,batchSize=4)
dataGenerator2=DataGeneratorHALOCImages(dataSet2,batchSize=4)
print('[[ GENERATORS CREATED ]]\n\n')

# Create the model, 384 is the lenght of the hash, this model corresponds to the swapt dense layer 
# configuration of the model wrapper: 1024,512,384
print('[[ CREATING THE MODEL ]]')
theModel=ModelWrapper(outputSize=384) 
theModel.create()
print('[[ MODEL CREATED ]]')

# Train the model with dataset2 and validate with dataset1
print('[[ TRAINING WITH DATASET2 AND VALIDATING WITH DATASET1 ]]')
theModel.train(trainGenerator=dataGenerator2,valGenerator=dataGenerator1,nEpochs=10)
print('[[ MODEL TRAINED ]]')

# Save the model
print('[[ SAVING THE MODEL ]]')
theModel.save('TEST_MODEL_trainDS2_valDS1_swapt_dense_layers')
print('[[ MODEL SAVED ]]')

# Loading the model (not necessary, since it is already loaded. Loading is
# performed just for the sake of completeness)
print('[[ LOADING THE MODEL ]]')
theModel.load('TEST_MODEL_trainDS2_valDS1_swapt_dense_layers')
print('[[ MODEL SAVED ]]')

# Plot the training history
print('[[ PLOTTING TRAINING HISTORY ]]')
theModel.plot_training_history()
print('[[ PLOT DONE ]]')

# Load the test dataset
print('[[ LOADING DATASET 1 ]]')
dataSet=DataSet('DATASETS/DATASET1.TXT')
print('[[ DATASET LOADED ]]')

# Load the model, this model corresponds to the original dense layer configuration of the model wrapper: 512, 1024,384
print('[[ LOADING THE MODEL ]]')
theModel=ModelWrapper()
theModel.load('TRAINED_MODELS/HALOC/loops/Loops_TEST_MODEL_trainDS3_valDS2')
print('[[ MODEL LOADED ]]')


# Creating the tester
from tester import Tester

print('[[ CREATING THE TESTER ]]')
theTester=Tester(theModel,dataSet)
print('[[ TESTER CREATED ]]')

# Computing and plotting hit ratio evolution
print('[[ COMPUTING AND PLOTTING HIT RATIO EVOLUTION ]]')
theHR,thaAUC=theTester.compute_hitratio_evolution()
theTester.plot_hitratio_evolution()
print('[[ HIT RATIO EVOLUTION COMPUTED AND PLOTTED ]]')

# Computing and printing full stats
print('[[ COMPUTING AND PRINTING FULL STATS ]]')
tp,fp,tn,fn,tdist,tloops=theTester.compute_fullstats()
print('[[ FULL STATS COMPUTED AND PRINTED ]]')
