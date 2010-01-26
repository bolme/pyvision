'''
Created on Jan 21, 2010

@author: nayeem
'''
import pyvision as pv
import numpy as np
import scipy as sp
from pyvision.vector.SVM import SVM
import csv
import os.path




class multiSVM:
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.data = self.readData()
        self.train_data = self.data[0]
        self.train_labels = self.data[1]
        self.test_data = self.data[2]
        #self.runSVM()
        
    def trainData(self):
        return self.train_data
    def trainLabels(self):
        return self.train_labels
    def testData(self):
        return self.test_data
        
        
    def runSVM(self):
        svm = SVM()
        print "I am in the SVM module now"   
    
    def readData(self):
        IRIS_PATH = os.path.join(pv.__path__[0],'data','ml','iris.csv')
        readcsv = csv.reader(open(IRIS_PATH,"rb"))


        data = []
        labels = []
        readcsv.next()
        train_data = []
        test_data = []
        train_labels = []
        pred_labels = []
        
        for row in readcsv:
            data_point = map(float, row[1:5])
            label = row[5]
            
            data.append(data_point)
            labels.append(label)


        iris_data = np.array(data)
        iris_labels = np.array(labels)
        
        data_length = len(iris_data)
        iris_training = np.arange(0, data_length, 2)
        iris_testing = iris_training + 1
        

        for i in iris_training:
            train_data.append(iris_data[i, :])
            train_labels.append(iris_labels[i])
    
        for i in iris_testing:
            test_data.append(iris_data[i, :])


        
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_labels = np.array(train_labels)
        
        data = train_data, train_labels, test_data
        return data

class Node:
        def __init__(self):
            self.classList = []
            self.classData = []
            self.pos = 0
            self.leftChild = Node() 
            self.rightChild = Node()
 
class Tree:
        root = Node()    
        
        def __init__(self):
            self.root = None
        
        def insert(self,classlist,classData, pos):
            newNode = Node()
            newNode.classList = classlist
            newNode.classData = classData
            newNode.pos = pos
            if self.root == None:
                self.root = self.newNode
            else:
                curr = self.root
                parent = Node()
                while True:
                    parent = curr
                    if newNode.pos == -1:
                        curr = curr.leftChild
                        if curr == None:
                            parent.leftChild = newNode()
                            return
                    else:
                        curr = curr.rightChild
                        if curr == None:
                            parent.rightChild = newNode()
                            return

                        
                        
                    
        
ms = multiSVM()
trainingdata = ms.trainData()
traininglabels = ms.trainLabels()
testdata = ms.testData()
                       

classes = np.unique(traininglabels)  # Unique classes
num_classes = len(classes)
num_features = np.size(trainingdata,axis=1)  # Columns of training Data
num_samples = np.size(trainingdata,axis=0)  # Number of samples

for i in np.arange(0,num_classes):
    print classes[i]
    mask = traininglabels==classes[i]
    numThisClass = sum(mask)
    print numThisClass
    trThisClass = trainingdata[mask,:]
    
#    centerThisClass = trThisClass.mean(axis=0)
#    print '**********************************************************************************'
#    covThisClass = np.cov(trThisClass)
#    print np.cov(trThisClass)
#    print '**********************************************************************************'
#    print np.shape(covThisClass)
#    invCovMatThisClass = np.linalg.inv(covThisClass)
#    print np.shape(invCovMatThisClass)
    assert(0)