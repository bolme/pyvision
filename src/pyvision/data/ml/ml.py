'''
Created on Jan 14, 2010

@author: bolme
'''

import pyvision as pv
import numpy as np
import csv
import os.path
from pyvision.vector.SVM import SVM
from pyvision.vector.LDA import trainLDA


IRIS_PATH = os.path.join(pv.__path__[0],'data','ml','iris.csv')

reader = csv.reader(open(IRIS_PATH, "rb"))

data = []
labels = []
reader.next()
for row in reader:
    data_point = map(float, row[1:5])
    label = row[5]
    
    data.append(data_point)
    labels.append(label)
    
iris_data = np.array(data)
iris_labels = np.array(labels)

iris_training = np.arange(0,150,2)
iris_testing = iris_training+1

def testSVM():
    svm = SVM()
    for i in iris_training:
        svm.addTraining(labels[i],iris_data[i,:])
        
    svm.train(verbose = 1)
    
    success = 0.0
    total = 0.0
    for i in iris_testing:
        c = svm.predict(iris_data[i,:])
        #print c, labels[i]
        if c == labels[i]:
            success += 1
        total += 1
    print "SVM Rate:",success/total
    
def testLDA():
    training = iris_data[iris_training,:]
    labels = iris_labels[iris_training]
    w,v = trainLDA(training,labels)
    print -3.5634683*w[:,0],-2.6365924*w[:,1],v/v.sum()
    
    
if __name__ == "__main__":
    testLDA()
    testSVM()