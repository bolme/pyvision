'''
Created on May 21, 2010

@author: bolme
'''

import csv
import numpy as np
from scipy.io import savemat

def processSpectf():
    f = csv.reader(open("spect/SPECTF.train.txt",'rb'))
    
    labels = []
    data = []
    for line in f:
        labels.append(int(line[0]))
        data.append([float(each) for each in line[1:]])
    
    train_labels = np.array(labels)
    train_data = np.array(data)
    
    f = csv.reader(open("spect/SPECTF.test.txt",'rb'))
    
    labels = []
    data = []
    for line in f:
        labels.append(int(line[0]))
        data.append([float(each) for each in line[1:]])
    
    test_labels = np.array(labels)
    test_data = np.array(data)
    
    filename = "spectf.mat"
    
    savemat(filename,{'train_labels':train_labels,'train_data':train_data,'test_labels':test_labels,'test_data':test_data})

def processVoting():
    f = csv.reader(open("voting/house-votes-84.data.txt",'rb'))
    
    labels = []
    data = []
    for line in f:
        print line
        labels.append(int(line[0]))
        data.append([float(each) for each in line[1:]])
    
    train_labels = np.array(labels)
    train_data = np.array(data)
    
    f = csv.reader(open("spect/SPECTF.test.txt",'rb'))
    
    labels = []
    data = []
    for line in f:
        labels.append(int(line[0]))
        data.append([float(each) for each in line[1:]])
    
    test_labels = np.array(labels)
    test_data = np.array(data)
    
    filename = "spectf.mat"
    
    savemat(filename,{'train_labels':train_labels,'train_data':train_data,'test_labels':test_labels,'test_data':test_data})

processVoting()