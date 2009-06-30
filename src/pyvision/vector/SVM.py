# PyVision License
#
# Copyright (c) 2006-2008 David S. Bolme
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# 
# 3. Neither name of copyright holders nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
# 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import tempfile
import os
import random
import unittest
import csv 
import pickle

import numpy as np

from svm import *

import pyvision
from pyvision.vector.VectorClassifier import *
from pyvision.analysis.Table import Table

# SVM Types
TYPE_C_SVC       = C_SVC
TYPE_NU_SVC      = NU_SVC
TYPE_EPSILON_SVR = EPSILON_SVR
TYPE_NU_SVR      = NU_SVR

TYPE_SVC=TYPE_C_SVC
TYPE_SVR=TYPE_NU_SVR

# SVM Kernels
KERNEL_LINEAR='LINEAR'
KERNEL_RBF='RBF'


class SVM(VectorClassifier):
    def __init__(self, type=TYPE_SVC, kernel=KERNEL_RBF, svr_epsilon=0.1, nu = 0.5, random_seed=None, validation_size=0.33,**kwargs):
        '''
        Create an svm.
        
        Make sure you choose "classifacition" or "regression".  Other parameters control
        features of the SVM.
        
        also passes keyword args to VectorClassifier
        '''
        #TODO: Document constructor
        #TODO: Add an option to specify SVM parameters directly and disable automatic tuning.
        self.svm = None
        self.svm_type = type
        self.kernel = kernel
        self.epsilon=svr_epsilon
        self.nu = nu
        self.random_seed = random_seed
        self.validation_size = validation_size
        
        if type in (TYPE_C_SVC,TYPE_NU_SVC): 
            VectorClassifier.__init__(self,TYPE_MULTICLASS,**kwargs)
        else:
            VectorClassifier.__init__(self,TYPE_REGRESSION,**kwargs)
            
    def __getstate__(self):
        '''This function is neccessary for pickling'''
        # Translate everything but the svm because that cannot be simply pickled.
        state = {}
        for key,value in self.__dict__.iteritems():
            if key == 'svm':
                filename = tempfile.mktemp()
                self.svm.save(filename)
                buffer = open(filename).read()
                os.remove(filename)
                state[key] = buffer
                continue
            
            state[key] = value
            
        return state
            
    def __setstate__(self,state):
        '''This function is neccessary for pickling'''
        # Translate everything but the svm because that cannot be simply pickled.
        for key,value in state.iteritems():
            if key == 'svm':
                filename = tempfile.mktemp()
                open(filename,'w').write(value)
                self.svm = svm_model(filename)
                os.remove(filename)
                continue

            self.__dict__[key] = value
            
    
    def trainClassifer(self,labels,vectors,ilog=None,verbose=False,callback=None, C_range = 2.0**np.arange(-5,16,1), G_range = 2.0**np.arange(-15,4,1)):
        '''
        Do not call this function instead call train.
        '''
        self.training_size = len(labels)
        
        if verbose: print
        if verbose: print "Training the SVM"
        # Convert the vectors to lists of floats
        new_vectors = []
        for vec in vectors:
            new_vec = []
            for value in vec:
                new_vec.append(float(value))
            new_vectors.append(new_vec)
        vectors = new_vectors
        
        #TODO: Select the training and validation sets here instead of in each function.  
        
        if self.svm_type in (TYPE_C_SVC,TYPE_NU_SVC) and self.kernel == KERNEL_RBF:
            if verbose: print "TRAINING SVC RBF"
            self.train_SVC_RBF(labels,vectors,verbose,C_range,G_range,callback=callback)
        elif self.svm_type in (TYPE_C_SVC,TYPE_NU_SVC) and self.kernel == KERNEL_LINEAR:
            if verbose: print "TRAINING SVC Linear"
            self.train_SVC_Linear(labels,vectors,verbose,C_range,callback=callback)
        elif self.svm_type in (TYPE_NU_SVR,TYPE_EPSILON_SVR) and self.kernel == KERNEL_RBF:
            if verbose: print "TRAINING SVC RBF"
            self.train_SVR_RBF(labels,vectors,verbose,C_range,G_range,callback=callback)
        elif self.svm_type in (TYPE_NU_SVR,TYPE_EPSILON_SVR) and self.kernel == KERNEL_LINEAR:
            if verbose: print "TRAINING SVC Linear"
            self.train_SVR_Linear(labels,vectors,verbose,C_range,callback=callback)
        else:
            raise NotImplementedError("Unknown SVM type or kernel")
                    
        
    def train_SVC_RBF(self,labels,vectors, verbose, C_range, G_range, callback=None):
        '''Private use only'''
        # combine the labels and vectors into one set.
        data = []
        for i in range(len(labels)):
            data.append([labels[i],vectors[i]])
            
        #shuffle the data
        rng = random.Random()
        if self.random_seed != None:
            rng.seed(self.random_seed)
        rng.shuffle(data)
                
        # partition into validation and training
        if type(self.validation_size) == float and self.validation_size > 0.0 and self.validation_size < 1.0:
            training_cutoff = int(len(data)*(1.0-self.validation_size))
        elif type(self.validation_size) == int and self.validation_size < len(labels):
            training_cutoff = len(labels)-self.validation_size
        else:
            raise NotImplementedError("Cannot determine validation set from %s"%self.validation_size)
            
        if verbose: print "Training Cutoff:",len(labels),training_cutoff
        training_data = data[:training_cutoff]
        validation_data = data[training_cutoff:]
        
        tmp_labels = []
        tmp_vectors = []
        for each in training_data:
            tmp_labels.append(each[0])
            tmp_vectors.append(each[1])
        
        prob = svm_problem(tmp_labels,tmp_vectors)
        
        training_info = []
        training_svm = []
        training_table = Table()
        self.training_table = training_table
        i=0
        for C in C_range:
            for G in G_range:
                
                param = svm_parameter(svm_type=self.svm_type,kernel_type = RBF, C = C, gamma=G,p=self.epsilon,nu=self.nu)
                
                test_svm = svm_model(prob, param)
                
                successes = 0.0
                total = len(validation_data)
                for label,vector in validation_data:
                    pred = test_svm.predict(vector)
                    if pred == label:
                        successes += 1
                rate  = successes/total
 
                if verbose: print C,G,rate
                training_svm.append(test_svm)
                training_info.append([C,G,rate])
                
                training_table.setElement(i,'C',C)
                training_table.setElement(i,'G',G)
                training_table.setElement(i,'rate',rate)
                i+=1
                
            if callback != None:
                callback(int(100*float(i)/(len(C_range)*len(G_range))))
                
        if verbose: print 
        if verbose: print "------------------------------"
        if verbose: print " Tuning Information:"
        if verbose: print "         C      gamma    rate"
        if verbose: print "------------------------------"
        best = training_info[0]
        best_svm = training_svm[0]
        for i in range(len(training_info)):
            each = training_info[i]
            if verbose: print " %8.3e  %8.3e  %0.8f"%(each[0],each[1],each[-1])
            if best[-1] < each[-1]:
                best = each
                best_svm = training_svm[i]
        if verbose: print "------------------------------"
        if verbose: print 
        if verbose: print "------------------------------"
        if verbose: print " Best Tuning:"
        if verbose: print "         C      gamma    rate"
        if verbose: print "------------------------------"
        if verbose: print " %8.3e  %8.3e  %0.8f"%(best[0],best[1],best[-1])
        if verbose: print "------------------------------"
        if verbose: print
        self.training_info = training_info
        self.C     = best[0]
        self.gamma = best[1]
        self.tuning_rate = best[2]

        self.svm = best_svm
        
        
    def train_SVR_RBF(self,labels,vectors,verbose, C_range, G_range, callback=None):
        '''Private use only'''
        # combine the labels and vectors into one set.
        data = []
        for i in range(len(labels)):
            data.append([labels[i],vectors[i]])
            
        #shuffle the data
        rng = random.Random()
        if self.random_seed != None:
            rng.seed(self.random_seed)
        rng.shuffle(data)
                
        # partition into validation and training
        if type(self.validation_size) == float and self.validation_size > 0.0 and self.validation_size < 1.0:
            training_cutoff = int(len(data)*(1.0-self.validation_size))
        elif type(self.validation_size) == int and self.validation_size < len(labels):
            training_cutoff = len(labels)-self.validation_size
        else:
            raise NotImplementedError("Cannot determine validation set from %s"%self.validation_size)
            
        if verbose: print "Training Cutoff:",len(labels),training_cutoff
        training_data = data[:training_cutoff]
        validation_data = data[training_cutoff:]
        
        tmp_labels = []
        tmp_vectors = []
        for each in training_data:
            tmp_labels.append(each[0])
            tmp_vectors.append(each[1])
        
        prob = svm_problem(tmp_labels,tmp_vectors)
        
        training_info = []
        training_svm = []
        training_table=Table()
        self.training_table = training_table
        i = 0
        for C in C_range:
            for G in G_range:
                if verbose: print "Testing: %10.5f %10.5f"%(C,G),
                
                param = svm_parameter(svm_type=self.svm_type,kernel_type = RBF, C = C, gamma=G,p=self.epsilon,nu=self.nu)
                
                test_svm = svm_model(prob, param)
                
                mse = 0.0
                total = len(validation_data)
                for label,vector in validation_data:
                    pred = test_svm.predict(vector)
                    error = label - pred
                    mse += error*error
                mse = mse/total
 
                if verbose: print "%15.8e"%mse
                training_svm.append(test_svm)
                training_info.append([C,G,mse])
                training_table.setElement(i,'C',C)
                training_table.setElement(i,'G',G)
                training_table.setElement(i,'mse',mse)
                i+=1

            if callback != None:
                callback(int(100*float(i)/(len(C_range)*len(G_range))))
                
        if verbose: print 
        if verbose: print "------------------------------"
        if verbose: print " Tuning Information:"
        if verbose: print "         C      gamma   error"
        if verbose: print "------------------------------"
        best = training_info[0]
        best_svm = training_svm[0]
        for i in range(len(training_info)):
            each = training_info[i]
            if verbose: print " %8.3e  %8.3e  %0.8f"%(each[0],each[1],each[-1])
            if best[-1] > each[-1]:
                best = each
                best_svm = training_svm[i]
        if verbose: print "------------------------------"
        if verbose: print 
        if verbose: print "------------------------------"
        if verbose: print " Best Tuning:"
        if verbose: print "         C      gamma   error"
        if verbose: print "------------------------------"
        if verbose: print " %8.3e  %8.3e  %0.8f"%(best[0],best[1],best[-1])
        if verbose: print "------------------------------"
        if verbose: print
        self.training_info = training_info
        self.C     = best[0]
        self.gamma = best[1]
        self.error = best[2]

        self.svm = best_svm
        

    def train_SVC_Linear(self,labels,vectors,verbose, C_range, callback=None):
        '''Private use only.'''
        # combine the labels and vectors into one set.
        data = []
        for i in range(len(labels)):
            data.append([labels[i],vectors[i]])
            
        #shuffle the data
        rng = random.Random()
        if self.random_seed != None:
            rng.seed(self.random_seed)
        rng.shuffle(data)
                
        # partition into validation and training
        if type(self.validation_size) == float and self.validation_size > 0.0 and self.validation_size < 1.0:
            training_cutoff = int(len(data)*(1.0-self.validation_size))
        elif type(self.validation_size) == int and self.validation_size < len(labels):
            training_cutoff = len(labels)-self.validation_size
        else:
            raise NotImplementedError("Cannot determine validation set from %s"%self.validation_size)
            
        if verbose: print "Training Cutoff:",len(labels),training_cutoff
        training_data = data[:training_cutoff]
        validation_data = data[training_cutoff:]
        
        tmp_labels = []
        tmp_vectors = []
        for each in training_data:
            tmp_labels.append(each[0])
            tmp_vectors.append(each[1])
        
        prob = svm_problem(tmp_labels,tmp_vectors)
        
        training_info = []
        training_svm = []
        training_table = Table()
        self.training_table = training_table
        i=0
        for C in C_range:
                
            param = svm_parameter(svm_type=self.svm_type,kernel_type = LINEAR, C = C, p=self.epsilon,nu=self.nu)
                
            test_svm = svm_model(prob, param)
                
            successes = 0.0
            total = len(validation_data)
            for label,vector in validation_data:
                pred = test_svm.predict(vector)
                if pred == label:
                    successes += 1.0
            rate = successes/total
 
            training_svm.append(test_svm)
            training_info.append([C,rate])
            training_table.setElement(i,'C',C)
            training_table.setElement(i,'rate',rate)
            i+=1

            if callback != None:
                callback(int(100*float(i)/len(C_range)))
                
        if verbose: print 
        if verbose: print "------------------------------"
        if verbose: print " Tuning Information:"
        if verbose: print "         C   error"
        if verbose: print "------------------------------"
        best = training_info[0]
        best_svm = training_svm[0]
        for i in range(len(training_info)):
            each = training_info[i]
            if verbose: print " %8.3e  %0.8f"%(each[0],each[1])
            if best[-1] < each[-1]:
                best = each
                best_svm = training_svm[i]
        if verbose: print "------------------------------"
        if verbose: print 
        if verbose: print "------------------------------"
        if verbose: print " Best Tuning:"
        if verbose: print "         C   error"
        if verbose: print "------------------------------"
        if verbose: print " %8.3e  %0.8f"%(best[0],best[1])
        if verbose: print "------------------------------"
        if verbose: print
        self.training_info = training_info
        self.C     = best[0]
        self.tuned_rate = best[1]

        self.svm = best_svm
        
        
    def train_SVR_Linear(self,labels,vectors,verbose, C_range, callback=None):
        '''Private use only'''
        # combine the labels and vectors into one set.
        data = []
        for i in range(len(labels)):
            data.append([labels[i],vectors[i]])
            
        #shuffle the data
        rng = random.Random()
        if self.random_seed != None:
            rng.seed(self.random_seed)
        rng.shuffle(data)
                
        # partition into validation and training
        if type(self.validation_size) == float and self.validation_size > 0.0 and self.validation_size < 1.0:
            training_cutoff = int(len(data)*(1.0-self.validation_size))
        elif type(self.validation_size) == int and self.validation_size < len(labels):
            training_cutoff = len(labels)-self.validation_size
        else:
            raise NotImplementedError("Cannot determine validation set from %s"%self.validation_size)
            
        if verbose: print "Training Cutoff:",len(labels),training_cutoff
        training_data = data[:training_cutoff]
        validation_data = data[training_cutoff:]
        
        tmp_labels = []
        tmp_vectors = []
        for each in training_data:
            tmp_labels.append(each[0])
            tmp_vectors.append(each[1])
        
        prob = svm_problem(tmp_labels,tmp_vectors)
        
        training_info = []
        training_svm = []
        training_table = Table()
        self.training_table = training_table
        i=0
        for C in C_range:
                
            param = svm_parameter(svm_type=self.svm_type,kernel_type = LINEAR, C = C, p=self.epsilon,nu=self.nu)
                
            test_svm = svm_model(prob, param)
                
            mse = 0.0
            total = len(validation_data)
            for label,vector in validation_data:
                pred = test_svm.predict(vector)
                error = label - pred
                mse += error*error
            mse = mse/total
 
            training_svm.append(test_svm)
            training_info.append([C,mse])
            training_table.setElement(i,'C',C)
            training_table.setElement(i,'mse',mse)
            i+=1

            if callback != None:
                callback(int(100*float(i)/len(C_range)))
                
        if verbose: print 
        if verbose: print "------------------------------"
        if verbose: print " Tuning Information:"
        if verbose: print "         C   error"
        if verbose: print "------------------------------"
        best = training_info[0]
        best_svm = training_svm[0]
        for i in range(len(training_info)):
            each = training_info[i]
            if verbose: print " %8.3e  %0.8f"%(each[0],each[1])
            if best[-1] > each[-1]:
                best = each
                best_svm = training_svm[i]
        if verbose: print "------------------------------"
        if verbose: print 
        if verbose: print "------------------------------"
        if verbose: print " Best Tuning:"
        if verbose: print "         C   error"
        if verbose: print "------------------------------"
        if verbose: print " %8.3e  %0.8f"%(best[0],best[1])
        if verbose: print "------------------------------"
        if verbose: print
        self.training_info = training_info
        self.C     = best[0]
        self.error = best[1]

        self.svm = best_svm

        
    def predictValue(self,data,ilog=None):
        '''
        Please call predict instead.
        '''
        assert self.svm != None
        new_vec = []
        for value in data:
            new_vec.append(float(value))
        return self.svm.predict(new_vec)
    
    def predictSVMProbability(self,data):
        assert self.svm != None
        data = self.normalizeVector(data)
        new_vec = []
        for value in data:
            new_vec.append(float(value))
        prd, prb = self.svm.predict_probability(new_vec)
        #if self.type == TYPE_TWOCLASS or self.type == TYPE_MULTICLASS:
        #    return self.invertClass(value)
        #if self.type == TYPE_REGRESSION:
        #    return self.invertReg(value)
        return prd,prb

    def predictSVMValues(self,data):
        assert self.svm != None
        data = self.normalizeVector(data)
        new_vec = []
        for value in data:
            new_vec.append(float(value))
        d = self.svm.predict_values(new_vec)
        
        result = {}
        for key,value in d.iteritems():
            new_key = tuple()
            for each in key:
                if self.type == TYPE_TWOCLASS or self.type == TYPE_MULTICLASS:
                    each =  self.invertClass(each)
                #if self.type == TYPE_REGRESSION:
                #    return self.invertReg(value)
                new_key = new_key + (each,)
            result[new_key] = value
                
        #if self.type == TYPE_TWOCLASS or self.type == TYPE_MULTICLASS:
        #    return self.invertClass(value)
        #if self.type == TYPE_REGRESSION:
        #    return self.invertReg(value)
        return result


class _TestSVM(unittest.TestCase):
    ''' Unit tests for SVM '''
    
    def setUp(self):
        pass
    
        
    def test_sv_xor_rbf(self):
        # a simple binary two class
        xor = SVM(random_seed=0)
        for i in range(20):
            xor.addTraining(0,[0,0])
            xor.addTraining(0,[1,1])
            xor.addTraining(1,[0,1])
            xor.addTraining(1,[1,0])

        xor.train()
        #if verbose: print "XOR"
        self.assertEqual(xor.predict([0,0]),0)
        self.assertEqual(xor.predict([1,1]),0)
        self.assertEqual(xor.predict([1,0]),1)
        self.assertEqual(xor.predict([0,1]),1)

    def test_sv_pickle(self):
        # a simple binary two class
        xor = SVM(random_seed=0)
        for i in range(20):
            xor.addTraining(0,[0,0])
            xor.addTraining(0,[1,1])
            xor.addTraining(1,[0,1])
            xor.addTraining(1,[1,0])

        xor.train()

        tmp = pickle.dumps(xor)
        xor = pickle.loads(tmp)
        
        self.assertEqual(xor.predict([0,0]),0)
        self.assertEqual(xor.predict([1,1]),0)
        self.assertEqual(xor.predict([1,0]),1)
        self.assertEqual(xor.predict([0,1]),1)

    def test_sv_xor_linear(self):
        # a simple binary two class
        xor = SVM(kernel=KERNEL_LINEAR,random_seed=1)
        for i in range(20):
            xor.addTraining(0,[0,0])
            xor.addTraining(0,[1,1])
            xor.addTraining(1,[0,1])
            xor.addTraining(1,[1,0])

        xor.train()
        #print "XOR"
        
        # A linear model should not be able to perficly fit this data
        self.assertEqual(xor.predict([0,0]),0) 
        self.assertEqual(xor.predict([1,1]),0)
        self.assertEqual(xor.predict([1,0]),1) 
        self.assertEqual(xor.predict([0,1]),0) # ERROR 

    def test_sv_regression_rbf(self):
        rega = SVM(type=TYPE_EPSILON_SVR,kernel=KERNEL_RBF,random_seed=0)
        filename = os.path.join(pyvision.__path__[0],'data','synthetic','regression.dat')
        reg_file = open(filename,'r')
        labels = []
        vectors = []
        for line in reg_file:
            datapoint = line.split()
            labels.append(float(datapoint[0]))
            vectors.append([float(datapoint[3]),float(datapoint[4]),float(datapoint[5])])
            
        for i in range(50):
            rega.addTraining(labels[i],vectors[i])
        rega.train()
        
        mse = 0.0
        total = 0
        for i in range(50,len(labels)):
            p = rega.predict(vectors[i])
            e = p - labels[i]
            mse += e*e
        mse = mse/(len(labels)-50)
        #print "Regression Error:",mse
        
        self.assertAlmostEqual(mse,0.47066712325873877,places=4)
        
    def test_sv_regression_linear(self):
        # synthetic linear regression
        rega = SVM(type=TYPE_EPSILON_SVR,kernel=KERNEL_LINEAR,random_seed=0)
        filename = os.path.join(pyvision.__path__[0],'data','synthetic','regression.dat')
        reg_file = open(filename,'r')
        labels = []
        vectors = []
        for line in reg_file:
            datapoint = line.split()
            labels.append(float(datapoint[0]))
            vectors.append([float(datapoint[3]),float(datapoint[4]),float(datapoint[5])])

        for i in range(50):
            rega.addTraining(labels[i],vectors[i])
        rega.train()

        mse = 0.0
        total = 0
        for i in range(50,len(labels)):
            p = rega.predict(vectors[i])
            e = p - labels[i]
            #print labels[i],p,e
            mse += e*e
        mse = mse/(len(labels)-50)
        self.assertAlmostEqual(mse,0.52674701087510767,places=4)
        

    def test_gender(self):
        
        # image classification
        gender = SVM(type=TYPE_SVC,random_seed=0)
        filename = os.path.join(pyvision.__path__[0],'data','csuScrapShots','gender.txt')
        f = open(filename,'r')
        labels = []
        vectors = []
        for line in f:
            im_name, class_name = line.split()
            im_name = os.path.join(pyvision.__path__[0],'data','csuScrapShots',im_name)
            im = Image(im_name)
            im = Image(im.asPIL().resize((200,200)))
            labels.append(class_name)
            vectors.append(im)
            
        for i in range(100):
            gender.addTraining(labels[i],vectors[i])

        gender.train()
        
        sucesses = 0.0
        total = 0.0
        for i in range(100,len(labels)):
            guess = gender.predict(vectors[i])
            #print gender.predictSVMValues(vectors[i])
            if guess == labels[i]:
                sucesses += 1
            total += 1
        #print "ScrapShots Male/Female Rate:", sucesses/total
        self.assertAlmostEqual(sucesses/total,0.86301369863013699,places=4)
        

    def test_svm_breast_cancer_rbf(self):
        filename = os.path.join(pyvision.__path__[0],'data','ml','breast-cancer-wisconsin.data')
        reader = csv.reader(open(filename, "rb"))
        breast_cancer_labels = []
        breast_cancer_data = []
        for row in reader:
            data = []
            for item in row[1:-2]:
                if item == '?':
                    data.append(0)
                else:
                    data.append(int(item))
            breast_cancer_labels.append(int(row[-1]))
            breast_cancer_data.append(data)

        cancer = SVM(type=TYPE_SVC,random_seed=0)
        for i in range(300):
            cancer.addTraining(breast_cancer_labels[i],breast_cancer_data[i])        
        cancer.train()
        success = 0.0
        total = 0.0
        for i in range(300,len(breast_cancer_labels)):
            label = cancer.predict(breast_cancer_data[i])
            if breast_cancer_labels[i] == label:
                success += 1
            #else:
            #    print breast_cancer_labels[i],cancer.predictSVMValues(breast_cancer_data[i])
            total += 1
            
        #print "Breast Cancer Rate:",success/total
        #print cancer.training_table    
        
        self.assertAlmostEqual(success/total, 0.97744360902255634,places=4)

    
    def test_svm_breast_cancer_lin(self):
        filename = os.path.join(pyvision.__path__[0],'data','ml','breast-cancer-wisconsin.data')
        reader = csv.reader(open(filename, "rb"))
        breast_cancer_labels = []
        breast_cancer_data = []
        for row in reader:
            data = []
            for item in row[1:-2]:
                if item == '?':
                    data.append(0)
                else:
                    data.append(int(item))
            breast_cancer_labels.append(int(row[-1]))
            breast_cancer_data.append(data)

        cancer = SVM(type=TYPE_SVC,kernel=KERNEL_LINEAR,random_seed=0)
        for i in range(300):
            cancer.addTraining(breast_cancer_labels[i],breast_cancer_data[i])        
        cancer.train()
        success = 0.0
        total = 0.0
        for i in range(300,len(breast_cancer_labels)):
            label = cancer.predict(breast_cancer_data[i])
            if breast_cancer_labels[i] == label:
                success += 1
            #else:
            #    print breast_cancer_labels[i],cancer.predictSVMValues(breast_cancer_data[i])
            total += 1
        #print cancer.training_table    
        #print "Breast Cancer Rate:",success/total
        
        self.assertAlmostEqual(success/total, 0.97994987468671679,places=4)

    
    def test_svm_breast_cancer_nu(self):
        filename = os.path.join(pyvision.__path__[0],'data','ml','breast-cancer-wisconsin.data')
        reader = csv.reader(open(filename, "rb"))
        breast_cancer_labels = []
        breast_cancer_data = []
        for row in reader:
            data = []
            for item in row[1:-2]:
                if item == '?':
                    data.append(0)
                else:
                    data.append(int(item))
            breast_cancer_labels.append(int(row[-1]))
            breast_cancer_data.append(data)

        cancer = SVM(type=TYPE_NU_SVC,random_seed=0)
        for i in range(300):
            cancer.addTraining(breast_cancer_labels[i],breast_cancer_data[i])        
        cancer.train()
        success = 0.0
        total = 0.0
        #print
        #print cancer.training_table
        for i in range(300,len(breast_cancer_labels)):
            label = cancer.predict(breast_cancer_data[i])
            if breast_cancer_labels[i] == label:
                success += 1
            #else:
            #    print breast_cancer_labels[i],cancer.predictSVMValues(breast_cancer_data[i])
            total += 1
            
        #print "Breast Cancer Rate:",success/total
        
        self.assertAlmostEqual(success/total, 0.97994987468671679,places=4)
        
