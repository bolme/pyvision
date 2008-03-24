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

'''
The purpose of this class is to provied common services 
and a common interface to classifers.  For the most part
this class provides normalization services.  Many 
classification algorthms assume that the input values have 
zero mean and a unit variance.  This class also provides 
PCA based normalization that also reduces dimensionality.
'''

from numpy import array,mean,std
import pyvision
from pyvision.types.Image import Image
from pyvision.vector.PCA import PCA
import unittest
import os.path


NORM_NONE="NONE"
NORM_PCA="PCA_WHITEN"
NORM_VALUE="VALUE"
NORM_AUTO="AUTO"

REG_NORM_NONE="NONE"
REG_NORM_VALUE="VALUE"

TYPE_TWOCLASS="TWOCLASS"
TYPE_MULTICLASS="MULTICLASS"
TYPE_REGRESSION="REGRESSION"

class VectorClassifier:
    def __init__(self, classifer_type, normalization=NORM_AUTO, reg_norm=REG_NORM_VALUE, pca_basis=0.95, pca_drop=0):
        '''
        Configure some defaults for the classifier value normalizion.
        
        This configures some defalts for the classifier such as the
        type of classifier, and how values are normalized.
        '''
        
        # Setup basic configuration
        self.type = classifer_type
        self.norm = normalization
        self.reg_norm = reg_norm
        self.pca_basis = pca_basis
        self.pca_drop = pca_drop
        
        self.labels = []
        self.vectors = []
        self.vector_length = None
        
        self.reg_mean = 0.0
        self.reg_std  = 1.0
    
    
    def trainNormalization(self):
        '''
        Learn the range of values that are expected for labels and data.
        Then setup for normalization.
        '''
        
        assert len(self.labels) >= 2
        
        if self.type == TYPE_TWOCLASS or self.type == TYPE_MULTICLASS:
            # Learn the classes
            n_classes = 0
            self.class_map = {}
            
            for label in self.labels:
                if not self.class_map.has_key(label):
                    self.class_map[label] = n_classes
                    n_classes+=1
            
            if self.type == TYPE_MULTICLASS:
                assert n_classes >= 2
            if self.type == TYPE_TWOCLASS:
                assert n_classes == 2
            
            self.class_inv = {}
            for key,value in self.class_map.iteritems():
                self.class_inv[value] = key

            new_labels=[]
            for each in self.labels:
                new_labels.append(self.class_map[each])
            self.labels = new_labels
                
        if self.type == TYPE_REGRESSION:
            self.reg_mean = mean(self.labels)
            self.reg_std = std(self.labels)  
            
            new_labels=[]
            for each in self.labels:
                new_labels.append((each - self.reg_mean)/self.reg_std)
            self.labels = new_labels
            
        #test length
        shape = self.vectors[0].shape
        assert len(shape) == 1

        for each in self.vectors:
            assert shape == each.shape
            
        #crate a data matrix
        data = array(self.vectors,'d')
        if self.norm == NORM_AUTO:
            self.norm = NORM_VALUE
            if data.shape[1] > 128:
                self.norm = NORM_PCA
        
        #Setup value normalization
        if self.norm == NORM_VALUE:
            self.dmean = data.mean(axis=0)
            self.dstd  = data.std(axis=0)
            self.vectors = (data-self.dmean)/self.dstd
            
        elif self.norm == NORM_PCA:
            self.pca = PCA()
            for vec in self.vectors:
                self.pca.addFeature(vec)

            if self.pca_basis > 1:
                self.pca.train(drop_front=self.pca_drop,number=self.pca_basis)
            else:
                self.pca.train(drop_front=self.pca_drop,energy=self.pca_basis)
                
            new_vectors = []
            for each in self.vectors:
                new_vectors.append(self.pca.project(each,whiten=True))
                self.vectors=array(new_vectors,'d')
                
        
    
    def normalizeVector(self,data):
        '''
        Normalize the values in a data vector to be mean zero.
        '''
        if self.norm == NORM_NONE:
            return data
        elif self.norm == NORM_VALUE:
            return (data-self.dmean)/self.dstd
        elif self.norm == NORM_PCA:
            return self.pca.project(data,whiten=True)
        else:
            raise NotImplementedError("Could not determine nomalization type: "+ self.norm)
        
    
    def addTraining(self,label,data):
        '''
        Add a training sample.  Data must be a vector of numbers.
        '''
        if self.type == TYPE_REGRESSION:
            self.labels.append(float(label))
        else:
            self.labels.append(label)
            
        if isinstance(data,Image):
            data = data.asMatrix2D().flatten()   
        data = array(data,'d').flatten()
        
        self.vectors.append(data)
        
    def predict(self,data):
        '''
        Predict the class or the value for the input data.
        
        This function will perform value normalization and then 
        delegate to the subclass to perform classifiaction or 
        regression.
        '''
        if isinstance(data,Image):
            data = data.asMatrix2D().flatten()   
        data = array(data,'d').flatten()
        
        data = self.normalizeVector(data)
        
        value = self.predictValue(data)
        
        if self.type == TYPE_TWOCLASS or self.type == TYPE_MULTICLASS:
            return self.invertClass(value)
        if self.type == TYPE_REGRESSION:
            return self.invertReg(value)
        
    def predictValue(self,data):
        ''' 
        Override this method in subclasses.
        Input should be a numpy array of doubles
        
        If classifer output is int
        If regression output is float
        '''
        raise NotImplementedError("This is an abstract method")
        
    def train(self):
        '''
        Train the classifer on the training data.
        
        This normalizes the data and the labels, and then passes the 
        results to the subclass for training.
        '''
        self.trainNormalization()
        
        self.trainClassifer(self.labels,self.vectors)
        
        # remove training data
        del self.labels
        del self.vectors
    
    def trainClassifer(self,labels,vectors):
        raise NotImplementedError("This is an abstract method")
    

    def invertReg(self,value):
        '''Convert the regression value back to the label scales'''
        return value*self.reg_std + self.reg_mean
         
    
    def invertClass(self,value):
        '''Map an integer back into a class label'''
        return self.class_inv[value]
        
        
def _mse(a,b):
    assert len(a) == len(b)
    ss = 0.0
    for i in range(len(a)):
        d = float(a[i])-float(b[i])
        ss += d*d
    return ss/len(a)

class _TestVectorClassifier(unittest.TestCase):
    
    def setUp(self):
        
        # a simple binary two class
        xor = VectorClassifier(TYPE_TWOCLASS)
        xor.addTraining(0,[0,0])
        xor.addTraining(0,[1,1])
        xor.addTraining(1,[0,1])
        xor.addTraining(1,[1,0])
        self.xor = xor
        
        # synthetic linear regression
        rega = VectorClassifier(TYPE_REGRESSION)
        filename = os.path.join(pyvision.__path__[0],'data','synthetic','regression.dat')
        reg_file = open(filename,'r')
        for line in reg_file:
            datapoint = line.split()
            rega.addTraining(float(datapoint[0]),[float(datapoint[3]),float(datapoint[4]),float(datapoint[5])])
        self.rega = rega      
        
        # image classification
        gender = VectorClassifier(TYPE_TWOCLASS)
        filename = os.path.join(pyvision.__path__[0],'data','csuScrapShots','gender.txt')
        f = open(filename,'r')
        for line in f:
            im_name, class_name = line.split()
            im_name = os.path.join(pyvision.__path__[0],'data','csuScrapShots',im_name)
            im = Image(im_name)
            im = Image(im.asPIL().resize((200,200)))
            gender.addTraining(class_name,im)
        self.gender = gender
    
    def test_vc_create(self):
        vc = VectorClassifier(TYPE_TWOCLASS)
        vc = VectorClassifier(TYPE_MULTICLASS)
        vc = VectorClassifier(TYPE_REGRESSION)

    def test_vc_normalize(self):
        # This should test class normalization
        self.xor.trainNormalization()
        self.assert_(self.xor.norm == NORM_VALUE)
        self.assert_( _mse(self.xor.dmean, [0.5,0.5]) < 0.0001 )
        self.assert_( _mse(self.xor.dstd, [0.5,0.5]) < 0.0001 )
        self.assert_(self.xor.class_map == {0:0,1:1})
        self.assert_(self.xor.class_inv == {0:0,1:1})
        
        # This should test value normalization
        self.rega.trainNormalization()
        self.assert_(self.rega.norm == NORM_VALUE)
        self.assertAlmostEqual( self.rega.reg_mean, 85.49472, places = 4)
        self.assertAlmostEqual( self.rega.reg_std,  12.20683, places = 4)
        self.assert_( _mse(self.rega.dmean, [29.082505, 29.9741642, 30.4516687]) < 0.0001 )
        self.assert_( _mse(self.rega.dstd, [11.08164301,11.983678,11.18806686]) < 0.0001 )
        
        # This should test PCA normalization
        self.gender.trainNormalization()
        self.assertEqual(self.gender.norm, NORM_PCA)
        self.assertEqual(len(self.gender.pca.getValues()), 73)
        self.assert_(self.gender.class_map == {'M': 1, 'F': 0})
        self.assert_(self.gender.class_inv == {0: 'F', 1: 'M'})
        
        
        
        
        
        