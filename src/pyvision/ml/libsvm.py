'''
Created on Nov 23, 2010

@author: bolme
'''

import svm
import numpy as np
import tempfile
import os

class UntrainedClassifierError(Exception):
    pass


class _LabelMap:
    '''Converts class labels back and forth to integer codes.'''
    
    def __init__(self):
        '''Create and initialze the mapping.'''
        self._forward_map = {}
        self._backward_map = {}
        self._current_code = 0
        
    def toCode(self,label):
        '''Converts a label to an integer code. Codes are defined as needed.'''
        if not self._forward_map.has_key(label):
            self._forward_map[label] = self._current_code
            self._backward_map[self._current_code] = label
            self._current_code += 1
            
        return self._forward_map[label]

    def toLabel(self,code):
        '''Converts a code back into a label'''
        return self._backward_map[code]
    
class _LabelScale:
    '''Converts class labels back and forth to integer codes.'''
    
    def __init__(self):
        '''Create and initialze the mapping.'''
        self.mean = 0.0
        self.std = 0.0
        
    def train(self,labels,data):
        labels = np.array(labels)
        
        self.mean = labels.mean()
        self.std = labels.std()
        
        labels = (labels - self.mean)/self.std
        
        return labels,data
        
    def toScaled(self,label):
        ''''''
        return  (label - self.mean)/self.std

    def toOrig(self,code):
        ''''''
        return code * self.std + self.mean
    

class FeaturePreprocessor:
    def __init__(self):
        pass
    
    def train(self,labels,data):
        raise NotImplementedError()
    
    def __call__(self,vector):
        raise NotImplementedError()

class NoNorm(FeaturePreprocessor):
    
    def __init__(self):
        pass
    
    def train(self,labels,data):
        return labels,data
    
    def __call__(self,vector):
        return vector


class ZNormValues(FeaturePreprocessor):
    
    def __init__(self):
        pass
    
    def train(self,labels,data):
        self.means = data.mean(axis=0)
        self.stds = data.std(axis=0)

        n = len(self.means)
        self.means.shape = (1,n)
        self.stds.shape = (1,n)
        
        # TODO: Need to correct for zero values in stds
        # self.stds[np.abs(self.stds) < 1e-6] = 1.0        
        
        data = (data - self.means) / self.stds
        
        self.means = self.means.flatten()
        self.stds = self.stds.flatten()
        
        return labels,data
    
    def __call__(self,vector):
        #if not isinstance(vector,np.ndarray):
        vector = np.array(vector)
        vector =  (vector - self.means) / self.stds
        return vector

class Classifier:
    pass

class Regression:
    pass

class SVC(Classifier):
    
    def __init__(self,C=1.0,gamma=1.0,preprocessor=ZNormValues()):
        '''Create a support vector machine classifier.'''
        
        self._model = None
        
        assert isinstance(preprocessor,FeaturePreprocessor)
        self._preprocessor = preprocessor
        
        self._C = C
        self._gamma = gamma
        self._label_map = _LabelMap()
        
        
    def __getstate__(self):
        '''This function is neccessary for pickling'''
        # Translate everything but the svm because that cannot be simply pickled.
        state = {}
        for key,value in self.__dict__.iteritems():
            if key == '_model':
                filename = tempfile.mktemp()
                self._model.save(filename)
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
            if key == '_model':
                filename = tempfile.mktemp()
                open(filename,'w').write(value)
                self._model = svm.svm_model(filename)
                os.remove(filename)
                continue

            self.__dict__[key] = value

    
    def train(self,labels,data):
        '''
        Train the classifier.
        
        @param labels: A list of class labels.
        @param data: A 2D array or list of feature vectors.  One feature vector per row.
        '''
        
        # Check the types and convert to np arrays
        if isinstance(data,list) or isinstance(data,tuple):
            data = np.array(data,dtype=np.double)
            

        labels = [self._label_map.toCode(each) for each in labels]        
        labels = np.array(labels)
            
        # Preprocess the data    
        labels,data = self._preprocessor.train(labels,data)
        
        # Create the svm parameter data and problem description
        param = svm.svm_parameter(svm_type=svm.C_SVC,kernel_type = svm.RBF, C = self._C, gamma=self._gamma)
        prob = svm.svm_problem(labels.tolist(),data.tolist())
        
        # train the svm
        self._model = svm.svm_model(prob, param)
        
    
    def __call__(self,vector):
        '''Classify a feature vector.'''
        
        if self._model == None:
            raise UntrainedClassifierError()
        
        # convert to an array
        if isinstance(vector,list) or isinstance(vector,tuple):
            vector = np.array(vector,dtype=np.double)
        
        # preprocess the data
        vector = self._preprocessor(vector)

        # return the prediction
        code =  self._model.predict(vector.tolist())
        return self._label_map.toLabel(code)

class SVR(Regression):
    
    def __init__(self,epsilon=0.01,gamma=1.0,preprocessor=ZNormValues()):
        '''Create a support vector machine classifier.'''
        
        self._model = None
        
        assert isinstance(preprocessor,FeaturePreprocessor)
        self._preprocessor = preprocessor
        self._label_scale = _LabelScale()
        
        self._epsilon = epsilon
        self._gamma = gamma
        
        
    
    def __getstate__(self):
        '''This function is neccessary for pickling'''
        # Translate everything but the svm because that cannot be simply pickled.
        state = {}
        for key,value in self.__dict__.iteritems():
            if key == '_model':
                filename = tempfile.mktemp()
                self._model.save(filename)
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
            if key == '_model':
                filename = tempfile.mktemp()
                open(filename,'w').write(value)
                self._model = svm.svm_model(filename)
                os.remove(filename)
                continue

            self.__dict__[key] = value

    
    def train(self,labels,data):
        '''
        Train the classifier.
        
        @param labels: A list of class labels.
        @param data: A 2D array or list of feature vectors.  One feature vector per row.
        '''
        
        # Check the types and convert to np arrays
        if isinstance(data,list) or isinstance(data,tuple):
            data = np.array(data,dtype=np.double)
            

        labels = np.array(labels,dtype=np.double)
            
        # Preprocess the data    
        labels,data = self._preprocessor.train(labels,data)
        labels,data = self._label_scale.train(labels,data)
        
        
        # Create the svm parameter data and problem description
        param = svm.svm_parameter(svm_type=svm.EPSILON_SVR,kernel_type = svm.RBF, eps = self._epsilon, gamma=self._gamma)
        prob = svm.svm_problem(labels.tolist(),data.tolist())
        
        # train the svm
        self._model = svm.svm_model(prob, param)
        
    
    def __call__(self,vector):
        '''Classify a feature vector.'''
        
        if self._model == None:
            raise UntrainedClassifierError()
        
        # convert to an array
        if isinstance(vector,list) or isinstance(vector,tuple):
            vector = np.array(vector,dtype=np.double)
        
        # preprocess the data
        vector = self._preprocessor(vector)

        # return the prediction
        value =  self._model.predict(vector.tolist())
        return self._label_scale.toOrig(value)
