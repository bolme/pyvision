'''
Created on Nov 23, 2010

@author: bolme
'''

import numpy as np
import itertools as it
import copy
import pyvision as pv
import time

class CrossValidation:
    
    def __init__(self,n_folds=5):
        ''' Initializing a cross validation algorithm. '''
        self.n_folds=n_folds
        self.best_score = None
        self.best_tuning=None
        self.training_data = None
        self.training_labels = None
        self.folds = None
        self.options = []
        self.tuning_data = None
        self.classification = True
        
        self._alg_class = None
        
        
    def setAlgorithm(self,alg_class,args=[],kwargs={}):
        '''
        
        '''
        assert issubclass(alg_class,Classifier) or issubclass(alg_class,Regression)
        self.classification = issubclass(alg_class,Classifier)
        self._alg_class = alg_class
        self._alg_args = args
        self._alg_kwargs = kwargs
        
    def getTunedAlgorithm(self):
        if self.best_tuning == None:
            self.tuneAlogirthmExaustive()

        alg = self._alg_class(*self._alg_args,**self.best_tuning)
        labels = self.training_labels
        data = self.training_data
        alg.train(labels,data)
        
        alg.tuning_data = self.tuning_data
        alg.best_score = self.best_score
        alg.best_tuning = self.best_tuning
        
        return alg

        
    def setTraining(self,labels,data,folds=None):
        '''
        Adds training data.
        
        @param label: a list of labels (int,str) or regression outputs (float)
        @param data: a matrix of data values each row is a feature vector
        @keyword fold: a list specifying the folds for validation. If None it will be randomly assigned.
        '''
        self.training_labels = np.array(labels)
        self.training_data = np.array(data)
        if folds == None:
            n = len(self.training_labels)
            reps = n/self.n_folds + 1
            folds = np.tile(np.arange(self.n_folds),reps)
            folds = folds[:n]
            #print folds
            np.random.shuffle(folds)
            #print repr(folds)
            
        self.folds = np.array(folds)
        
        
    def addTunableOption(self,keyword,values):
        '''
        This specifie
        @param keyword: A keyword to use in the algorithm initializer.
        @type keyword: str
        @param values: A list or tuple of values to use for keyword.
        @type values: list
        '''
            
        self.options.append([keyword,values])
        
    def tuneAlogirthmExaustive(self,verbose = True):
        '''
        This conducts an exaustive search of all tunable parameters to find the best tuning.
        '''
        if len(self.options) == 0:
            score = self.runTest(self._alg_class,self._alg_args,self._alg_kwargs)
            self.best_score = score
            self.best_tuning = copy.deepcopy(self._alg_kwargs)
            results = pv.Table()
            results[0,'tuning'] = 'None'
            results[0,'score'] = score
            self.tuning_data = results
        else:
            # Get keywords and values
            keywords = [key for key,_ in self.options]
            values = [val for _,val in self.options]

            r = 0
            results = pv.Table()
            
            # Test all possible tuning assignments
            for vals in it.product(*values):
                
                # construct kwargs for this assignment
                kwargs = copy.deepcopy(self._alg_kwargs)
                for i in range(len(keywords)):
                    kw = keywords[i]
                    val = vals[i]
                    kwargs[kw] = val
                    results[r,kw] = val
                
                # run a cross validation test
                score = self.runTest(self._alg_class,self._alg_args,kwargs,verbose=verbose)
                
                # save the best score
                if self.classification and (self.best_score == None or score > self.best_score):
                    self.best_score = score
                    self.best_tuning = kwargs
                if not self.classification and (self.best_score == None or score < self.best_score):
                    self.best_score = score
                    self.best_tuning = kwargs
                    
                # construct a table of tuning information
                results[r,'score'] = score
                r += 1
                #print results

            self.tuning_data = results
            #print results
            
    def runTest(self, alg_class, args, kwargs,verbose=True):

        squared_error = 0.0
        successes = 0
        
        for fold in range(self.n_folds):
            alg = alg_class(*args,**kwargs)
            
            labels = self.training_labels[fold != self.folds]
            data = self.training_data[fold != self.folds]
            
            #print len(labels),len(self.training_labels)
            
            alg.train(labels,data)
            
            for i in range(len(self.folds)):
                
                if self.folds[i] != fold:
                    continue
                
                prediction = alg(self.training_data[i])
                truth = self.training_labels[i]
                
                if isinstance(prediction, float):
                    squared_error += (prediction-truth)**2
                    #self.classification = False
                else:
                    successes += prediction == truth
                    #print successes
                
        if self.classification:
            score = float(successes)/len(self.folds)
            #print score
        else:
            score = np.sqrt(squared_error/len(self.folds))
            #print score
        tmp = str(kwargs)
        if verbose:
            print("%-40s %8.6f"%(tmp[:40],score),successes,'/',len(self.folds))
        return score
                
            
            
            
                
        
        
    def getBestTuning(self):
        '''
        @returns: the best known tuning.
        '''
        
        
class Validation(CrossValidation):
    
    def __init__(self,n_folds=3):
        CrossValidation.__init__(self, n_folds=n_folds)
        
        self.best_alg = None
        self.best_eval = 0.0
        
        self.results = pv.Table()
        
        
    def runTest(self, alg_class, args, kwargs,verbose=True):
        alg = alg_class(*args,**kwargs)

        squared_error = 0.0
        successes = 0
        
        fold = 0
 
        alg = alg_class(*args,**kwargs)
        
        labels = self.training_labels[fold != self.folds]
        data = self.training_data[fold != self.folds]
                
        alg.train(labels,data)
        
        start = time.time()
        count = 0
        for i in range(len(self.folds)):
            if self.folds[i] != fold:
                continue
            
            count += 1
            
            prediction = alg(self.training_data[i])
            truth = self.training_labels[i]
            
            if isinstance(prediction, float):
                squared_error += (prediction-truth)**2
                #self.classification = False
            else:
                successes += prediction == truth
                #print successes
        stop = time.time()        
        
        
        if self.classification:
            score = float(successes)/count
            
            row = self.results.nRows()
            
            for key,value in kwargs.items():
                self.results[row,key] = value
                
            new_best = False
            if self.best_eval < score:
                new_best = True
                self.best_eval = score
                self.best_alg = alg
                self.best_time = stop-start
            elif self.best_eval == score and self.best_time > stop-start:
                new_best = True
                self.best_eval = score
                self.best_alg = alg
                self.best_time = stop-start

            self.results[row,'time'] = stop-start
            self.results[row,'score'] = score
            self.results[row,'new_best'] = new_best
            print(self.results)
        else:
            score = np.sqrt(squared_error/count)
        tmp = str(kwargs)
        if verbose:
            print("%-40s %8.6f"%(tmp[:40],score),successes,'/',len(self.folds))
        return score
                
            
            
    
    
    
        
