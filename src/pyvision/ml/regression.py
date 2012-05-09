'''
Created on Apr 27, 2012

@author: bolme
'''

import numpy as np
import copy
import scipy as sp

def logit_approx(p):
    p = copy.deepcopy(p)
    
    sel = p < 0.01
    p[sel] = 0.01
    sel = p > 0.99
    p[sel] = 0.99
    
    return np.log(p/(1-p))


class pdf2nll(object):
    ''' This is a wraper class that converts a pdf to a negitive log likelihood function. '''
    
    def __init__(self,pdf):
        self.pdf = pdf
        
    def __call__(self,params,obs):
        result = 0
        for x in obs: 
            result += np.log(self.pdf(params,x))
        return -result



def maxLikelihoodEstimate(obs,params,pdf=None,nll=None):
    '''
    Produces a maximum likelihood estimate of the parameters at least one pdf or nll needs to be specified.
    
    @param obs: a list of numerical values that are the observations.
    @type obs: list of numbers
    @param params: a list or numpy array containing estimates of the parameters.
    @type params: list of numbers
    @param pdf: a function returning the probability density for a distribution.
    @type pdf: pdf(params,x) -> probablity density
    @param nll: a function returning the negitive log likelihood given the parameters and observations as arguments.
    @type nll: nll(params,obs) -> negitive log likelihood.
    '''
    assert nll == None or pdf == None
    assert nll != None or pdf != None
    
    if nll == None and pdf != None:
        nll = pdf2nll(pdf)
        return sp.optimize.fmin(lambda params: nll(params,obs),params,disp=0,xtol=1.0e-7)
    elif nll != None and pdf == None:
        return sp.optimize.fmin(lambda params: nll(params,obs),params,disp=0,xtol=1.0e-7)
    else:
        raise ValueError("Must specify one of pdf or nll.")



class LogisticRegression(object):
    '''
    This object implements a logistic regression model.  
    
    For the house-votes-84.data.txt data set this learned a model that was consistant with R.
    '''
    
    def __init__(self):
        '''
        Create and initialize the model.
        '''
        self.params = None
    
    def train(self,obs,data,method='ml'):
        '''
        Train a logistic regression model.
        @param obs: these are the observations.  
        @type obs: float in [0.0:1.0]
        @param data: floating point matrix of data
        @type data: list of data points in rows
        @param method: determines the optimization criteria.
        @type method: can be 'ml' (maximum likelihood) or 'fast' (least sqaures estimate) 
        '''
        # Convert data to arrays
        data = np.array(data,dtype=np.float64)
        obs = np.array(obs,dtype=np.float64)
        
        # Set up approximate fitting
        N,D = data.shape
        r = logit_approx(obs)
        
        # add column for intercept
        data = np.concatenate((data,np.ones((N,1),dtype=np.float64)),axis=1)
        
        # get an estimate least squares fit
        x,ordinate_vals,rank,singular_vals = np.linalg.lstsq(data, r)
        x.shape = (D+1,1)
        
        if method == 'fast':
            self.params = x
            
        elif method == 'ml':
            # The negative log likelihood cost function
            def logitNll(params,obs):
                # project the data
                vals = np.dot(data,params)
                # compute the probablities
                vals = 1.0/(1.0 + np.exp(-vals))
                # Compute the negitive log likelihood
                return -(obs*np.log(vals) + (1-obs)*np.log(1-vals)).sum()
                
            # Find the optimal max likelihood estimate
            result = maxLikelihoodEstimate(obs, x.flatten(), nll=logitNll)
            
            # Save the parameters
            self.params = np.array(result)
            self.params.shape = (D+1,1)

        else: 
            raise ValueError("Unknown method type '%s'"%(method,))
        
    def predict(self,data):
        '''
        Predict the results for a dataset.
        
        @param data: floating point matrix of data
        @type data: list of data points in rows

        @returns: a probibility for each data point.
        '''
        # Convert data to arrays
        D = self.params.shape[0]-1
        data = np.array(data,dtype=np.float64)
        if len(data.shape) == 1:
            data.shape = (1,D)

        assert data.shape[1] == self.params.shape[0]-1
        N = data.shape[0]
        
        # add column for intercept
        data = np.concatenate((data,np.ones((N,1),dtype=np.float64)),axis=1)

        # project the data
        vals = np.dot(data,self.params)
        
        # compute the probablities
        vals = 1.0/(1.0 + np.exp(-vals))
        
        return vals.flatten()
