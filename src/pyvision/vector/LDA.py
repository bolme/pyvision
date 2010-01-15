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

##########################################################
## Multiple class classification using fisher LDA
## 
## TrainLDA(X,Y): trains the classifier
##              X -> Input dataset(training data)
##              Y -> label for each row
## Returns      lda -> the fisher discriminant w
## Usage: train the classifier to obtain the discriminant
##
## Author: Mohammad Nayeem Teli 
###########################################################

import unittest
from numpy import *
from scipy import *
import scipy as sp
import numpy as np
import scipy.linalg

### 
### train the fisher linear discriminant using the training data X
### with the class labels Y
###

def trainLDA(X, Y, reg="AUTO"):
    '''
    method trainLDA(X, Y) obtains the fisher discriminant for multiple class dataset.

    Example:
    X = matrix( [[1,1.5,1.2],[2.5,3,2.4],[1.2,1.6,1.0],[1,1.7,0.9],[1,3.1, 2.1] ])
    Y = array( [1,2,3,2,1] )  # class labels
    lda = trainLDA(X, Y)      # Train the classifier getting 
                  # the discriminant lda

    @param X: It is the input dataset with each row as a sample.
    @param Y: The labels for each sample of X
    @keyparam reg: a regularization parameter
    @return: a tuple containing the [Fisher discriminants , and eigenvalues]
    '''
    
    X = np.array(X)
    Y = np.array(Y)
    
    classes = unique(Y)  # Unique classes
    p = size(X, axis=1)  # columns of training dataset
    N = len(Y)         # Number of samples
    K = len(classes)     # Number of unique classes
    
    priors = zeros(K)
    means = tile(0.,(K,p)) # column means for each class
    meanSum = tile(0.,(1,p)) # column means for each class
    totalmean = tile(0.,(1,p)) # column means for each class
    # within class covariance matrix
    Sw = tile(0.,(p,p)) # sum of covariances of the classes
    Sb = tile(0.,(p,p)) # Between class scatter matrix
    #w = tile(0.,(p,K))    # fisher discriminant matrix
    ###
    ### Loop through the classes to obtain the class covariances 
    ###
    for i in arange(0,K):

        # Get the rows corresponding to a particular class
        mask = Y==classes[i]     
        
        numThisclass = sum(mask)

        # probability of a class
        priors[i] = float(numThisclass)/N

        Xmat = X[mask,:]
        means[i,:]= Xmat.mean(axis=0)
        meanSum = meanSum + numThisclass*means[i,:] 
        Xmean = Xmat - means[i,]

        # Within class scatter matrix
        Sw = Sw + dot(Xmean.T,Xmean)

    # Total mean 
    totalmean = meanSum / float(N)

    ###
    ### Loop through the classes to obtain the scatter matrix
    ### between classes
    ###
    for i in arange(0,K):

        # Get the rows corresponding to a particular class
        mask = Y==classes[i]     
        numThisclass = sum(mask)

        Xmat = X[mask,:]
        means[i,:]= Xmat.mean(axis=0)
        meanD = means[i,] - totalmean

        # Between class scatter matrix
        Sb = Sb + numThisclass*dot(meanD.T,meanD)


    # Fisher discriminant w is obtained by eigen value
    # decomposition of the equation 
    # (Sw^-1)*Sb* w = l * w
    
    # TODO: Figure out if regularization is needed. Sw is Sigular?
    if reg == "AUTO":
        # Automatically select the regulariazion parameter
        # TODO: Many people suggest scaling the by the mean of the diagonal which may be a good way to select a regularization parameter. Why?
        diagElements = diag(Sw)
        meanDiag = diagElements.mean(axis=0)
        delta = 0.001*meanDiag
        deltaI = delta*eye(p)
    elif isinstance(reg,float):
        # This method is suggested by wikipedia
        deltaI = reg*reg*eye(p)
    else: 
        deltaI = np.zeros((p,p))
    
    e = sp.linalg.eigh(Sw,eigvals_only=True)
    
    # check if the smallest eigenvalue is close to zero and if so... regularize
    if np.abs(e[0]) < 1e-7:
        Sw = Sw + deltaI
    #SwInv = linalg.inv(Sw)  # Inverse of the within class scatter matrix
    #print "SwInv",SwInv,Sw
    #print "Sb",Sb
    #val,vec = linalg.eigh(Sw)
    #print val
    #val,vec = linalg.eigh(SwInv)
    #print val
    #val,vec = linalg.eigh(Sb+deltaI)
    #print val

    #A = dot(SwInv,Sb+deltaI) # Matrix multiplication
    #print "A",A
    e, w = sp.linalg.eig(Sb,Sw) # Eigen value decomposition

    print e
    
    # eigen vector corresponding to the largets eigen value
    #w = w[:,0]  
    #w = dot(Sw.I,meandiff) # Matrix multiplication
    #print np.abs(e)[:K-1]
    #print np.abs(e)[K-1:]
    #print w
    #assert 0
    #print e[0:-(K-1)],e[-(K-1):]
    
    order = e.argsort()
    order = order[::-1]
    
    w = w[:,order]
    e = e[order]
    
    w = w[:,:(K-1)].real
    e = e[:(K-1)].real
    lda = w,e
    
    #print dot(w.transpose(),w)  # This should be the identity
    return lda


class _LDATest(unittest.TestCase):
    def setUp(self):
        pass
    
    def testLDA(self):
        self.assert_(False)