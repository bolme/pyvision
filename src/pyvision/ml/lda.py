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
Created on Jan 27, 2011

@author: bolme
'''
import numpy as np
import scipy as sp
import scipy.linalg as la


def lda(data,labels,reg=0.0):
    '''
    Compute the lda basis vectors.  Based on Wikipedia and verified against R
    
    @param data: the data matrix with features in rows.
    @type data: np.array
    @param labels: a corresponding 1D array of labels, one label per row in data
    @type labels: np.array (int or str)
    @return: (lda_values,lda_basis,means,priors)
    @rtype: (np.array,np.array,dict,dict)
    '''
    means = {}
    priors = {}
    
    classes = list(set(labels))
    classes.sort()
    
    # number of classes
    C = len(classes)
    
    # number of data points
    N = data.shape[0]
    
    # number of dimensions
    D = data.shape[1]

    for key in classes:
        priors[key] = float((labels == key).sum())/labels.shape[0]
        means[key] = data[labels==key,:].mean(axis=0)
    
    # Compute the between class cov
    t1 = [mean for key,mean in means.iteritems()]
    t1 = np.array(t1)
    
    # mean of class means
    t2 = t1.mean(axis=0)
    t3 = t2 - t1
    Sb = np.dot(t3.T,t3)/(C-1)
    
    # size of cov matrix should be DxD
    assert Sb.shape == (D,D)
    
    # Compute the within class cov
    Sw = None
    
    data_w = data.copy()
    for key in classes:
        # subtract the class mean (c_mean) from each data point
        c_mean = means[key].reshape(1,D)
        data_w[labels == key,:] -= c_mean
        
    Sw = np.dot(data_w.T,data_w) / (N-C) # within class scatter

    # Check the shape of SW    
    assert Sw.shape == (D,D)
    
    #Compute vectors using generalized eigenvector solver: Sb v = l Sw v
    if reg >= 0:
        Sw = Sw+reg*np.eye(Sw.shape[0]) # regularization for stability
    val,vec = la.eigh(Sb,Sw)
    
    # Reorder vectors so the most important comes first
    order = val.argsort()[::-1] # reverse order
    val = val[order]
    vec = vec[:,order]
    val = val[:C-1]
    vec = vec[:,:C-1]
    
    #scale the eigen values
    val = val/val.sum()
    
    return val,vec,means,priors
    
    
    

