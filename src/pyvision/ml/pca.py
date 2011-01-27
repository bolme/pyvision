'''
Created on Jan 17, 2011

@author: bolme
'''

import numpy as np
import scipy.linalg as la

def prcomp(data,center=True,scale=False):
    '''
    data vectors are in rows
    
    This function has been compared to R to verify that it produces similar results.
    
    @param data: a data matrix with vectors in rows
    @param center: subtract the mean vector from the data
    @param scale: scale the values to have unit variance
    @returns: stdevs,rotation,[center],[scale]
    '''
    data = data.copy()
    r,c = data.shape
    
    # center the data 
    if center:
        ctr = data.mean(axis=0).reshape(1,c)
        data = data - ctr 
    
    # scale the data
    if scale:
        scl = data.std(axis=0,ddof=1).reshape(1,c)
        data = data/scl 
        
    # decompose the data using svd
    u,val,vt = la.svd(data,full_matrices=False)

    # compute the standard deviations from the singular values
    standard_dev = val/np.sqrt(r-1)
    
    # Vt.T are the basis vectors
    result = [standard_dev,vt.T]
    
    # format result
    if center:
        result.append(ctr)
        
    if scale:
        result.append(scl)
        
    return result


