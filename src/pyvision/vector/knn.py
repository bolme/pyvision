# PyVision License (http://pyvision.sourceforge.net)
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

import numpy as np

class PNorm:
    def __init__(self,p):
        self.p = float(p)
    
    def __call__(self, points, data):
        r,c = data.shape
        
        dist_mat = []
        if self.p == np.inf:
            for pt in points:
                pt = pt.reshape(1,c)
                row =  np.amax(np.abs(data - pt),axis=-1)
                dist_mat.append(row)
        else:
            for pt in points:
                pt = pt.reshape(1,c)
                row = np.sum(np.abs(data - pt)**self.p,axis=1)**(1.0/self.p)
                dist_mat.append(row)
        
        return np.array(dist_mat)
    
def chisquared(points, data):
        r,c = data.shape
        
        dist_mat = []
        for pt in points:
            pt = pt.reshape(1,c)
            tmp1 = (data - pt)**2
            tmp2 = data + pt + 0.00001
            tmp3 = tmp1/tmp2
            
            row = np.sum(tmp3,axis=1)
            dist_mat.append(row)
        
        return np.array(dist_mat)
        
        
def correlation(points,data):
    '''
    Compute the correlation between points and data where 
    points are stored as rows.
    '''
    pr,pc = points.shape
    dr,dc = data.shape
    
    points = points - points.mean(axis=1).reshape(pr,1)
    data = data - data.mean(axis=1).reshape(dr,1)
    
    ps = 1.0/np.sqrt((points*points).sum(axis=1)).reshape(pr,1)
    ds = 1.0/np.sqrt((data*data).sum(axis=1)).reshape(dr,1)
 
    points = points * ps
    data = data * ds
    
    corr = np.dot(points,data.transpose())
    return corr
    

class KNearestNeighbors(object):
    """
    Basic k nearest neighbors algorithm.

    Based on the scipy.spatial.kdtree interface written by Anne M. Archibald 2008.
    
    This class performs a search over a set of D dimensional points and returns
    the k points that are nearest to a given point. 
    
    This class supports by default Minkowski p-norm distance measures and also
    cosine angle, and correlation similarity measures.  The class also supports
    the uses of user defined distance and similarity measures.
    """

    def __init__(self, data, p=2, is_distance=True):
        """Construct a nearest neighbor algorithm.

        Parameters:
        ===========

        data : array of n points with dimensionality d, shape (n,d).
            The data points to be indexed. This array is not copied, and
            so modifying this data will result in bogus results.
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use. 
            1 is the sum-of-absolute-values "Manhattan" distance
            2 is the usual Euclidean distance
            infinity is the maximum-coordinate-difference distance
            Also accepts the keywords "Manhattan", "Euclidean", and 
            "Correlation", or p can also be a user defined function with will 
            compute a distance matrix between two sets of points. 
        is_distance: True or False.  Determines if a user defined function is
            treated as a distance (smaller is better) or a similarity (larger
            values are better).
        """
        # Some basic tests to make sure data is of the correct type.
        if isinstance(p,int): 
            assert p > 0
        elif isinstance(p,str):
            assert p in ("Manhattan","Euclidean","Correlation")
        else:
            pass # Assume that p is a user specified function of proper type
            
        self.data = data
        self.p = p
        self.is_distance = is_distance
        
        
    def query(self, x, k=1, p=None, is_distance=True):
        """query the instance for nearest neighbors

        Parameters:
        ===========

        x : array-like, last dimension self.k
            An array of points to query.
        k : integer
            The number of nearest neighbors to return.
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use. 
            1 is the sum-of-absolute-values "Manhattan" distance
            2 is the usual Euclidean distance
            infinity is the maximum-coordinate-difference distance
            Also accepts the keywords "Manhattan", "Euclidean", and 
            "Correlation", or p can also be a user defined function with will 
            compute a distance matrix between two sets of points. 

        Returns:
        ========
        
        d : array of floats
            The distances to the nearest neighbors. 
            If x has shape tuple+(self.k,), then d has shape tuple if 
            k is one, or tuple+(k,) if k is larger than one.  Missing 
            neighbors are indicated with infinite distances.  If k is None, 
            then d is an object array of shape tuple, containing lists 
            of distances. In either case the hits are sorted by distance 
            (nearest first).
        i : array of integers
            The locations of the neighbors in self.data. i is the same
            shape as d.
        """
        # check the input
        assert x.shape[-1] == self.data.shape[1]
        assert isinstance(k,int) and k > 0
        
        if len(x.shape) == 1:
            x = x.reshape(1,k)
            
        if p == None:
            p = self.p
            is_distance = self.is_distance
            
        # compute the distances between the input points and output points
        if p==np.inf or (isinstance(p,float) or isinstance(p,int)) and p >= 1.0:
            is_distance = True
            dist = PNorm(self.p)
        elif p == "Correlation":
            is_distance = False
            dist = correlation()
        elif p == "Euclidean":
            is_distance = True
            dist = PNorm(2)
        elif p == "Manhattan":
            is_distance = True
            dist = PNorm(1)
        else:
            dist = p #assume p is a user defined function
            
            
        # find the distance matrix between points
        dist_mat = dist(x,self.data)
        
        # set the index matrix
        dist_sort = np.argsort(dist_mat)
        if not is_distance:
            dist_sort = np.fliplr(dist_sort)
        dist_sort = dist_sort[:,:k]
        
        # sort the matrix
        rows = []
        for i in range(dist_mat.shape[0]):
            row = dist_mat[i,dist_sort[i]]
            rows.append(row)
        
        # assemble the results into arrays for the proper return types 
        return np.array(rows), dist_sort   
        
  
        
        