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

import logging
from pyvision.types.Image import Image
import numpy
from numpy.linalg import svd

def show(feature):
    Image(feature.reshape(64,64)).show()


#####################################################################
class PCA:
    ''' Performs principal components analysis on a set of images, features, or vectors. '''

    
    #----------------------------------------------------------------
    def __init__(self, center_points=True, one_std=True ):
        ''' Create a PCA object '''
        self.logger = logging.getLogger("pyvis")
        self.features = []
        self.center_points = center_points
        
    
    #----------------------------------------------------------------
    def __setstate__(self,state):
        # Translate everything but the svm because that cannot be simply pickled.
        for key,value in state.iteritems():
            if key == 'basis':
                # Workaround for a numpy pickle problem.
                # TODO: Remove this after numpy bug 551 is fixed.
                self.basis = numpy.array(value)
                continue

            self.__dict__[key] = value


    #----------------------------------------------------------------
    def addFeature(self,feature):
        ''' Add a feature vector to the analysis. '''
        feat = None
        
        feat = self.toVector(feature)

        self.features.append(feat)

    
    #----------------------------------------------------------------
    def toVector(self,feature):
        if isinstance(feature,Image):
            feat = feature.asMatrix2D().flatten()
            return feat
        if isinstance(feature,numpy.ndarray):
            feat = feature.flatten()
            return feat
        raise TypeError("Could not create feature from type: %s"%type(feature))


    #----------------------------------------------------------------        
    def train(self, drop_front=None, number=None, energy=None): 
        ''' Compute the PCA basis vectors using the SVD '''
        self.logger.info("Computing PCA basis vectors.")
        
        # One feature per row
        features = numpy.array(self.features)
        
        if self.center_points:
            self.center = features.mean(axis=0)
            for i in range(features.shape[0]):
                features[i,:] -= self.center
                #show(features[i,:])
        
        features = features.transpose()

        u,d,vt = svd(features,full_matrices=0)
        if drop_front:
            u = u[:,drop_front:]
            d = d[drop_front:]
            
        if number:
            u = u[:,:number]
            d = d[:number]
        if energy:
            # compute teh sum of energy
            s = 0.0
            for each in d:
                s += each*each
            cutoff = energy * s
            
            # compute the cutoff
            s = 0.0
            for i in range(len(d)):
                each = d[i]
                s += each*each
                if s > cutoff:
                    break
            
            u = u[:,:i]
            d = d[:i]   

        self.basis = u.transpose()
        self.values = d
        
        # Remove features because they are no longer needed and 
        # they take up a lot of space.
        del self.features
        

    #----------------------------------------------------------------
    def project(self, feature, whiten=False):
        ''' Transform a feature into its low dimentional representation '''
        feat = self.toVector(feature)
        
        if self.center_points:
            feat = feat - self.center
        
        vec = numpy.dot(self.basis,feat)
        
        if whiten:
            vec = vec/numpy.sqrt(self.values)
            
        return vec
        
        
    #----------------------------------------------------------------
    def reconstruct(self,feat):
        ''' return the eigen values for this computation '''
        feat = numpy.dot(self.basis.transpose(),feat)
        if self.center_points:
            feat += self.center
        return feat

        
    #----------------------------------------------------------------
    def getBasis(self):
        ''' return the eigen vectors returned by the computation '''
        return self.basis

        
    #----------------------------------------------------------------
    def getValues(self):
        ''' return the bases used for transforming features '''
        return self.values


# TODO:  Needs Unit Tests
    