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

from pyvision.vector.VectorClassifier import VectorClassifier,TYPE_REGRESSION
from numpy import array,dot
from numpy.linalg import lstsq

class Polynomial2D(VectorClassifier):
    def __init__(self,order=2,**kwargs):
        #FIXME: DOcument this code
        '''
        This class fits a polynomial to a function of 2 variables.
        '''
        self.order = order
        self.x = None
        VectorClassifier.__init__(self,TYPE_REGRESSION,**kwargs)
        
        
    def trainClassifer(self,labels,vectors,ilog=None):
        '''
        Train the polynomial.  Do not call this function
        manually, instead call the train function on the super
        class.
        '''
        #build matrix
        matrix = []
        for each in vectors:
            if len(each) != 2:
                raise ValueError("ERROR: Vector length=%d.  Polynomial2D only predicts for vectors of length 2."%len(each))
            x,y = each
            matrix.append(self.buildRow(x,y))
        
        matrix = array(matrix)
        labels = array(labels)
        
        x,resids,rank,s = lstsq(matrix,labels)
        
        self.x = x
        self.resids = resids
        self.rank = rank
        self.s = s
        
        if rank != matrix.shape[1]:
            print "WARNING: Polynomial is not fully constrained."
        
                

    def buildRow(self,x,y):
        '''
        Method for private use only.
        '''
        row = [1.0]
        for o in range(1,self.order+1):
            for i in range(o+1):
                row.append(float(x)**i*float(y)**(o-i))
        return row

    def predictValue(self,data,ilog=None):
        '''
        Method for private use only.  Call super class predict.
        '''
        if len(data) != 2:
            raise ValueError("Polynomial2D only predicts for vectors of length 2.")

        x,y = data
        return dot(self.x , self.buildRow(x,y))

import unittest

class _PolyTest(unittest.TestCase):
    def test_buildrow(self):
        poly = Polynomial2D(order=3)
        row = array(poly.buildRow(2,2))
        error = row-array([1.0,2.0,2.0,4.0,4.0,4.0,8.0,8.0,8.0,8.0])
        sse = (error*error).sum()
        self.assert_(sse < 0.001)

        row = array(poly.buildRow(2,1))
        error = row-array([1.0,1.0,2.0,1.0,2.0,4.0,1.0,2.0,4.0,8.0])
        sse = (error*error).sum()
        self.assert_(sse < 0.001)

        row = array(poly.buildRow(1,2))
        error = row-array([1.0,2.0,1.0,4.0,2.0,1.0,8.0,4.0,2.0,1.0])
        sse = (error*error).sum()
        self.assert_(sse < 0.001)

    def test_train(self):
        poly = Polynomial2D(order=2)
        for x in range(-8,9):
            for y in range(-8,9):
                val = -5 + 3*y + 2*x - 4*y*y + 2*y*x+ x*x
                poly.addTraining(val,[x,y])
        poly.train()
        for x in range(-8,9):
            for y in range(-8,9):
                val  = -5 + 3*y + 2*x - 4*y*y + 2*y*x+ x*x
                pred = poly.predict([x,y])
                self.assertAlmostEqual(val,pred,delta=0.0001)





