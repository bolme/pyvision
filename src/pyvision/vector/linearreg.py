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
Implementation of linear regression.
'''
import unittest
import random
import numpy
from numpy import dot,sqrt
from numpy.linalg.linalg import pinv

#TODO: make this a subclass of VectorClassifier
class LinearReg:
    def __init__(self):
        pass
    
    def train_least_squares(self, inputs, outputs):
        '''
        inputs  - is a matrix where each row is an input.
        outputs - is a matrix where each row is a corresponding output.
        
        based on: http://en.wikipedia.org/wiki/Linear_regression (2007/06/07)
        '''
        self.mat = []
        self.RMSE = []
        # create the data matrix
        for output in range(outputs.shape[1]):
            print "Training output ", output
            y = outputs[:,output]
            #print "y:\n",y
            X = inputs
            tmp = numpy.ones(shape=(X.shape[0],1))
            X = numpy.concatenate([tmp, X],axis=1)
            #print "X:\n",X
            B = dot(dot(pinv(dot(X.transpose(),X)),X.transpose()),y)
            #print "B:\n",B
            
            E = y - dot(X,B)
            self.RMSE.append(sqrt((E*E).sum()))
            #print "E:", E, E < 0.0001
            self.mat.append( B )

        self.mat =  numpy.array(self.mat)
        
        return self.RMSE
    
    def map(self, input):
        X = numpy.concatenate([numpy.ones((1,)),input.flatten()])
        #print X
        return dot(self.mat,X)

    def __call__(self,input):
        return self.map(input)
        
        
        
class TestLinearReg(unittest.TestCase):
    
    def setUp(self):
        pass   
        
    def tearDown(self):
        pass
    
    def test_line(self):
        inputs = numpy.array([[0],[1],[2],[3]],'f')
        targets = numpy.array([[0,3],[1,2],[2,1],[3,0]],'f')
        
        reg = LinearReg()
        reg.train_least_squares(inputs,targets)
        
        for i in range(10):
            print i,reg.map(numpy.array([i],'f'))

            
    def test_cos(self):
        pass
    
    def test_sin(self):
        pass

if __name__ == "__main__":
    #unittest.main()
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLinearReg)
    unittest.TextTestRunner(verbosity=2).run(suite)
    

        