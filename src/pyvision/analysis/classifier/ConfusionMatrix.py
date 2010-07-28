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

import unittest

from pyvision.analysis.stats import pbinom,qbinom,cibinom
from pyvision.analysis.Table import Table

class McnemarsExactTest:
    def __init__(self,algorithm_a=None, algorithm_b=None,test_name="Unknown"):
        self.algorithm_name = algorithm_name
        self.test_name = test_name
        
    def addData(self,alg1,alg2,weight=1):
        ''' Add data from a test '''
        
        
        

class ConfusionMatrix(Table):
    def __init__(self, algorithm_name=None, test_name=None):
        Table.__init__(self,default_value=0)

        self.algorithm_name = algorithm_name
        self.test_name = test_name
        
        self.classes = set()
        self.successes = 0
        self.failures = 0
        self.total = 0
        
        self.setColumnFormat('Rate',"%0.4f")
        self.setColumnFormat('Lower',"%0.4f")
        self.setColumnFormat('Upper',"%0.4f")
        self.setColumnFormat('Bar',"%-10s")
        
        
    def addData(self, truth, prediction, weight=1):
        """
        Add data to a confusion matrix.  "truth" is the true/correct and 
        "prediction" is the output of the classifier.  Typically you would 
        build this matrix one "test" at a time.  To add multiple test at a 
        time you can use the weight to populate the data more quickly.
        """
        self.classes.add(truth)
        self.classes.add(prediction)
        
        self.accumulateData(truth,prediction,weight)
        if truth == prediction:
            self.successes += weight
        else: 
            self.failures += weight
        self.total += weight
        
        
    def update_rate(self):
        '''Returns a point estimate of the probability of success'''
        return float(self.successes)/float(self.total)
    
    
    def confidenceInterval(self,alpha=0.05):
        '''
        Returns the estimated a confidence interval for the success update_rate by 
        modeling the success update_rate as a binomial distribution.
        '''
        return cibinom(self.total,self.successes,alpha=alpha)


    def computeRates(self,alpha=0.05):
        ''' 
        Populates the distance matrix with more information such as 
        recognition rates for each row. Call this only after all of the 
        data has been added. 
        '''
        self.row_headers.sort()
        self.col_headers.sort()
        
        for row in self.classes:
            successes = 0
            total = 0
            for col in self.classes:
                total += self.element(row,col)
                if row == col:
                    successes += self.element(row,col)
            rate = float(successes)/total
            self.setData(row,'Rate',rate)
            self.setData(row,'Bar',"#"*int(10*rate+0.5))
            self.setData(row,'Lower',cibinom(total,successes,alpha)[0])
            self.setData(row,'Upper',cibinom(total,successes,alpha)[1])
        
        for col in self.classes:
            successes = 0
            total = 0
            for row in self.classes:
                total += self.element(row,col)
                if row == col:
                    successes += self.element(row,col)
            rate = float(successes)/total
            self.setData('Total',col,"%0.4f"%rate)
        
        self.setData('Total','Rate',self.update_rate())
        self.setData('Total','Bar',"#"*int(10*self.update_rate()+0.5))
        self.setData('Total','Lower',self.confidenceInterval(alpha)[0])
        self.setData('Total','Upper',self.confidenceInterval(alpha)[1])
        
        
        
                


                
                
class _TestConfusionMatrix(unittest.TestCase):
    def setUp(self):
        color = ConfusionMatrix()
        color.addData('red','red')
        color.addData('red','red')
        color.addData('red','red')
        color.addData('blue','blue')
        color.addData('blue','blue')
        color.addData('blue','blue')
        color.addData('blue','blue')
        color.addData('pink','pink')
        color.addData('pink','pink')
        color.addData('pink','pink')
        color.addData('pink','pink')
        color.addData('pink','pink')
        color.addData('pink','red')
        color.addData('pink','red')
        color.addData('blue','red')
        color.addData('blue','red')
        color.addData('red','blue')
        color.addData('green','green')
        color.addData('red','green')
        color.computeRates()
        self.color = color
        
        # Simulate a face recognition problem with a
        # probe set of 1000 and a gallery set of 1000
        # 0.001 FAR and 0.100 FRR
        sim_face = ConfusionMatrix()
        sim_face.addData('accept','accept',900)
        sim_face.addData('reject','reject',998001)
        sim_face.addData('accept','reject',100)
        sim_face.addData('reject','accept',999)
        sim_face.computeRates()
        self.sim_face = sim_face
        
    def test_color(self):
        #print
        #print self.color
        self.assertAlmostEquals(self.color.update_rate(),0.6842,places=4)
        self.assertAlmostEquals(self.color.confidenceInterval()[0],0.4345,places=4)
        self.assertAlmostEquals(self.color.confidenceInterval()[1],0.8742,places=4)
    
    def test_verification(self):
        self.assertAlmostEquals(self.sim_face.update_rate(),0.99890100000000004,places=4)
        self.assertAlmostEquals(self.sim_face.confidenceInterval()[0],0.99883409247930877,places=4)
        self.assertAlmostEquals(self.sim_face.confidenceInterval()[1],0.99896499025635421,places=4)
        
        
        
        
        
        
        
        
        
        
        
        