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

from scipy.stats import distributions
from scipy.stats.stats import median
from scipy.optimize import fsolve
from numpy import array
from math import sqrt

from scipy.stats import binom

def pbinom(q,size,prob):
    '''Binomial probabilites - measured from the left.'''
    dist = binom(size,prob)
    return dist.cdf(q)
    
    
def qbinom(p,size,prob):
    '''Binomial quantiles'''
    # This seems to be broken in scipy so it is determined from pbinom here.
    # preform a binary search to find the correct quantile.
    minq = 0
    maxq = size
    minp = pbinom(minq,size,prob)
    maxp = pbinom(maxq,size,prob)
    
    if minp > p: return minq
    if maxp < p: return maxq
    
    while(maxq > minq+1):
        newq = (minq+maxq)/2
        newp = pbinom(newq,size,prob)
        #print "p's and q's:",minq,minp,newq,newp,maxq,maxp
        if p > newp:
            minp = newp
            minq = newq
        else:
            maxp = newp
            maxq = newq
    
    return maxq

def cibinom(size,success,alpha=0.05):

    goal = 0.5*alpha
        
    # find the upper limit
    lower_prob = 0.0
    lower_p    = pbinom(success,size,lower_prob)
    upper_prob = 1.0
    upper_p    = pbinom(success,size,upper_prob)
        
    for i in range(32):
        new_prob = (lower_prob + upper_prob)*0.5
        new_p    = pbinom(success,size,new_prob)
        
        if new_p < goal:
            upper_prob = new_prob
            upper_p = new_p
        else:
            lower_prob = new_prob
            lower_p = new_p
            
    ub = upper_prob
       
       
       
    #find the lower limit
    success = size - success
    lower_prob = 0.0
    lower_p    = pbinom(success,size,lower_prob)
    upper_prob = 1.0
    upper_p    = pbinom(success,size,upper_prob)
        
    for i in range(64):
        new_prob = (lower_prob + upper_prob)*0.5
        new_p    = pbinom(success,size,new_prob)
        
        if new_p < goal:
            upper_prob = new_prob
            upper_p = new_p
        else:
            lower_prob = new_prob
            lower_p = new_p
            
    lb = 1-upper_prob
    
    return (lb,ub)
    
         
def mcnemar_test(sf,fs):
    ''' 
    From Zhoo and Chellappa. "Face Processining". Chapter 3.
    
    Compairs the output of two algorithms on the same set of trials.
    Input to the function are two counts:
        sf - the number of trials algorithm A succeeded and algorithm B failed.
        fs - the number of trials algorithm A failed and algorithm B succeeded.
    Notice that trials where they both succeeded or both failed are ignored.
    Output: The two-sided p-value for McNemar's Exact Test.  For one sided
            divide by two.
    
    If sf+fs is large you may want to use the approximate test.
    
    Here is an example on a simple classifer...  The problem is classifing 
    images of bananas, apples, and oranges.  Two algorithms are compaired by 
    running the algorithms on the same set of 9 test images.  Here are the
    outcomes.
    
    |---|--------|--------|--------|-----------|-----------|
    |   | truth  | Alg A  | Alg B  | Success A | Success B |
    |---|--------|--------|--------|-----------|-----------|
    | 1 | banana | banana | banana | T         | T         |
    | 2 | apple  | apple  | banana | T         | F         |
    | 3 | orange | apple  | orange | F         | T         |
    | 4 | orange | apple  | apple  | F         | F         |
    | 5 | apple  | apple  | apple  | T         | T         |
    | 6 | banana | banana | banana | T         | T         |
    | 7 | apple  | apple  | banana | T         | F         |
    | 8 | orange | orange | apple  | T         | F         |
    | 9 | banana | None   | banana | T         | T         |
    |---|--------|--------|--------|-----------|-----------|
    
    Now you can count the number of times both algorithms succeed, both 
    algorithms fail, A succeeds and B fails, and A fails and B succeeds.
    
    |-------|-----|-----|-------|
    |       | A=T | A=F | Total |
    |-------|-----|-----|-------|
    | B=T   |   4 |   1 |     5 |
    | B=F   |   3 |   1 |     4 |
    |-------|-----|-----|-------|
    | Total |   7 |   2 |     9 |
    |-------|-----|-----|-------|
    
    From this table you can compute success rates (A=T)/Total...

    > 7.0/9.0 # A Success Rate
    0.77777777777777779
    
    > 5.0/9.0 # B Success Rate
    0.55555555555555558    
    
    The input to McNemar's Test are the SF (A=T,B=F) = 3 and FS (A=F,B=T) = 1.
    
    > mcnemar_test(3,1) # Two-sided p-value
    0.625
    '''
    def factorial(n):
        if n <= 1: return 1
        else: return n*factorial(n-1)
           
    low = min(sf,fs)
    high = max(sf,fs)
    n = low + high
    
    pvalue = 0.0
    for i in range(0,low+1):
        pvalue += (0.5**n)*factorial(n)/float(factorial(i)*factorial(n-i))
            
    for i in range(high,n+1):
        pvalue += (0.5**n)*factorial(n)/float(factorial(i)*factorial(n-i))
    
    return pvalue
    

class SummaryStats:
    
    def __init__(self, x, name="NO_NAME", alpha=0.05):
        '''
        Computes some basic information about the data set x.
        
        Assumes x comes from a t distribution.
        '''

        self.name = name
        
        x = array(x,'d')
                
        self.alpha  = alpha
        self.n      = len(x)
        self.mean   = x.mean()
        self.var    = ((x-self.mean)*(x-self.mean)).sum()/(self.n-1)
        self.std    = sqrt(self.var)
        self.ste    = self.std/sqrt(self.n)
        self.df     = self.n - 1
        
        tci = distributions.t.ppf(1-alpha/2,self.df)
        lcim = self.mean-tci*self.ste
        ucim = self.mean+tci*self.ste
        self.mean_ci = [lcim,ucim]
    
        tci = distributions.t.ppf(1-alpha/2,self.df)
        lci = self.mean-tci*self.std
        uci = self.mean+tci*self.std
        self.ci = [lci,uci]
    
        self.median = median(x)
        
                
    def __str__(self):
        data = self.asTable()
        title = "Summary of %s"%self.name
        
        title_length = len(title)
        key_length = 0
        value_length = 0
        
        formatted_data = []

        for key,value in data:
            key_length = max(key_length,len(key))
            
            if isinstance(value,int):
                value = "%d      "%value
            elif isinstance(value,str):
                value = "%s      "%value
            elif isinstance(value,float):
                if value < 0.0001 and value > 0.0:
                    value = "< 0.00001"
                else:
                    value = "%0.5f"%value
            else:
                raise TypeError("Only int, float, and str supported")
            
            value_length = max(value_length,len(value))
            formatted_data.append([key,value])
            
        row_format = "| %%-%ds | %%%ds |"%(key_length,value_length)
        row_length = 7+key_length+value_length

        rows = []

        rows.append(title)
        rows.append('-'*row_length)
        for key,value in formatted_data:
            rows.append(row_format%(key,value))
        rows.append('-'*row_length)
        
        result = ""
        for each in rows:
            result += each
            result += '\n'
                        
        return result
        
        
        
    def asTable(self):
        data = []
        data.append(['N',self.n])
        data.append(['Mean',self.mean])
        if self.alpha == 0.01:
            data.append(['Mean 99% CI LB',self.mean_ci[0]])
            data.append(['Mean 99% CI UB',self.mean_ci[1]])
        elif self.alpha == 0.05:
            data.append(['Mean 95% CI LB',self.mean_ci[0]])
            data.append(['Mean 95% CI UB',self.mean_ci[1]])
        elif self.alpha == 0.1:
            data.append(['Mean 90% CI LB',self.mean_ci[0]])
            data.append(['Mean 90% CI UB',self.mean_ci[1]])
        elif self.alpha == 0.25:
            data.append(['Mean 75% CI LB',self.mean_ci[0]])
            data.append(['Mean 75% CI UB',self.mean_ci[1]])
        else:
            data.append(['Mean %0.4f CI LB'%(1.-self.alpha,),self.mean_ci[0]])
            data.append(['Mean %0.4f CI UB'%(1.-self.alpha,),self.mean_ci[1]])
        data.append(['Var',self.var])
        data.append(['Std Dev',self.std])
        data.append(['Std Error',self.ste])
        if self.alpha == 0.01:
            data.append(['99% CI LB',self.ci[0]])
            data.append(['99% CI UB',self.ci[1]])
        elif self.alpha == 0.05:
            data.append(['95% CI LB',self.ci[0]])
            data.append(['95% CI UB',self.ci[1]])
        elif self.alpha == 0.1:
            data.append(['90% CI LB',self.ci[0]])
            data.append(['90% CI UB',self.ci[1]])
        elif self.alpha == 0.25:
            data.append(['75% CI LB',self.ci[0]])
            data.append(['75% CI UB',self.ci[1]])
        else:
            data.append(['%0.4f CI LB'%(1.-self.alpha,),self.ci[0]])
            data.append(['%0.4f CI UB'%(1.-self.alpha,),self.ci[1]])
        #data.append(['DF',self.df])
        data.append(['Median',self.median])
        
        return data
    

def rsquared():
    #Normally: SSModel/SSTotal
    #ZeroInt (sum(y_hat^2)/sum(y^2))
    pass
    # TODO: add sum of squared.
            

import unittest
class _TestStats(unittest.TestCase):
    def setUp(self):
        self.normal_data = [1.3139,  5.2441,  0.0756,  4.4679,  2.3845,  
                            2.9330,  2.9803,  2.3844,  0.7643,  -2.2058,  
                            1.9057,  -0.1609,  4.4459,  -0.0158,  5.9733,  
                            2.8994,  0.2282,  1.0099,  2.8802,  2.3120,  
                            1.8388,  -2.1818,  -0.3264,  -0.0711,  4.8463,  
                            0.6059,  6.1048,  1.7795,  1.2986,  5.4349,  
                            2.2219,  3.0162,  -1.6250,  2.8928,  -6.7314,  
                            2.5222,  2.2261,  3.3774,  2.7479,  2.7690,  
                            4.6934,  3.0834,  8.9465,  5.5662,  5.1551,  
                            -1.6149,  -1.2087,  1.8739,  7.6589,  4.9503]
        
    def test_summarystats(self):
        # Verified with R and SAS
        stats = SummaryStats(self.normal_data, name="Normal Data")
        
        self.assertEquals(stats.n,50)
        self.assertAlmostEqual(stats.mean,       2.273416, places=6) #R
        self.assertAlmostEqual(stats.var,        7.739672, places=6) #R
        self.assertAlmostEqual(stats.std,        2.782027, places=6) #R
        self.assertAlmostEqual(stats.ste,        0.393438, places=6) #SAS
        self.assertAlmostEqual(stats.ci[0],     -3.317276, places=6) # TODO: verify with t distribution
        self.assertAlmostEqual(stats.ci[1],      7.864108, places=6) # TODO: verify with t distribution
        self.assertAlmostEqual(stats.mean_ci[0], 1.482773, places=6) #R
        self.assertAlmostEqual(stats.mean_ci[1], 3.064059, places=6) #R
        self.assertAlmostEqual(stats.median,     2.38445, places=5) #R
        
        #                                       The MEANS Procedure
        #
        #                                      Analysis Variable : x
        #
        #                   Lower 95%       Upper 95%
        #        Mean     CL for Mean     CL for Mean         Minimum         Maximum          Median     N
        #   2.2734160       1.4827728       3.0640592      -6.7314000       8.9465000       2.3844500    50
        #
        #                                      Analysis Variable : x
        #
        #                                Std Dev        Variance       Std Error
        #                              2.7820266       7.7396721       0.3934380

        
    def test_pbinom(self):
        # probabilities verified with R
        self.assertAlmostEquals(pbinom(50,100,0.5),   0.5397946)
        self.assertAlmostEquals(pbinom(25,100,0.3),   0.1631301)
        self.assertAlmostEquals(pbinom(20,100,0.1),   0.9991924)
        self.assertAlmostEquals(pbinom(8,100,0.05),   0.9369104)
        self.assertAlmostEquals(pbinom(0,100,0.01),   0.3660323)
        self.assertAlmostEquals(pbinom(100,100,0.98), 1.0000000)
        
    def test_qbinom(self):
        # quantiles verified with R
        self.assertEquals(qbinom(0.5,100,0.5),  50)
        self.assertEquals(qbinom(0.1,100,0.01),  0)
        self.assertEquals(qbinom(1.0,100,0.9), 100)
        self.assertEquals(qbinom(0.2,100,0.4),  36)
        self.assertEquals(qbinom(0.4,100,0.85), 84)

    def test_cibinom(self):
        # Intervals verified by: http://statpages.org/confint.html
        self.assertAlmostEqual( cibinom(100,50)[0], 0.3983, places=4)
        self.assertAlmostEqual( cibinom(100,50)[1], 0.6017, places=4)
        self.assertAlmostEqual( cibinom(1000,50)[0], 0.0373, places=4)
        self.assertAlmostEqual( cibinom(1000,50)[1], 0.0654, places=4)
        self.assertAlmostEqual( cibinom(10000,50)[0], 0.0037, places=4)
        self.assertAlmostEqual( cibinom(10000,50)[1], 0.0066, places=4)
        self.assertAlmostEqual( cibinom(1,1)[0], 0.0250, places=4)
        self.assertAlmostEqual( cibinom(1,1)[1], 1.0000, places=4)
        self.assertAlmostEqual( cibinom(10,10)[0], 0.6915, places=4)
        self.assertAlmostEqual( cibinom(10,10)[1], 1.0000, places=4)
        self.assertAlmostEqual( cibinom(100,100)[0], 0.9638, places=4)
        self.assertAlmostEqual( cibinom(100,100)[1], 1.0000, places=4)
        self.assertAlmostEqual( cibinom(1,0)[0], 0.0000, places=4)
        self.assertAlmostEqual( cibinom(1,0)[1], 0.9750, places=4)
        self.assertAlmostEqual( cibinom(10,0)[0], 0.0000, places=4)
        self.assertAlmostEqual( cibinom(10,0)[1], 0.3085, places=4)
        self.assertAlmostEqual( cibinom(100,0)[0], 0.0000, places=4)
        self.assertAlmostEqual( cibinom(100,0)[1], 0.0362, places=4)
        
    def test_mcnemar(self):
        # From Zhoo and Chellappa. "Face Processining". Chapter 3.
        self.assertAlmostEqual(mcnemar_test(16,24),0.268,places=3)
        self.assertAlmostEqual(mcnemar_test(64,96),0.014,places=3)
        

        
        
        
        