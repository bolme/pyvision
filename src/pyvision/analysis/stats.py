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

#import scipy as sp
import scipy.stats.distributions as distributions
import numpy as np
import scipy as sp
import copy


def pbinom(q,size,prob):
    '''Binomial probabilites - measured from the left.'''
    dist = distributions.binom(size,prob)
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
    '''Confidence interval for a binomial distribution.'''
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
    Notice that trials where they both succeeded or both failed are ignored.
    Output: The two-sided p-value for McNemar's Exact Test.  For one sided divide by two.
    
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
    
    @param sf: the number of trials algorithm A succeeded and algorithm B failed.
    @param fs: the number of trials algorithm A failed and algorithm B succeeded.

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
    
def pdfWeibull(x,shape,scale):
    pdf = (shape/scale)*((x/scale)**(shape-1))*np.exp(-(x/scale)**shape)
    return pdf

    
def cdfWeibull(x,shape,scale):
    cdf = 1.0-np.exp(-(x/scale)**shape)
    return cdf

    
def fitWeibull(x,ilog=None):
    '''
    Emperically fit a Weibull distribution to x 
    
    @param x: a list containing the x.
    @type x: a list of floats
    @param ilog: an image log to save fit information.
    @type ilog: pv.ImageLog
    @returns: (k,lambda) 
    '''
    x = np.array(x)
    assert x.min() >= 0.0
    
    n = len(x)

    def nll(params):
        ''' Negative log liklyhood'''
        shape,scale = params
        
        mask = x > 0
        pdf = pdfWeibull(x, shape, scale)
        #pdf = (shape/scale)*((x/scale)**(shape-1))*np.exp(-(x/scale)**shape)
        ll = np.log(pdf[mask]).sum()
        t1 = (~mask).sum()
        ll += t1*np.log(0.000001)

        return -ll/len(x)

    tmp = sp.optimize.fmin(nll,[1.0,np.mean(x)],disp=0)
    shape,scale = tmp
    
    if ilog != None:
        # Plot the CDF
        order = x.argsort()
        x = x[order]
        del order
        
        points = [0,0]
        points = [[0,0]] + [ [x[i],float(i)/n] for i in range(n)]
        plot = pv.Plot(title="Weibull CDF")
        plot.points(points)
        
        cdf = cdfWeibull(x, shape, scale)
        
        lines = [[0,0]] + [ [x[i],cdfWeibull(x[i], shape, scale)] for i in range(n)]
        plot.lines(lines)
        ilog(plot,"WeibullCDF")

        plot = pv.Plot(title="Weibull PDF")
        y = pdfWeibull(x, shape, scale)
        points = np.array([x,y]).reshape(2,n).T
        plot.lines(points,color='red',width=3)
        
        hist, bins = np.histogram(x, 5, normed=True)
        
        t1 = 0.5*(bins[:-1] + bins[1:])
        points = np.array([t1,hist]).reshape(2,5).T
        plot.lines(points,color='blue',width=3)
        ilog(plot,"WeibullPDF")
        
        
    return shape,scale

    
def cov(x,y=None):
    '''
    A function that simply computes the covariance: Cov(X,Y).  Data points 
    should be stored in rows.  This function should have similar conventions
    and results as the R function of the same name.
    
    @param x: a matrix containing data.
    @type x: np.array
    @param y: an optional second matrix.  By default y = x.
    @type x: np.array
    '''
    if y == None:
        y = x
    x = copy.deepcopy(x)
    y = copy.deepcopy(y)
    nx,dx = x.shape
    n,dy = y.shape
    
    assert nx == n
    
    mx = x.mean(axis=0).reshape(1,dx)
    my = y.mean(axis=0).reshape(1,dy)
    x -= mx
    y -= my
    s = 1.0/(n-1)
    return s*np.dot(x.T,y)


def cor(x,y=None):
    '''
    A function that simply computes the correlation matrix: Corr(X,Y).  Data points 
    should be stored in rows.  This function should have similar conventions
    and results as the R function of the same name.
    
    @param x: a matrix containing data.
    @type x: np.array
    @param y: an optional second matrix.  By default y = x.
    @type x: np.array
    '''
    if y == None:
        y = x
    x = copy.deepcopy(x)
    y = copy.deepcopy(y)
    nx,dx = x.shape
    n,dy = y.shape
    
    assert nx == n
    
    mx = x.mean(axis=0).reshape(1,dx)
    my = y.mean(axis=0).reshape(1,dy)
    sx = x.std(axis=0,ddof=1).reshape(1,dx)
    sy = y.std(axis=0,ddof=1).reshape(1,dy)
    x = (x-mx)/sx
    y = (y-my)/sy
    s = 1.0/(n-1)
    return s*np.dot(x.T,y)


def cov2cor(v):
    '''
    Converts a symmetric positive definite matrix to a correlation matrix by 
    normalizing by the diagonal.
    '''
    r,c = v.shape
    assert r == c
    s = 1.0/np.sqrt(np.diag(v))
    v *= s.reshape(r,1)
    v *= s.reshape(1,c)
    return v

class SummaryStats:
    
    def __init__(self, x, name="NO_NAME", alpha=0.05):
        '''
        Computes some basic information about the data set x.
        
        Assumes x comes from a t distribution.
        '''

        self.name = name
        
        x = np.array(x,'d')
                
        self.alpha  = alpha
        self.n      = len(x)
        self.mean   = x.mean()
        self.var    = ((x-self.mean)*(x-self.mean)).sum()/(self.n-1)
        self.std    = np.sqrt(self.var)
        self.ste    = self.std/np.sqrt(self.n)
        self.df     = self.n - 1
        
        tci = distributions.t.ppf(1-alpha/2,self.df)
        lcim = self.mean-tci*self.ste
        ucim = self.mean+tci*self.ste
        self.mean_ci = [lcim,ucim]
    
        tci = distributions.t.ppf(1-alpha/2,self.df)
        lci = self.mean-tci*self.std
        uci = self.mean+tci*self.std
        self.ci = [lci,uci]
    
        self.median = np.median(x)
        
                
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
    
            

import unittest
import pyvision as pv
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
        # data from R
        self.longley = [[83.0,  234.289,      235.6,        159.0,    107.608, 1947,   60.323,],
                        [88.5,  259.426,      232.5,        145.6,    108.632, 1948,   61.122,],
                        [88.2,  258.054,      368.2,        161.6,    109.773, 1949,   60.171,],
                        [89.5,  284.599,      335.1,        165.0,    110.929, 1950,   61.187,],
                        [96.2,  328.975,      209.9,        309.9,    112.075, 1951,   63.221,],
                        [98.1,  346.999,      193.2,        359.4,    113.270, 1952,   63.639,],
                        [99.0,  365.385,      187.0,        354.7,    115.094, 1953,   64.989,],
                        [100.0, 363.112,      357.8,        335.0,    116.219, 1954,   63.761,],
                        [101.2, 397.469,      290.4,        304.8,    117.388, 1955,   66.019,],
                        [104.6, 419.180,      282.2,        285.7,    118.734, 1956,   67.857,],
                        [108.4, 442.769,      293.6,        279.8,    120.445, 1957,   68.169,],
                        [110.8, 444.546,      468.1,        263.7,    121.950, 1958,   66.513,],
                        [112.6, 482.704,      381.3,        255.2,    123.366, 1959,   68.655,],
                        [114.2, 502.601,      393.1,        251.4,    125.368, 1960,   69.564,],
                        [115.7, 518.173,      480.6,        257.2,    127.852, 1961,   69.331,],
                        [116.9, 554.894,      400.7,        282.7,    130.081, 1962,   70.551,],]
        
        self.longley = np.array(self.longley)

        
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
        
    def test_fitWeibull(self):
        ilog = None
        #ilog = pv.ImageLog()
        
        # data genereated in R with shape=1 and scale=1 
        # from fitdistr
        #      shape       scale  
        #      1.1082557   1.0212356 
        #      (0.1358495) (0.1535614)
        data = [0.39764089, 0.60824086, 0.40285732, 1.54531775, 1.73364323, 1.23747338, 1.12446222, 3.15989785, 0.22271289,
                1.28213280, 1.68005746, 0.58658749, 0.83938237, 1.25577118, 0.64729513, 1.92565971, 0.36610902, 0.10363669,
                0.15618127, 0.02262031, 0.25985175, 0.14230431, 1.54069502, 1.06272791, 0.05364079, 0.93874689, 1.01770360,
                0.40204781, 0.40660520, 0.12017453, 0.73480365, 3.73042281, 1.37838373, 0.17739429, 1.21166837, 3.79022634,
                0.91822186, 1.07417484, 0.37926781, 0.66128749,]
        shape,scale = fitWeibull(data,ilog = ilog)
        self.assertAlmostEqual(shape,1.1082557,places=4)
        self.assertAlmostEqual(scale,1.0212356,places=4)
        
        # data genereated in R with shape=2 and scale=5 
        # from fitdistr
        #     shape       scale  
        #     1.8456678   5.3412324 
        #     (0.2288101) (0.4831310)
        data = [9.0668007,  7.5244193,  1.3643692,  2.4980839,  3.8229886,  0.7847899, 10.2635502,  6.4853731,  4.1691479,
                4.7222325,  3.6751391, 10.5038682,  1.8489645,  5.5697636, 10.3385587,  1.8399665,  7.8512893,  1.6301032,
                7.1892784,  3.4151212,  2.1018280,  3.0128155,  5.4290304,  3.9759659,  6.4867134,  4.8687895,  1.2671571,
                6.4746843,  3.6922549,  3.6133898,  5.8451979,  5.5435995,  4.2617657,  3.3490959,  6.3412869,  1.3440581,
                2.7830355,  2.1482365,  2.5091446,  9.5137472]
        shape,scale = fitWeibull(data,ilog = ilog)
        self.assertAlmostEqual(shape,1.8456678,places=4)
        self.assertAlmostEqual(scale,5.3412324,places=4)
        
        # data genereated in R with shape=0.5 and scale=0.2
        # from fitdistr
        #    shape        scale   
        #    0.51119109   0.15523840 
        #    (0.06552176) (0.05033735)
        data = [4.368635e-02, 1.716870e-01, 5.265532e-01, 2.387941e-04, 1.836984e-01, 4.835876e-01, 2.159292e-03, 1.060331e+00,
                1.945628e-02, 4.110887e-01, 1.257612e-01, 2.911412e-02, 6.198067e-01, 5.143289e-01, 1.047416e-01, 3.997763e-01,
                4.596470e-07, 7.417249e-03, 5.209768e-03, 4.370919e-03, 3.816381e-01, 8.640891e-03, 4.125977e-02, 2.129932e-02,
                6.916213e-03, 1.037448e-01, 1.946721e-02, 1.445826e-01, 9.911569e-01, 2.074493e-01, 2.726630e-03, 3.030224e-02,
                1.991381e+00, 1.616899e-01, 1.251923e+00, 4.915620e-01, 1.826906e-01, 8.091978e-04, 7.905816e-03, 5.381265e-02]
        shape,scale = fitWeibull(data,ilog=ilog)
        self.assertAlmostEqual(shape,0.51119109,places=4)
        self.assertAlmostEqual(scale,0.15523840,places=4)
        
        if ilog != None:
            ilog.show()
            
    def test_cov(self):
        t = cov(self.longley)
        
    def test_cor2cov(self):
        v = cov(self.longley)
        cov2cor(v)
        
    def test_cor(self):
        print cor(self.longley)

        
        
        
        
        
        
        