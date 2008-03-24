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

from pyvision.analysis.Table import Table
import unittest

from scipy import zeros
from os import listdir
from os.path import join

def MatchScoreMatrixFromCSU(dir_name):
    
    files = listdir(dir_name)
    probes = []
    for each in files:
        if each[-4:] != ".sfi": continue
        filename = join(dir_name,each)
        probes.append(each.split('.')[0])

    
    
    matrix = MatchScoreMatrix(probes,probes)
    
    for each in files:
        if each[-4:] != ".sfi": continue
        filename = join(dir_name,each)
        print "Reading file:", each
        f = open(filename,'r')
        probe = each.split('.')[0]
        for line in f:
            gallery,score = line.split()
            gallery = gallery.split('.')[0]
            score = float(score)
            #print probe,gallery,score
            matrix.addScore(probe,gallery,score)

    return matrix
    

# TODO: Fix this up.
def firstFive(probe,gallary):
    return probe[:5] == gallary[5]


class MatchScoreMatrix(Table):
    '''
    This table stores a typical face recoginition distance matrix as 
    defined by the CSU Face Recogintion Evaluation System.
    '''
    
    def __init__(self,probe_set,gallery_set,algorithm_name=None, data_set=None, distances=True, is_match=firstFive):
        Table.__init__(self)
        self.algorithm_name = algorithm_name
        
        i = 0
        self.probe_set   = {}
        for each in probe_set:
            self.probe_set[each] = i
            i += 1
        probe_size = i
        
        i = 0
        self.gallery_set = {}
        for each in gallery_set:
            self.gallery_set[each] = i
            i += 1
        gallery_size = i
        
        self.distances = True
        self.data = zeros((probe_size,gallery_size),'d')
        
    def setData(self,row,col,value):
        '''Faster Version'''
        i = self.probe_set[row]
        j = self.gallery_set[col]
        #print i,j,self.data.shape
        self.data[i,j] = value
        
        
    def element(self,row,col):
        '''pass'''
        i = self.probe_set[row]
        j = self.gallery_set[col]
        return self.data[i,j]
        
    def addScore(self,probe,gallery,score):
        self.setData(probe,gallery,score)
    
    def getSubMatrix(self,probe_set,gallery_set,**kwargs):
        submatrix = MatchScoreMatrix(probe_set,gallery_set,**kwargs)
        for probe in probe_set:
            for gallery in gallery_set:
                submatrix.addScore(probe,gallery,self.element(probe,gallery))
        return submatrix
        
    def getROC(self):
        '''Returns an ROC curve.  If probe and gallery sets are None the full matrix is used.'''
        
    def getRateAtFAR(self,FAR=0.01,probe_set=None,gallery_set=None):
        '''Returns the recognition rate at a given false accept rate'''
        
    def getRateAtRank(self,rank=0):
        '''Returns the rank one recognition rate.'''
        successes = 0
        total = 0
        for probe in self.probe_set.keys():
            for gallery in self.gallery_set.keys(): pass
            for gallery in self.gallery_set.keys(): pass
        
    def getEER(self,probe_set=None,gallery_set=None):
        '''Returns the Equal Error Rate'''
    
    
class MatchScoreMatrixTest(unittest.TestCase):
    def setUp(self):
        pass
            
    def test__str__(self):
        pass
            
    def test_verification(self):
        pass        
    
from pyvision.analysis.FaceAnalysis.FERET import *

if __name__ == '__main__':
    #TODO: Remove this
    matrix = MatchScoreMatrixFromCSU("/Users/bolme/vision/csuFaceIdEval/distances/feret/PCA_MahCosine")
    dup2_matrix = matrix.getSubMatrix(FERET_DUP2,FERET_GALLERY,data_set="dup2")
    print dup2_matrix
