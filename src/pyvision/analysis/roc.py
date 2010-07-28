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

import copy
import numpy as np
import pyvision as pv
    
def buildPositiveNegativeLists(names,matrix,class_equal):
    positive = []
    negative = []
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            if class_equal(names[i],names[j]):
                positive.append(matrix[i][j])
            else:
                negative.append(matrix[i][j])
    return positive,negative
    
def readCsuDistanceMatrix(directory):
    from os import listdir
    from os.path import join
    
    filenames = []
    for each in listdir(directory):
        if each[-4:] == ".sfi":
            filenames.append(each)
        
    filenames.sort()
    matrix = []
    for filename in filenames:
        f = open(join(directory,filename),'r')
        row = []
        count = 0
        for line in f:
            fname,sim = line.split()
            assert fname == filenames[count]
            sim = -float(sim)
            row.append(sim)
            count += 1
        f.close()
        assert len(row) == len(filenames)
        matrix.append(row)
    assert len(matrix) == len(filenames)
        
    return filenames,matrix

class ROCPoint:
    def __init__(self,nscore,nidx,n,far,mscore,midx,m,frr):
        self.nscore,self.nidx,self.n,self.far,self.mscore,self.midx,self.m,self.frr = nscore,nidx,n,far,mscore,midx,m,frr
        self.tar = 1.0 - self.frr
        self.trr = 1.0 - self.far
    
    def __str__(self):
        return "ROCPoint %f FRR at %f FAR"%(self.frr,self.far)

ROC_LOG_SAMPLED = 1
ROC_MATCH_SAMPLED = 2
ROC_PRECISE_SAMPLED = 2

class ROC:
    
    def __init__(self,match,nonmatch,is_distance=True):
        self.match = np.array(match).copy()
        self.nonmatch = np.array(nonmatch).copy()
        self.is_distance = is_distance
        
        if not is_distance:
            self.match    = -self.match
            self.nonmatch = -self.nonmatch
        
        
        self.match.sort()
        self.nonmatch.sort()
        
        
    def getCurve(self,method=ROC_LOG_SAMPLED):
        """
        returns header,rows
        """
        header = ["score","frr","far","trr","tar"]
        rows = []
        
        if method == ROC_LOG_SAMPLED:
            for far in 10**np.arange(-6,0.0000001,0.01):
                point = self.getFAR(far)
                
                row = [point.nscore,point.frr,point.far,point.trr,point.tar]
                rows.append(row)
        if method == ROC_MATCH_SAMPLED:
            for score in self.match:
                if self.is_distance:
                    point = self.getMatch(score)
                else:
                    point = self.getMatch(-score)
                row = [point.nscore,point.frr,point.far,point.trr,point.tar]
                rows.append(row)
            
        return header,rows
        
            
    def getFAR(self,far):
        match = self.match
        nonmatch = self.nonmatch
        orig_far = far
        
        m = len(match)
        n = len(nonmatch)
        
        nidx = int(round(far*n))
        far = float(nidx)/n
        if nidx >= len(nonmatch):
            nscore = None
        #elif nidx == 0:
        #    nscore = nonmatch[nidx]
        else:
            nscore = nonmatch[nidx]
        
        if nscore != None:
            midx = np.searchsorted(match,nscore,side='left')    
        else:
            midx = m
             
        frr = 1.0-float(midx)/m
        if midx >= len(match):
            mscore = None
        else:
            mscore = match[midx]

        #assert mscore == None or mscore <= nscore
        
        #if nidx == 0:
        #print "Zero:",orig_far,nscore,nidx,n,far,mscore,midx,m,frr
        #print nonmatch
        #print match
        if self.is_distance:
            return ROCPoint(nscore,nidx,n,far,mscore,midx,m,frr)
        else:
            if nscore != None:
                nscore = -nscore
            if mscore != None:
                mscore = -mscore
            return ROCPoint(nscore,nidx,n,far,mscore,midx,m,frr)

    def getFRR(self,frr):
        match = self.match
        nonmatch = self.nonmatch
        
        m = len(match)
        n = len(nonmatch)
        
        midx = int(round((1.0-frr)*m))
        frr = 1.0 - float(midx)/m
        if midx >= len(match):
            mscore = None
        else:
            mscore = match[midx-1]
    
        nidx = np.searchsorted(nonmatch,mscore)
        far = float(nidx)/n
        if nidx-1 < 0:
            nscore = None
        else:
            nscore = nonmatch[nidx-1]
                
        assert nscore == None or mscore >= nscore
        
        if self.is_distance:
            return ROCPoint(nscore,nidx,n,far,mscore,midx,m,frr)
        else:
            if nscore != None:
                nscore = -nscore
            if mscore != None:
                mscore = -mscore
            return ROCPoint(nscore,nidx,n,far,mscore,midx,m,frr)

                
    def getMatch(self,mscore):
        if not self.is_distance:
            mscore = -mscore
        match = self.match
        nonmatch = self.nonmatch
        
        m = len(match)
        n = len(nonmatch)
        
        midx = np.searchsorted(match,mscore)
        #midx = int(round((1.0-frr)*m))
        frr = 1.0 - float(midx)/m
    
        nidx = np.searchsorted(nonmatch,mscore)
        far = float(nidx)/n
        if nidx-1 < 0:
            nscore = None
        else:
            nscore = nonmatch[nidx-1]
                
        assert nscore == None or mscore >= nscore
        
        if self.is_distance:
            return ROCPoint(nscore,nidx,n,far,mscore,midx,m,frr)
        else:
            if nscore != None:
                nscore = -nscore
            if mscore != None:
                mscore = -mscore
            return ROCPoint(nscore,nidx,n,far,mscore,midx,m,frr)
                
                
    def results(self):
        table = pv.Table()
        pt = self.getFAR(0.001)
        #print pt
        table[0,'FAR'] = 0.001
        table[0,'TAR'] = pt.tar
        pt = self.getFAR(0.01)
        #print pt
        table[1,'FAR'] = 0.01
        table[1,'TAR'] = pt.tar
        pt = self.getFAR(0.1)
        #print pt
        table[2,'FAR'] = 0.1
        table[2,'TAR'] = pt.tar
        
        return table

    
        

if __name__ == '__main__':
    filenames,matrix = readCsuDistanceMatrix("/Users/bolme/vision/data/celeb_db/distances/PCA_EU")
    print len(filenames),len(matrix),len(matrix[0])
    positive,negative = buildPositiveNegativeLists(filenames,matrix,lambda x,y: x[:10] == y[:10])
    
    print len(positive), len(negative)    
    curve = computeRoc(positive,negative)
    
    positive.sort()
    negative.sort()
    tmp1 = positive[::len(positive)/100]
    tmp2 = negative[::len(negative)/100]
    
    writeRocFile(curve,"/Users/bolme/pca_eu.txt")
    #print curve   
    
    
    
    
    
    
    