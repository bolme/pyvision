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

class ROCAnalysis:
    def __init__(self,positives,negatives):
        self._computeRoc(positives,negatives)
        
    def _computeRoc(self, positive, negative):
        '''
        input: similarity scores for positive examples and negitive examples.
        output: the ROC curve [(fp rate 1,tp rate 1, threshold 1), ... ,(tp rate n,fp rate n, threshold n)] 
        '''
        
        # Create a copy so this method does not mess up the originals
        positive = copy.deepcopy(positive)
        negative = copy.deepcopy(negative)
        
        # Sort the lists from high to low.
        positive.sort()
        positive.reverse()
        negative.sort()
        negative.reverse()
        
        p_index = 0
        n_index = 0
        p_count = len(positive)
        n_count = len(negative)
    
        curve = []
        while p_index < p_count:
            threshold = positive[p_index]
    
            while p_index < p_count and positive[p_index] >= threshold:
                p_index += 1
    
            while n_index < n_count and negative[n_index] >= threshold:
                n_index += 1
            
            tp = float(p_index)/float(p_count)
            fp = float(n_index)/float(n_count)

            curve.append( (fp,tp,threshold) )
            
        self.curve = curve


    def findFalsePositiveRate(self,fpr=0.01):
        i = 1
        while i < len(self.curve) and self.curve[i][0]<fpr:
            i+=1
        return self.curve[i-1]
    
    def findEqualError(self):
        i = 1
        while i < len(self.curve) and (1.0-self.curve[i][1]) > self.curve[i][0]:
            i+=1
        return self.curve[i-1]
    
        

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
    
    
    
    
    
    
    