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

class FaceRecognizer:
    ''' Base class for a face recognition algorithm.  Output is a similarity score. '''
    def __init__(self):
        self.training_data=[]

    def getTrainingMatches(self):
        '''
        Returns a list of all pairs of images in the training set that 
        are of the same person.
        '''
        matches = []
        for i in range(len(self.training_data)):
            for j in range(i+1,len(self.training_data)):
                if i == j:
                    continue
                if self.training_data[i][3] == self.training_data[j][3]:
                    matches.append([self.training_data[i],self.training_data[j]])
        return matches
                
    
    def getTrainingNonMatches(self):
        '''
        Returns a list of all pairs in the training images that are of
        different people.
        '''
        nonmatches = []
        for i in range(len(self.training_data)):
            for j in range(i+1,len(self.training_data)):
                if i == j:
                    continue
                if self.training_data[i][3] != self.training_data[j][3]:
                    nonmatches.append([self.training_data[i],self.training_data[j]])
        return nonmatches
    
    def addTraining(self,img,rect=None,eyes=None,id=None):
        '''Adds a training face for the algorithm.'''
        self.training_data.append([img,rect,eyes,id])
        

    def distance(self, fr1, fr2):
        '''Compute the similarity of two faces'''
        raise NotImplementedError()
    
    
    def computeFaceRecord(self,im,rect=None,eyes=None):
        '''
        Given an image and face location, compute a face record.
        
        @param im: image containing the face
        @param rect: specifies the location of the face in the image, and is 
               typically defined as a detection rectangle or eye coordinates.
               
        @returns: data that represents the identity of the face, such as
                 eigen coeffecients for PCA.
        '''

        raise NotImplementedError()
    
    
        