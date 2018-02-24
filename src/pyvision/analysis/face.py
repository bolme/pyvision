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

import os
import pyvision as pv
from pyvision.analysis.FaceAnalysis.FaceDetectionTest import is_success

class EyesFile:
    '''
    Reads and manages the data in an eye coordinate file.
    '''
    
    def __init__(self,filename):
        '''
        Inits and reads in the data.
        '''
        self.filename = filename
        self.images = {}
        
        self._readEyesFile()
    
    def files(self):
        '''
        Returns the list of file names.
        '''
        names = list(self.images.keys())
        names.sort()
        return names
    
    def findFace(self,filename,rect):
        fname = self._parseName(filename)
        if fname in self.images:
            faces = self.images[fname]
            for each in faces:
                truth_rect = each[3]
                if is_success(truth_rect,rect):
                    return each
        return None

    def getFaces(self,filename):
        fname = self._parseName(filename)
        if fname in self.images:
            faces = self.images[fname]
            boxes = []
            for _,_,_,box in faces:
                boxes.append(box)
            return boxes
        return []
       
    def getEyes(self,filename):
        fname = self._parseName(filename)
        if fname in self.images:
            faces = self.images[fname]
            eyes = []
            for _,left,right,_ in faces:
                eyes.append([left,right])
            return eyes
        return []

    def _readEyesFile(self):
        '''
        Private: Do not call directly. Reads the eye file.  
        '''
        if self.filename[-4:] == '.csv':
            f = open(self.filename,'r')
            for line in f:
                #print line,
                line = line.split(',')
                fname = self._parseName(line[0])
                eye1 = pv.Point(float(line[1]),float(line[2]))
                eye2 = pv.Point(float(line[3]),float(line[4]))
                
                truth_rect = pv.BoundingRect(eye1,eye2)
                truth_rect.w = 2.0 * truth_rect.w
                truth_rect.h = truth_rect.w
                truth_rect.x = truth_rect.x - 0.25*truth_rect.w
                truth_rect.y = truth_rect.y - 0.3*truth_rect.w
    
                #print fname,eye1,eye2,truth_rect
                
                if fname not in self.images:
                    self.images[fname] = []
                    
                self.images[fname].append([fname,eye1,eye2,truth_rect])
            
        else:
            f = open(self.filename,'r')
            for line in f:
                #print line,
                line = line.split()
                fname = self._parseName(line[0])
                eye1 = pv.Point(float(line[1]),float(line[2]))
                eye2 = pv.Point(float(line[3]),float(line[4]))
                
                truth_rect = pv.BoundingRect(eye1,eye2)
                truth_rect.w = 2.0 * truth_rect.w
                truth_rect.h = truth_rect.w
                truth_rect.x = truth_rect.x - 0.25*truth_rect.w
                truth_rect.y = truth_rect.y - 0.3*truth_rect.w
    
                #print fname,eye1,eye2,truth_rect
                
                if fname not in self.images:
                    self.images[fname] = []
                    
                self.images[fname].append([fname,eye1,eye2,truth_rect])
            
    def _parseName(self,fname):
        '''
        Private: Do not call directly.  Parses the base filename.
        '''
        fname = os.path.basename(fname)
        fname = os.path.splitext(fname)[0]
        return fname
    
            
class CSU_SRT:
    
    class ImageRecord:
        def __init__(self,filename,subject_id,image_id):
            self.filename = filename
            self.subject_id = subject_id
            self.image_id = image_id
        
    def __init__(self,filename):
        '''Process a Subject Replicate Table file'''
        self.images = []
        self.filenames = {}
        
        f = open(filename,'r')
        
        subject_id = 0
        image_id = 0
        filename = None
        for line in f:
            images = line.split()
            if images:
                for image in images:
                    name = image.split('.')[0]
                    image_id += 1
                    print(name, image_id, subject_id)
                    ir = CSU_SRT.ImageRecord(name,subject_id,image_id)
                    self.images.append(ir)
                    self.filenames[name] = ir
                subject_id += 1
                
        self.total_subjects = subject_id
        self.total_images = image_id
    
    def getNames(self):
        tmp = list(self.filenames.keys())
        tmp.sort()
        return tmp;
    
    def getRecord(self,name):
        if name in self.filenames:
            return self.filenames[name]
        
        return None
                    
        
class CSU_Dist:
    def __init__(self,directory,srt,extention='.sfi'):
        #names = srt.getNames()
        self.matrix = {}
        self.srt = srt
        
        count = 0
        for iname in srt.getNames():
            self.matrix[iname] = {}
            filename = directory+'/'+iname+extention
            print("Reading:",iname)
            f = open(filename,'r')
            for line in f:
                jname,dist = line.split()
                jname = jname.split('.')[0]
                if srt.getRecord(jname):
                    self.matrix[iname][jname] = -float(dist)
                    count += 1
        print("Read:",count)
                    
    
    def getPosNeg(self):
        names = self.srt.getNames()
        pos = []
        neg = []
        for i in range(len(names)):
            for j in range(i+1,len(names)):
                iname = names[i]
                jname = names[j]
                if self.srt.getRecord(iname).subject_id == self.srt.getRecord(jname).subject_id:
                    pos.append(self.matrix[iname][jname])
                else:
                    neg.append(self.matrix[iname][jname])
        return pos,neg
                
   
                
        
if __name__ == "__main__":
    srt = CSU_SRT("/Users/bolme/vision/csuFaceIdBenchmark/imagelists/list640.srt")
    ebgm_dist = CSU_Dist("/Users/bolme/vision/csuFaceIdBenchmark/distances/feret/EBGM",srt)
    pca_dist  = CSU_Dist("/Users/bolme/vision/csuFaceIdBenchmark/distances/feret/PCA_Euclidean",srt)
    ebgm_pos, ebgm_neg = ebgm_dist.getPosNeg()
    pca_pos,  pca_neg  = pca_dist.getPosNeg()
    
    from pyvis.analysis.roc import *
    ebgm_roc = pv.ROC(ebgm_pos,ebgm_neg)
    pca_roc = pv.ROC(pca_pos,pca_neg)
    
    
        
    