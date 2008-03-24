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
import time

from pyvision.types.Rect import Rect,BoundingRect
from pyvision.types.Image import Image
from pyvision.analysis import Table
from pyvision.analysis.stats import cibinom

def face_from_eyes(eye1, eye2):
    '''
    Given eye coordinates estimate the face rectangle
    Assumes the face is reasonably horizontal.
    '''
    truth_rect = BoundingRect(eye1, eye2)
    truth_rect.w = 3.0 * truth_rect.w
    truth_rect.h = truth_rect.w
    truth_rect.x = truth_rect.x - 0.33 * truth_rect.w
    truth_rect.y = truth_rect.y - 0.4 * truth_rect.w
    return truth_rect



def is_success(truth_rect, detected_rect, threshhold=0.25):
    '''
    This code takes a truth rect and a detected rect and determines
    if the detection is a success or a failure. The default behavior
    is true if the intersection of the two rects is has 50% of the
    area of the larger rectangle. 
    '''

    if overlap_score(truth_rect, detected_rect) < threshhold:
        return False
    return True

    
def overlap_score(truth_rect, detected_rect):
    '''
    This code takes a truth rect and a detected rect and determines
    if the detection is a success or a failure. The default behavior
    is true if the intersection of the two rects is has 50% of the
    area of the larger rectangle. 
    '''
    same = truth_rect.intersect(detected_rect)
    max_area = max(truth_rect.area(), detected_rect.area())
    if same == None:
        return 0.0
    return same.area()/max_area


#############################################################################
class FaceDetectionTest:
    def __init__(self,name=None,threshold=0.25):
        '''
        Create a face detection test.
        
        INPUTS:
            name      - Label for the test.
            threshold - The fraction of joint area that counts as success.
        '''
        self.name = name
        self.threshold=threshold
        self.sample_id = 1
        
        self.table = Table.Table()
        self.summary_table = Table.Table()
        
        # Cumulative statistic
        self.images = 0
        self.positives = 0
        self.successes = 0
        self.negatives = 0
        self.pixels = 0
        
        # Summary statistics
        self.pos_rate = 0.0
        self.pos_bounds = (0.0,0.0)
        self.neg_rate = 0.0
        
        self.image_time = None
        self.total_time = None
        self.start_time = time.time()
        self.end_time = None
    
    def addSample(self, truth_rects, detected_rects, im=None, annotate=False):
        '''
        Adds a sample to face detection test.
        
        Input:
            truth_rects    - truth for an image.
            detected_rects - output of the detector
            im             - the image or filename to assciate with the sample.
            annotate       - add diagnostic annotations to the images.
        '''
        self.images += 1
        name = None
        detected_rects = copy.copy(detected_rects)
        
        if isinstance(im,Image):
            name = im.filename
            if self.pixels != None:
                self.pixels += im.asPIL().size[0] * im.asPIL().size[1]
        elif isinstance(im,str):
            name = im
            self.pixels = None
        else:
            name = "%d"%self.sample_id
            self.pixels = None
        
        table = self.table
        
        for i in range(len(truth_rects)):
            truth = truth_rects[i]
            self.positives += 1
            success = False
            best_overlap = 0.0
            best_detection = None
            
            for j in range(len(detected_rects)):
                detected = detected_rects[j]
                overlap = overlap_score(truth,detected)
                if overlap >= self.threshold and overlap > best_overlap:
                    success = True
                    best_overlap = overlap
                    best_detection = j
                    
            table.setData(self.sample_id,'id',self.sample_id)
            table.setData(self.sample_id,'name',name)
            table.setData(self.sample_id,'truth_rect',str(truth))
            if best_detection != None:
                table.setData(self.sample_id,'detection_rect',str(detected_rects[best_detection]))
            else:
                table.setData(self.sample_id,'detection_rect',None)
            table.setData(self.sample_id,'success',success)
            table.setData(self.sample_id,'overlap',best_overlap)
            self.sample_id+=1 
            
            if success:
                self.successes += 1
            if annotate and isinstance(im,Image):
                if success:
                    im.annotateEllipse(truth,color='green')
                else:
                    im.annotateEllipse(truth,color='red')
                
                if best_detecion != None:
                    im.annototeRect(detected_rects[best_detection],color='green')


            # Remove the best detection if success
            if best_detection != None:
                del detected_rects[best_detection]
            
        if annotate:
            for each in detected_rects:
                im.annotateRect(each,color='red')
                
        self.negatives += len(detected_rects)
        
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        self.image_time = self.total_time/float(self.images)
        # Update summary statistics
        if self.positives > 0:
            self.pos_rate = float(self.successes)/self.positives
            self.pos_bounds = cibinom(self.positives,self.successes,alpha=0.05)
        if self.pixels != None:
            self.neg_rate = float(self.negatives)/float(1.0e-6*self.pixels)  
            
        self.createSummary() 
                
    def createSummary(self):    
        '''
        Summary of a test as a table.
        '''
        self.summary_table.setElement('PosRate','Value',self.pos_rate)  
        self.summary_table.setElement('Lower95','Value',self.pos_bounds[0])  
        self.summary_table.setElement('Upper95','Value',self.pos_bounds[1])  
        self.summary_table.setElement('NegRate','Value',self.neg_rate)  
        self.summary_table.setElement('NegCount','Value',self.negatives)  
        self.summary_table.setElement('ImageCount','Value',self.images)  
        self.summary_table.setElement('TotalTime','Value',self.total_time)  
        self.summary_table.setElement('TimePerImage','Value',self.image_time)
          
    def __str__(self):
        ''' One line summary of the test '''
        return "FaceDetectionTest(name:%s,PosRate:%f,PosBounds:%s,NegRate:%f,Neg:%d,Im:%d)"%(self.name,self.pos_rate,self.pos_bounds,self.neg_rate,self.negatives,self.images)

        
#############################################################################
def summarizeDetectionTests(tests):
    '''
    Create a summary table for a list containing FaceDetectionTest objects.
    '''
    summary = Table.Table()
    summary.setColumnFormat('PosRate','%0.4f')
    summary.setColumnFormat('Lower95','%0.4f')
    summary.setColumnFormat('Upper95','%0.4f')
    summary.setColumnFormat('NegRate','%0.4f')
    summary.setColumnFormat('Time','%0.2f')
    for test in tests:
        summary.setElement(test.name,'PosRate',test.pos_rate)
        summary.setElement(test.name,'Lower95',test.pos_bounds[0])
        summary.setElement(test.name,'Upper95',test.pos_bounds[1])
        summary.setElement(test.name,'NegRate',test.neg_rate)
        summary.setElement(test.name,'Time',test.total_time)
    return summary
    
#TODO: Unit tests
