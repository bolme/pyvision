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

    
def detection_test(image, detection_rects, truth_rects):
    '''
    returns (true_positives, true_negitives, false_positives, false_negitives, annotated_image)
    '''
    true_positives = []
    false_negitives = []
    false_positives = []
    for truth in truth_rects:
        truth_rect = None
        
        #print truth
        if isinstance(truth,Rect):
            truth_rect = truth
        else:
            eye1,eye2 = truth
            # Create a bounding rect
            truth_rect = face_from_eyes(eye1, eye2)

        success = False
        for i in range(len(detection_rects)):
            detected_rect = detection_rects[i]
            #print detected_rect
            #print truth_rect
            if is_success(truth_rect,detected_rect):
                success = True
                true_positives.append((truth_rect,detected_rect))
                del detection_rects[i]
                break
        if not success:
            false_negitives.append(truth_rect)
    false_positives = detection_rects
    
    for each in true_positives:
        image.annotateEllipse(each[0],color='green')
        image.annotateRect(each[1],color='green')
    for each in false_positives:
        image.annotateRect(each,color='red')
    for each in false_negitives:
        image.annotateEllipse(each)
    return true_positives, [], false_positives, false_negitives
            
    
    
    
def check_faces(images_file,eyes_file,detections_file):
    from pyvis.geonorm import readEyesFile
    eyes = readEyesFile(eyes_file)
    detections = readDetectionsFile(detections_file)
    image_names = readImagesFile(images_file)
        
    # Normalize the lists
    for each in image_names.keys():
        if not detections.has_key(each):
            detections[each] = []
    
    for each in image_names.keys():
        if not eyes.has_key(each):
            eyes[each] = []
            
    print "Eye images",len(eyes)
    print "Detect images", len(detections)
    
    tp = tn = fp = fn = 0
    for each in image_names.keys():
        print each, image_names[each]
        im = Image(image_names[each])
        #im.annotateRect(Rect(10,10,50,50),color='blue')
        results = detection_test(im,detections[each],eyes[each])
        print "    %d %d %d %d"%(len(results[0]),len(results[1]),len(results[2]),len(results[3]))
        tp += len(results[0])
        tn += len(results[1])
        fp += len(results[2])
        fn += len(results[3])
        
        im.show()
    print "Summary:",tp,tn,fp,fn, 100.0*float(tp)/(tp+fn)
    
def readDetectionsFile(filename):
    from os.path import basename
    f = open(filename)
    
    images = {}
    for line in f:
        line = line.split()
        if len(line) < 5:
            continue
        image = basename(line[0])
        if image[-4:] in ['.gif','.jpg','.tif','.png','.pgm','.ppm']:
            image = image[:-4]

        x = line[1]
        y = line[2]
        w = line[3]
        h = line[4]
        detection = Rect(x,y,w,h)
        if not images.has_key(image):
            images[image] = []
        images[image].append(detection)
    return images

def readImagesFile(filename):
    from os.path import basename, dirname, join
    f = open(filename)
    
    directory = dirname(filename)
    images = {}
    for line in f:
        line = line.split()
        if len(line) < 1:
            continue
        image = basename(line[0])
        fullname = line[0]
        if image[-4:] in ['.gif','.jpg','.tif','.png','.pgm','.ppm']:
            image = image[:-4]

        images[image] = join(directory,fullname)
    return images

def test():
    from pyvis.types.Rect import Rect
    truth = Rect(0,0,10,10)
    detect1 = Rect(1,1,10,10)
    detect2 = Rect(5,5,10,10)
    print "Should print True: ",is_success(truth,detect1)
    print "Should print False:",is_success(truth,detect2)
    
def summarizeDetectionTests(tests): #TODO: Remove
    '''
    Tests is a list containing FaceDetectionTest objects.
    '''
    summary = Table.Table()
    summary.setColumnFormat('Pos Rate','%0.4f')
    summary.setColumnFormat('L 95%','%0.4f')
    summary.setColumnFormat('U 95%','%0.4f')
    summary.setColumnFormat('Neg Rate','%0.4f')
    summary.setColumnFormat('Time','%0.2f')
    for test in tests:
        summary.setElement(test.name,'Pos Rate',test.pos_rate)
        summary.setElement(test.name,'L 95%',test.pos_bounds[0])
        summary.setElement(test.name,'U 95%',test.pos_bounds[1])
        summary.setElement(test.name,'Neg Rate',test.neg_rate)
        summary.setElement(test.name,'Time',test.total_time)
    return summary
    
class FaceDetectionTest: #TODO: Remove
    def __init__(self,name=None,log=None,threshold=0.25):
        self.name = name
        self.log = log
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
                
    def createSummary(self):    
        '''
        TODO: 
        '''
        self.summary_table.setElement('Positive Rate','Value',self.pos_rate)  
        self.summary_table.setElement('Lower Bound','Value',self.pos_bounds[0])  
        self.summary_table.setElement('Upper Bound','Value',self.pos_bounds[1])  
        self.summary_table.setElement('Negative Rate','Value',self.neg_rate)  
        self.summary_table.setElement('Negative Count','Value',self.negatives)  
        self.summary_table.setElement('Image Count','Value',self.images)  
        self.summary_table.setElement('Total Time','Value',self.total_time)  
        self.summary_table.setElement('Time Per Image','Value',self.image_time)
          
    def __str__(self):
        return "FaceDetectionTest(name:%s,PosRate:%f,PosBounds:%s,NegRate:%f,Neg:%d,Im:%d)"%(self.name,self.pos_rate,self.pos_bounds,self.neg_rate,self.negatives,self.images)
        
    
        
    
    