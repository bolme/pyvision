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

from os.path import join
import unittest

import pyvision
from pyvision.types.Image import Image
from pyvision.face.CascadeDetector import CascadeDetector
from pyvision.analysis.face import EyesFile
from pyvision.types.Point import Point
from pyvision.types.Rect import Rect
from pyvision.point.PointLocator import SVMLocator
from pyvision.types.Affine import *
from pyvision.analysis.FaceAnalysis.FaceDetectionTest import face_from_eyes, is_success

class FaceFinder:
    ''' 
    This class detects faces and then returns the detection rectangles and 
    the eye coordinates.
    '''
    
    #TODO: the left eye points and right eye points should be learned from the input data
    def __init__(self, face_detector=CascadeDetector(), tile_size=(128,128), validate=None):
        ''' 
        Create an eye locator.  This default implentation uses a 
        cascade classifier for face detection and then SVR for eye
        location. 
        '''
        self.face_detector = face_detector
        self.left_eye      = None
        self.right_eye     = None
        self.tile_size     = tile_size
        self.validate      = validate

        # point locators that learn to find the eyes.
        self.left_locator  = SVMLocator()
        self.right_locator = SVMLocator()
        
        # Number of training images where the face detection did not work.
        self.detection_failures = 0

        
    def addTraining(self, left_eye, right_eye, im):
        '''Train an eye detector givin a full image and the eye coordinates.'''
        
        # determine the face rect
        true_rect = face_from_eyes(left_eye,right_eye)
        
        # run the face detector
        rects = self.face_detector.detect(im)
        
        # find the best detection if there is one
        for pred_rect in rects:
            if is_success(pred_rect,true_rect):
                # Transform the face
                affine = AffineFromRect(pred_rect,self.tile_size)
                cropped = affine.transformImage(im)
                
                # Mark the eyes
                leye = affine.transformPoint(left_eye)
                reye = affine.transformPoint(right_eye)

                # Add training data to locators
                self.left_locator.addTraining(cropped,leye)
                self.right_locator.addTraining(cropped,reye)
                
                # Just use the first success
                return
            
        # The face was not detected
        self.detection_failures += 1
        
    def train(self):
        '''
        Train the eye locators.
        '''
        self.left_locator.train()
        self.right_locator.train()        
        
        self.left_eye      = self.left_locator.mean
        self.right_eye     = self.right_locator.mean

        
    def detect(self, im, n_iter=3, annotate=False):
        '''
        Returns a list of tuples where each tuple contains:
            (registered_image, detection_rect, left_eye, right_eye) 
        '''
        result = []
        
        rects = self.face_detector.detect(im)
        
        # Anotate Faces
        for rect in rects:
            
            # Transform the face
            affine = AffineFromRect(rect,self.tile_size)
            cropped = affine.transformImage(im)
            
            for p in range(n_iter):
                # Find the eyes            
                pleye = self.left_locator.predict(cropped)
                preye = self.right_locator.predict(cropped)
                
                pleye = affine.invertPoint(pleye)
                preye = affine.invertPoint(preye)
                
                # Seccond Pass
                affine = AffineFromPoints(pleye,preye,self.left_eye,self.right_eye,self.tile_size)
                cropped = affine.transformImage(im)
            
            #affine = AffineFromPoints(pleye,preye,self.left_eye,self.right_eye,self.tile_size)
            #reg = affine.transformImage(im)
            reg = cropped

            if self.validate != None and not self.validate(reg):
                # Validate the face.
                if annotate:
                    im.annotateRect(rect,color='red')        
                    im.annotatePoint(pleye,color='red')
                    im.annotatePoint(preye,color='red')
                continue
            
            if annotate:
                reg.annotatePoint(self.left_eye,color='green')
                reg.annotatePoint(self.right_eye,color='green')
                im.annotatePoint(pleye,color='green')
                im.annotatePoint(preye,color='green')
                im.annotateRect(rect,color='green')        
            result.append((reg,rect,pleye,preye))
            
        return result


def FaceFinderFromDatabase(eyes_file, image_dir, training_set = None, training_size=1000, image_ext='.jpg', **kwargs):
    '''
    Train a face finder using an Eye Coordanates file and a face dataset.
    '''

    # Create a face finder
    face_finder = FaceFinder(**kwargs)
    
    if training_set == None:
        training_set = eyes_file.files()
        
    if training_size == None or training_size > len(training_set):
        training_size = len(training_set)

    for filename in training_set[:training_size]:
        print "Processing file:",filename
        im_name = join(image_dir,filename+image_ext)
        
        # Detect faces
        im = Image(im_name)
        
        eyes = eyes_file.getEyes(filename)
        for left,right in eyes:
            face_finder.addTraining(left,right,im)
        
    face_finder.train()
    return face_finder           
 
 
#############################################################################
# Unit Tests
#############################################################################
class _TestFaceFinder(unittest.TestCase):
    
    def test_training(self):
        '''
        This trains the FaceFinder on the scraps database.
        '''
        
        # Load an eyes file
        eyes_filename = join(pyvision.__path__[0],'data','csuScrapShots','coords.txt')
        print "Creating eyes File."
        eyes_file = EyesFile(eyes_filename)
        
        # Create a face detector
        cascade_file = join(pyvision.__path__[0],'config','facedetector_celebdb2.xml')
        print "Creating a face detector from:",cascade_file
        face_detector = CascadeDetector(cascade_file)

        image_dir = join(pyvision.__path__[0],'data','csuScrapShots')
        
        face_finder = FaceFinderFromDatabase(eyes_file, image_dir, image_ext=".pgm", face_detector=face_detector)
        self.assert_(False) # remove training output
        self.assert_(False) # add a test
        
        
    def test_evaluation(self):
        '''
        This tests basing FaceFinder evaluation.
        '''
        self.assert_(False) # add a test
      
            
