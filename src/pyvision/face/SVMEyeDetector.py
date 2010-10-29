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
import random
from math import pi
import os.path

import pyvision as pv
import numpy as np

from pyvision.face.CascadeDetector import CascadeDetector
from pyvision.analysis.face import EyesFile
from pyvision.point.PointLocator import SVMLocator,KRRLocator
from pyvision.analysis.FaceAnalysis.FaceDetectionTest import face_from_eyes, is_success
from pyvision.analysis.FaceAnalysis.EyeDetectionTest import EyeDetectionTest
from pyvision.vector import VectorClassifier 
from pyvision.vector import SVM 


class SVMEyeDetector:
    ''' 
    This class detects faces and then returns the detection rectangles and 
    the eye coordinates.
    '''
    
    def __init__(self, face_detector=CascadeDetector(), tile_size=(128,128), validate=None, n_iter=1, annotate=False,**kwargs):
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
        self.n_iter        = n_iter
        self.annotate      = annotate
        self.perturbations = True

        # this object handles pca normalization
        self.normalize = VectorClassifier.VectorClassifier(
                                VectorClassifier.TYPE_REGRESSION,
                                reg_norm=VectorClassifier.REG_NORM_NONE)
        
        # point locators that learn to find the eyes.
        self.left_locator  = SVMLocator(type=SVM.TYPE_NU_SVR ,normalization=VectorClassifier.NORM_NONE)
        self.right_locator = SVMLocator(type=SVM.TYPE_NU_SVR ,normalization=VectorClassifier.NORM_NONE)
        
        # Number of training images where the face detection did not work.
        self.detection_failures = 0
        
        self.training_labels = []

        
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
                affine = pv.AffineFromRect(pred_rect,self.tile_size)

                w,h = self.tile_size
                
                if self.perturbations:
                    # Randomly rotate, translate and scale the images
                    center = pv.AffineTranslate(-0.5*w,-0.5*h,self.tile_size)
                    rotate = pv.AffineRotate(random.uniform(-pi/8,pi/8),self.tile_size)
                    scale = pv.AffineScale(random.uniform(0.9,1.1),self.tile_size)
                    translate = pv.AffineTranslate(random.uniform(-0.05*w,0.05*w),
                                               random.uniform(-0.05*h,0.05*h),
                                               self.tile_size)
                    inv_center = pv.AffineTranslate(0.5*w,0.5*h,self.tile_size)
                    
                    affine = inv_center*translate*scale*rotate*center*affine
                    #affine = affine*center*rotate*scale*translate*inv_center
                
                cropped = affine.transformImage(im)
                cropped = pv.meanStd(cropped)
                
                # Mark the eyes
                leye = affine.transformPoint(left_eye)
                reye = affine.transformPoint(right_eye)

                # Add training data to locators
                self.training_labels.append((leye,reye))

                self.normalize.addTraining(0.0,cropped)
                #self.left_locator.addTraining(cropped,leye)
                #self.right_locator.addTraining(cropped,reye)
                
                # Just use the first success
                return
            
        # The face was not detected
        self.detection_failures += 1
        
    def train(self,**kwargs):
        '''
        Train the eye locators.
        '''
        self.normalize.trainNormalization()
        
        vectors = self.normalize.vectors
        
        #print len(self.training_labels)
        #print vectors.shape
        
        for i in range(len(self.training_labels)):
            leye,reye = self.training_labels[i]
            vec = vectors[i]
            
            self.left_locator.addTraining(vec,leye)
            self.right_locator.addTraining(vec,reye)

        self.left_locator.train(**kwargs)
        self.right_locator.train(**kwargs)        
        
        self.left_eye      = self.left_locator.mean
        self.right_eye     = self.right_locator.mean
        
        del self.normalize.labels
        del self.normalize.vectors
        del self.training_labels

        
    def detect(self, im):
        '''
        @returns: a list of tuples where each tuple contains (registered_image, detection_rect, left_eye, right_eye) 
        '''
        result = []
        
        rects = self.face_detector.detect(im)
        
        # Anotate Faces
        for rect in rects:
            
            # Transform the face
            affine = pv.AffineFromRect(rect,self.tile_size)
            cropped = affine.transformImage(im)
            
            for p in range(self.n_iter):
                cropped = pv.meanStd(cropped)
                # Find the eyes
                data = cropped.asMatrix2D().flatten()   
                data = np.array(data,'d').flatten()
        
                data = self.normalize.normalizeVector(data)
      
                pleye = self.left_locator.predict(data)
                preye = self.right_locator.predict(data)
                
                pleye = affine.invertPoint(pleye)
                preye = affine.invertPoint(preye)
                
                # Seccond Pass
                affine = pv.AffineFromPoints(pleye,preye,self.left_eye,self.right_eye,self.tile_size)
                cropped = affine.transformImage(im)
            
            #affine = AffineFromPoints(pleye,preye,self.left_eye,self.right_eye,self.tile_size)
            #reg = affine.transformImage(im)
            reg = cropped

            if self.validate != None and not self.validate(reg):
                # Validate the face.
                if self.annotate:
                    im.annotateRect(rect,color='red')        
                    im.annotatePoint(pleye,color='red')
                    im.annotatePoint(preye,color='red')
                continue
            
            if self.annotate:
                reg.annotatePoint(self.left_eye,color='green')
                reg.annotatePoint(self.right_eye,color='green')
                im.annotatePoint(pleye,color='green')
                im.annotatePoint(preye,color='green')
                im.annotateRect(rect,color='green')        
            result.append((reg,rect,pleye,preye))
            
        return result


class RegressionEyeLocator2:
    ''' 
    This class detects faces and then returns the detection rectangles and 
    the eye coordinates.
    '''
    
    def __init__(self, face_detector=CascadeDetector(), tile_size=(128,128), subtile_size=(32,32), left_center=pv.Point(39.325481787836871,50.756936769089975), right_center=pv.Point(91.461135538006289,50.845357457309881), validate=None, n_iter=1, annotate=False,**kwargs):
        ''' 
        Create an eye locator.  This default implentation uses a 
        cascade classifier for face detection and then SVR for eye
        location. 
        '''
        #TODO: Learn the mean eye locations durring training.
        self.face_detector = face_detector
        self.left_center   = left_center
        self.right_center  = right_center
        self.tile_size     = tile_size
        self.subtile_size  = subtile_size
        self.validate      = validate
        self.n_iter        = n_iter
        self.annotate      = annotate
        self.perturbations = True

        # Number of training images where the face detection did not work.
        self.detection_failures = 0

        # point locators that learn to find the eyes.
        self.createLocators(**kwargs)
        
        
    def createLocators(self,**kwargs):
        ''' Create two point locators that use the methods of interest '''
        raise NotImplementedError
                
        
    def generateTransforms(self,detection):        
        # Transform the face
        affine = pv.AffineFromRect(detection,self.tile_size)

        w,h = self.tile_size
        
        if self.perturbations:
            # Randomly rotate, translate and scale the images
            center = pv.AffineTranslate(-0.5*w,-0.5*h,self.tile_size)
            rotate = pv.AffineRotate(random.uniform(-pi/8,pi/8),self.tile_size)
            scale = pv.AffineScale(random.uniform(0.9,1.1),self.tile_size)
            translate = pv.AffineTranslate(random.uniform(-0.05*w,0.05*w),
                                       random.uniform(-0.05*h,0.05*h),
                                       self.tile_size)
            inv_center = pv.AffineTranslate(0.5*w,0.5*h,self.tile_size)
            
            affine = inv_center*translate*scale*rotate*center*affine
            #affine = affine*center*rotate*scale*translate*inv_center

        lx=self.left_center.X()-self.subtile_size[0]/2
        ly=self.left_center.Y()-self.subtile_size[1]/2
        rx=self.right_center.X()-self.subtile_size[0]/2
        ry=self.right_center.Y()-self.subtile_size[1]/2
        
        laffine = pv.AffineFromRect(pv.Rect(lx,ly,self.subtile_size[0],self.subtile_size[1]),self.subtile_size)*affine
        raffine = pv.AffineFromRect(pv.Rect(rx,ry,self.subtile_size[0],self.subtile_size[1]),self.subtile_size)*affine
        return laffine,raffine
                

    def addTraining(self, left_eye, right_eye, im):
        '''Train an eye detector givin a full image and the eye coordinates.'''
        
        # determine the face rect
        true_rect = face_from_eyes(left_eye,right_eye)
        
        # run the face detector
        rects = self.face_detector.detect(im)
        
        # find the best detection if there is one
        for pred_rect in rects:
            if is_success(pred_rect,true_rect):
                
                laffine,raffine = self.generateTransforms(pred_rect)
                
                lcropped = laffine.transformImage(im)
                rcropped = raffine.transformImage(im)
                
                #Normalize the images
                lcropped = pv.meanStd(lcropped)
                rcropped = pv.meanStd(rcropped)
                
                # Mark the eyes
                leye = laffine.transformPoint(left_eye)
                reye = raffine.transformPoint(right_eye)

                # Add training data to locators
                self.left_locator.addTraining(lcropped,leye)
                self.right_locator.addTraining(rcropped,reye)
                
                # Just use the first success
                return
            
        # The face was not detected
        self.detection_failures += 1
        
    def train(self,**kwargs):
        '''
        Train the eye locators.
        '''
        self.left_locator.train(**kwargs)
        self.right_locator.train(**kwargs)        
        
        self.left_eye      = self.left_locator.mean
        self.right_eye     = self.right_locator.mean
                
        self.perturbations=False

        
    def detect(self, im):
        '''
        @returns: a list of tuples where each tuple contains (registered_image, detection_rect, left_eye, right_eye) 
        '''
        result = []
        
        rects = self.face_detector.detect(im)
        
        # Anotate Faces
        for rect in rects:
            
            # Transform the face
            laffine,raffine = self.generateTransforms(rect)
            lcropped = laffine.transformImage(im)
            rcropped = raffine.transformImage(im)

            #Normalize the images
            lcropped = pv.meanStd(lcropped)
            rcropped = pv.meanStd(rcropped)
                  
            pleye = self.left_locator.predict(lcropped)
            preye = self.right_locator.predict(rcropped)
                
            pleye = laffine.invertPoint(pleye)
            preye = raffine.invertPoint(preye)
                
            
            affine = pv.AffineFromPoints(pleye,preye,self.left_eye,self.right_eye,self.tile_size)
            reg = affine.transformImage(im)

            if self.validate != None and not self.validate(reg):

                # Validate the face.
                if self.annotate:
                    im.annotateRect(rect,color='red')        
                    im.annotatePoint(pleye,color='red')
                    im.annotatePoint(preye,color='red')
                continue
            
            if self.annotate:
                reg.annotatePoint(self.left_eye,color='green')
                reg.annotatePoint(self.right_eye,color='green')
                im.annotatePoint(pleye,color='green')
                im.annotatePoint(preye,color='green')
                im.annotateRect(rect,color='green')        
            result.append((reg,rect,pleye,preye))
            
        return result


class SVMEyeLocator2(RegressionEyeLocator2):
    ''' 
    This class detects faces and then returns the detection rectangles and 
    the eye coordinates.
    '''        
    def createLocators(self,**kwargs):
        ''' Create two point locators that use the methods of interest '''
        self.left_locator  = SVMLocator(type=SVM.TYPE_NU_SVR ,normalization=VectorClassifier.NORM_VALUE)
        self.right_locator = SVMLocator(type=SVM.TYPE_NU_SVR ,normalization=VectorClassifier.NORM_VALUE)
        
        
class KRREyeLocator2(RegressionEyeLocator2):
    ''' 
    This class detects faces and then returns the detection rectangles and 
    the eye coordinates.
    '''        
    def createLocators(self,**kwargs):
        ''' Create two point locators that use the methods of interest '''
        self.left_locator  = KRRLocator(**kwargs)
        self.right_locator = KRRLocator(**kwargs)
                



def SVMEyeDetectorFromDatabase(eyes_file, image_dir, training_set = None, training_size=1000, image_ext='.jpg', **kwargs):
    '''
    Train a face finder using an Eye Coordanates file and a face dataset.
    '''

    # Create a face finder
    face_finder = SVMEyeDetector(**kwargs)
    
    if training_set == None:
        training_set = eyes_file.files()
        
    if training_size == None or training_size > len(training_set):
        training_size = len(training_set)

    for filename in training_set[:training_size]:
        #print "Processing file:",filename
        im_name = join(image_dir,filename+image_ext)
        
        # Detect faces
        im = pv.Image(im_name)
        
        eyes = eyes_file.getEyes(filename)
        for left,right in eyes:
            face_finder.addTraining(left,right,im)
        
    face_finder.train()
    return face_finder           
 
 
#############################################################################
# Unit Tests
#############################################################################
class _TestSVMEyeDetector(unittest.TestCase):
    def setUp(self):
        self.images = []
        self.names = []
        
        SCRAPS_FACE_DATA = os.path.join(pv.__path__[0],"data","csuScrapShots")
        
        
        self.eyes = EyesFile(os.path.join(SCRAPS_FACE_DATA,"coords.txt"))
        for filename in self.eyes.files():
            img = pv.Image(os.path.join(SCRAPS_FACE_DATA, filename + ".pgm"))
            self.images.append(img)
            self.names.append(filename)
        
        self.assert_( len(self.images) == 173 )
    
    def test_training(self):
        '''
        This trains the FaceFinder on the scraps database.
        '''
        #import cProfile

        # Load an eyes file
        eyes_filename = join(pv.__path__[0],'data','csuScrapShots','coords.txt')
        #print "Creating eyes File."
        eyes_file = EyesFile(eyes_filename)
        
        # Create a face detector
        cascade_file = join(pv.__path__[0],'config','facedetector_celebdb2.xml')
        #print "Creating a face detector from:",cascade_file
        face_detector = CascadeDetector(cascade_file)

        image_dir = join(pv.__path__[0],'data','csuScrapShots')
        
        ed = SVMEyeDetectorFromDatabase(eyes_file, image_dir, image_ext=".pgm", face_detector=face_detector,random_seed=0)
        edt = EyeDetectionTest(name='scraps')

        #print "Testing..."
        for img in self.images:
            #print img.filename
            faces = ed.detect(img)

            #faces = ed.detect(img)
            pred_eyes = []
            for _,_,pleye,preye in faces:
                #detections.append(rect)
                pred_eyes.append((pleye,preye))

            truth_eyes = self.eyes.getEyes(img.filename)
            edt.addSample(truth_eyes, pred_eyes, im=img, annotate=False)
        
        #print edt.createSummary()
        self.assertAlmostEqual( edt.face_rate ,   0.924855491329, places = 3 )
        
        #TODO: Randomization is causing issues with getting the eye detector to perform consistently
        #self.assertAlmostEqual( edt.both25_rate , 0.907514450867, places = 3 )
        #self.assertAlmostEqual( edt.both10_rate , 0.745664739884, places = 3 )
        #self.assertAlmostEqual( edt.both05_rate , 0.277456647399, places = 3 )
        
        
      
            
