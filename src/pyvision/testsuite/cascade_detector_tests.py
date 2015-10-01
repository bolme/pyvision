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

import unittest
import pyvision as pv
from pyvision.face.CascadeDetector import CascadeDetector,CascadeNotFound,OPENCV_CASCADE
from pyvision.analysis.face import EyesFile

from pyvision.analysis.FaceAnalysis.FaceDetectionTest import FaceDetectionTest
import pickle

import os

SCRAPS_FACE_DATA = os.path.join(pv.__path__[0],"data","csuScrapShots")
NONFACE_DATA = os.path.join(pv.__path__[0],"data","NonFace")
BAD_CASCADE=os.path.join(pv.__path__[0],"config","not_there.xml")


class _TestCascadeDetector(unittest.TestCase):
    ''' Unit tests for the Cascade Detector '''
    
    
    def test_detect_bad_file(self):
        '''
        If the cascade file does not exist, opencv can crash without warning.
        This makes sure a test is run to make sure the cascade is there.
        '''
        
        self.assertRaises(CascadeNotFound,CascadeDetector,BAD_CASCADE)
        
    def test_face_detection_pickle(self):
        fd = CascadeDetector(OPENCV_CASCADE)
        
        fdt = FaceDetectionTest(name='scraps')

        data_buffer = pickle.dumps(fd)
        fd = pickle.loads(data_buffer)
        
        self.eyes = EyesFile(os.path.join(SCRAPS_FACE_DATA,"coords.txt"))
        for filename in self.eyes.files():
            img = pv.Image(os.path.join(SCRAPS_FACE_DATA, filename + ".pgm"))
            rects = fd(img)
            truth = self.eyes.getFaces(img.filename)
            fdt.addSample(truth,rects,im=img)

        self.assertAlmostEqual( fdt.pos_rate , 0.98265895953757221, places = 2 ) # TODO: Version 2 performance is better

        
    def test_detect_scraps_opencv(self):
        
        fd = CascadeDetector(OPENCV_CASCADE)
        fdt = FaceDetectionTest(name='scraps')
        
        self.eyes = EyesFile(os.path.join(SCRAPS_FACE_DATA,"coords.txt"))
        for filename in self.eyes.files():
            img = pv.Image(os.path.join(SCRAPS_FACE_DATA, filename + ".pgm"))
            rects = fd(img)
            truth = self.eyes.getFaces(img.filename)
            fdt.addSample(truth,rects,im=img)

        self.assertAlmostEqual( fdt.pos_rate , 0.98265895953757221, places = 2 ) # TODO: Version 2 performance is better
          

    def donttest_detector_train(self): # TODO: Cascade training fails for Version OpenCV 2.0

        positives = []    
        self.eyes = EyesFile(os.path.join(SCRAPS_FACE_DATA,"coords.txt"))
        n = len(self.eyes.files())    
        for filename in self.eyes.files()[:n/2]:
            img = pv.Image(os.path.join(SCRAPS_FACE_DATA, filename + ".pgm"))
            faces = self.eyes.getFaces(img.filename)
            positives.append([os.path.join(SCRAPS_FACE_DATA,img.filename),faces])   
            
        neg_files = []
        for im_name in os.listdir(NONFACE_DATA):    
            if im_name[-4:] != ".jpg": continue
            neg_files.append(os.path.join(NONFACE_DATA,im_name))

        fd = trainHaarClassifier(positives,neg_files,nstages=6,maxtreesplits=0,max_run_time=300)
        fdt = FaceDetectionTest(name='scraps')
        
        self.eyes = EyesFile(os.path.join(SCRAPS_FACE_DATA,"coords.txt"))
        for filename in self.eyes.files():
            img = pv.Image(os.path.join(SCRAPS_FACE_DATA, filename + ".pgm"))
            rects = fd(img)
            truth = self.eyes.getFaces(img.filename)
            fdt.addSample(truth,rects,im=img)

        self.assertAlmostEqual( fdt.pos_rate , 0.9942196531791907 )
 


