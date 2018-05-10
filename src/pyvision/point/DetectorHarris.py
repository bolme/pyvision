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
import os.path

import numpy as np

#import cv2
import scipy.ndimage as ndi
import pyvision as pv
from pyvision.point.DetectorROI import DetectorROI


class DetectorHarris(DetectorROI):
    def __init__(self,block_size = 7, aperture_size=3, k=0.04, **kwargs):
        pass
        DetectorROI.__init__(self,**kwargs)
        
        self.block_size = block_size
        self.aperture_size = aperture_size
        self.k = k


    def _detect(self,im):
        '''
        void cvCornerHarris( const CvArr* image, CvArr* harris_responce, int block_size, int aperture_size=3, double k=0.04 );
        '''
        import cv2
        
        gray = im.asOpenCV2BW()
        #gray = opencv.cvCreateImage( opencv.cvGetSize(cvim), 8, 1 );
        #corners = cv.CreateImage( cv.GetSize(gray), 32, 1 );
        #opencv.cvCvtColor( cvim, gray, opencv.CV_BGR2GRAY );
    
        corners = cv2.cornerHarris(gray.T,self.block_size,self.aperture_size,self.k)

        #data_buffer = corners.tostring()
        #corners = np.frombuffer(data_buffer,np.float32).reshape(corners.height,corners.width).transpose()        
        
        footprint = np.ones((3,3))
        mx = ndi.maximum_filter(corners, footprint = footprint)
        local_maxima = (corners == mx) * (corners != np.zeros(corners.shape)) # make sure to remove completly dark points

        points = np.nonzero(local_maxima)
        del local_maxima
        
        points = np.array([points[0],points[1]]).transpose()
        L = []
        for each in points:
            L.append((corners[each[0],each[1]],each[0],each[1],None))
        
        return L


class _HarrisTest(unittest.TestCase):
    def setUp(self):
        self.SHOW_IMAGES = False
        
    
        
    def testDetectorHarris1(self):
        detector = DetectorHarris()
        filename = os.path.join(pv.__path__[0],'data','nonface','NONFACE_1.jpg')
        im = pv.Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for _,pt,_ in points:
            im.annotatePoint(pt)
          
        if self.SHOW_IMAGES: im.show()  
        self.assertLess(len(points),550)
        self.assertGreater(len(points),350)

    def testDetectorHarris2(self):
        detector = DetectorHarris()
        filename = os.path.join(pv.__path__[0],'data','nonface','NONFACE_19.jpg')
        im = pv.Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for _,pt,_ in points:
            im.annotatePoint(pt)
            
        if self.SHOW_IMAGES: im.show()  
        self.assertLess(len(points),550)
        self.assertGreater(len(points),350)

    def testDetectorHarris3(self):
        detector = DetectorHarris()
        filename = os.path.join(pv.__path__[0],'data','nonface','NONFACE_22.jpg')
        im = pv.Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for _,pt,_ in points:
            im.annotatePoint(pt)
            
        if self.SHOW_IMAGES: im.show()  
        self.assertLess(len(points),550)
        self.assertGreater(len(points),350)

    def testDetectorHarris4(self):
        detector = DetectorHarris()
        filename = os.path.join(pv.__path__[0],'data','nonface','NONFACE_37.jpg')
        im = pv.Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for _,pt,_ in points:
            im.annotatePoint(pt)
            
        if self.SHOW_IMAGES: im.show()  
        self.assertLess(len(points),550)
        self.assertGreater(len(points),350)

    def testDetectorHarris5(self):
        detector = DetectorHarris(selector='best')
        filename = os.path.join(pv.__path__[0],'data','nonface','NONFACE_37.jpg')
        im = pv.Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for _,pt,_ in points:
            im.annotatePoint(pt)
            
        if self.SHOW_IMAGES: im.show()  
        self.assertEqual(len(points),250)
        
    def testDetectorHarris6(self):
        detector = DetectorHarris(selector='all')
        filename = os.path.join(pv.__path__[0],'data','nonface','NONFACE_37.jpg')
        im = pv.Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for _,pt,_ in points:
            im.annotatePoint(pt)
            
        
        if self.SHOW_IMAGES: im.show()  
        
        self.assertLess(len(points),7000)
        self.assertGreater(len(points),6800)
        

