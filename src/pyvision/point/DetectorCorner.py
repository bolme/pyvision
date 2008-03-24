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

# TODO: document module.

import unittest
import os.path  

from numpy import array,ones,zeros,nonzero
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter,maximum_filter
# TODO: At some point it would be nice to have the options of prewitt or sobel as filters.
# from scipy.ndimage import prewitt,sobel
from numpy.linalg import det

import pyvision
from pyvision.point.DetectorROI import DetectorROI
from pyvision.types.Image import Image
from pyvision.types.Point import Point


conv2 = convolve

class DetectorCorner(DetectorROI):
    def __init__(self,filter = [[-1,0,1]], radius=9, sigma=0.7, k=0.04, **kwargs):
        '''
        Corner Detector
        
        filter   - first dirivitive filter
        radius   - radius of the max filter
        sigma    - sigma of the smoothing gaussian.
        k        - not sure what this parameter means.
        
        Passed to superclass:
        n        - is the approximate number of points requested.
        bin_size - the width and height of each bin in pixels.
        corner_selector ('all', 'bins', or 'best') - stratagy for point selection.
        
        When corner_selector is set to bins, the image is subdivided in to bins of
        size <bin_size>X<bin_size> pixels and an equal number of points will be taken
        from each of those bins.  This insures that points are found in all parts of the
        image not just where the corners are strongest.
        
        This code is based on a function originally written for matlab.
        
        Original matlab code by: 
        Jingyu Yan and Marc Pollefeys
        Department of Computer Science
        University of North Carolina at Chapel Hill
        
        Converted to Python by: 
        David Bolme
        Department of Computer Science
        Colorado State Univerisity
        '''
        DetectorROI.__init__(self,**kwargs)
        
        self.filter = filter
        self.radius = radius
        self.sigma = sigma
        self.k = k
        
    def _detect(self,image):
        # Asssumes a two dimensional array
        A = None
        if isinstance(image,Image):
            A = image.asMatrix2D()
        elif isinstance(image,array) and len(image.shape)==2:
            A = image
        else:
            raise TypeError("ERROR Unknown Type (%s) - Only arrays and pyvision images supported."%type(image))
    
        filter = array(self.filter)
        assert len(filter.shape) == 2
        
        #feature window calculation
        del_A_1 = conv2(A,filter) 
        del_A_2 = conv2(A,filter.transpose())
    
    
        del_A_1_1 = del_A_1 * del_A_1
        matrix_1_1 = gaussian_filter(del_A_1_1, self.sigma)
        del del_A_1_1
            
        del_A_2_2 = del_A_2 * del_A_2
        matrix_2_2 = gaussian_filter(del_A_2_2, self.sigma)
        del del_A_2_2
        
        del_A_1_2 = del_A_1 * del_A_2
        matrix_1_2 = gaussian_filter(del_A_1_2, self.sigma)
        del del_A_1_2
        
        del del_A_1,del_A_2
        
        dM = matrix_1_1*matrix_2_2 - matrix_1_2*matrix_1_2
        tM = matrix_1_1+matrix_2_2
        
        del matrix_1_1 , matrix_1_2, matrix_2_2
        
        R = dM-self.k*pow(tM,2)
        
        footprint = ones((self.radius,self.radius))
        mx = maximum_filter(R, footprint = footprint)
        local_maxima = (R == mx) * (R != zeros(R.shape)) # make sure to remove completly dark points
        del mx
        
        points = nonzero(local_maxima)
        del local_maxima
        
        points = array([points[0],points[1]]).transpose()
        L = []
        for each in points:
            L.append((R[each[0],each[1]],each[0],each[1],None))
        
        del R
        
        return L



class _CornerTest(unittest.TestCase):
    def setUp(self):
        self.SHOW_IMAGES = True
    
        
    def testDetectorCorner1(self):
        detector = DetectorCorner()
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_1.jpg')
        im = Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for score,pt,radius in points:
            im.annotatePoint(pt)
            
        if self.SHOW_IMAGES: im.show()  
        self.assertEquals(len(points),390)

    def testDetectorCorner2(self):
        detector = DetectorCorner()
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_19.jpg')
        im = Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for score,pt,radius in points:
            im.annotatePoint(pt)
            
        if self.SHOW_IMAGES: im.show()  
        self.assertEquals(len(points),390)

    def testDetectorCorner3(self):
        detector = DetectorCorner()
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_22.jpg')
        im = Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for score,pt,radius in points:
            im.annotatePoint(pt)
            
        if self.SHOW_IMAGES: im.show()  
        self.assertEquals(len(points),390)

    def testDetectorCorner4(self):
        detector = DetectorCorner()
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_37.jpg')
        im = Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for score,pt,radius in points:
            im.annotatePoint(pt)
            
        if self.SHOW_IMAGES: im.show()  
        self.assertEquals(len(points),351)

    def testDetectorCorner5(self):
        detector = DetectorCorner(selector='best')
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_37.jpg')
        im = Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for score,pt,radius in points:
            im.annotatePoint(pt)
            
        if self.SHOW_IMAGES: im.show()  
        self.assertEquals(len(points),250)
        
    def testDetectorCorner6(self):
        detector = DetectorCorner(selector='all')
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_37.jpg')
        im = Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for score,pt,radius in points:
            im.annotatePoint(pt)
            
        if self.SHOW_IMAGES: im.show()  
        self.assertEquals(len(points),2149)
        

