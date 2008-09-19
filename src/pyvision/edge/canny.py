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

import opencv

import pyvision
from pyvision.types.Image import Image
from pyvision.analysis.ImageLog import ImageLog

from numpy import array,ones,zeros,nonzero
from scipy.ndimage import convolve
from scipy.ndimage import maximum_filter


def canny(im,threshold1=40.0,threshold2=100.0,aperture_size=3):
    '''
    void cvCanny( const CvArr* image, CvArr* edges, double threshold1,
              double threshold2, int aperture_size=3 );
    '''
    cvim = im.asOpenCV()
    edges = opencv.cvCreateImage( opencv.cvGetSize(cvim), 8, 1 );

    if cvim.nChannels == 3:
        gray = opencv.cvCreateImage( opencv.cvGetSize(cvim), 8, 1 );
        opencv.cvCvtColor( cvim, gray, opencv.CV_BGR2GRAY );
    else:
        gray = cvim

    opencv.cvCanny(gray,edges,threshold1,threshold2,aperture_size)
    
    return Image(edges)
    
    
class _TestCanny(unittest.TestCase):
    ''' Unit tests for the canny detector'''
    
    def setUp(self):
        self.show_results = False
        pass
        
    def test_canny1(self):
        '''
        This will run the code, but what is a good test for canny?
        '''
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_46.jpg')
                        
        img = Image(filename)
        out = canny(img)
        if self.show_results: out.show()
        

    def test_canny2(self):
        '''
        This will run the code, but what is a good test for canny?
        '''
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_10.jpg')
                        
        img = Image(filename)
        out = canny(img)
        if self.show_results: out.show()
        

    def test_canny3(self):
        '''
        This will run the code, but what is a good test for canny?
        '''
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_22.jpg')
                        
        img = Image(filename)
        out = canny(img)
        if self.show_results: out.show()
        

    def test_canny4(self):
        '''
        This will run the code, but what is a good test for canny?
        '''
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_44.jpg')
                        
        img = Image(filename)
        out = canny(img)
        if self.show_results: out.show()

    def test_canny5(self):
        '''
        This will run the code, but what is a good test for canny?
        '''
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_37.jpg')
                        
        img = Image(filename)
        out = canny(img)
        if self.show_results: out.show()
        
        

  