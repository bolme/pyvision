'''
Created on Oct 22, 2010
@author: Stephen O'Hara
'''
# PyVision License
#
# Copyright (c) 2006-2008 Stephen O'Hara
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
import scipy as sp
import pyvision as pv

class FrameDifferencer:
    '''
    This class is useful for simple N-frame differencing method of
    background subtraction. If you have a stationary camera, this can
    be a simple and effective way to isolate people/moving objects
    from the background scene.
    
    FrameDifferencer uses ImageBuffer for operation. Assume the buffer
    size is 5. The output of the frame differencing operation will
    be based on the middle image, the 3rd in the buffer. The output
    is the intersection of the following two absolute differences:
    abs(Middle-First) AND abs(Last-Middle).
    '''

    def __init__(self, imageBuffer, thresh=20):
        '''
        @param imageBuffer: An ImageBuffer object that has already been filled
        with the appropriate number of images. (Provide a full buffer...so a few
        frames of initialization will be required in most cases to fill up a
        newly created buffer.)     
        @param thresh: A noise threshold to remove very small differences.    
        '''
        self._imageBuffer = imageBuffer
        self._threshold = thresh
        
        
    def getDiffImage(self):
        '''
        @return: a version of imagebuffer.getMiddle() with background subtraction
            via frame differencing first and last frames.
        '''
        prevImg = self._imageBuffer[0].asMatrix2D()
        curImg = self._imageBuffer.getMiddle().asMatrix2D()
        nextImg = self._imageBuffer[-1].asMatrix2D()
        
        delta1 = sp.absolute(curImg - prevImg)   #frame diff
        delta1 = (delta1 > self._threshold)     #threshoopencv dilatelding to binary image
            
        delta2 = sp.absolute(nextImg - curImg)   #frame diff
        delta2 = (delta2 > self._threshold)     #thresholding
        
        mask = sp.logical_and(delta1,delta2)
    
        return pv.Image(mask*255.0) 