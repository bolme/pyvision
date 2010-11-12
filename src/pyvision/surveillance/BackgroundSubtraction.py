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

#Constants used to identify a background subtraction method,
# useful, for example, for specifying which method to use in the
# MotionDetector class.
BG_SUBTRACT_FD = 1  #frame differencer
BG_SUBTRACT_MF = 2  #median filter
BG_SUBTRACT_AMF = 3 #approx median filter

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
        
        
    def getForegroundMask(self):
        '''
        @return: a version of imagebuffer.getMiddle() with background subtraction
            via frame differencing first and last frames. Note, one will likely
            have to perform additional morphological operations on the foreground
            mask prior to use.
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
    
class MedianFilter:
    '''
    Uses median pixel values of the images in a buffer to
    approximate a background model.
    '''
    def __init__(self, imageBuffer, thresh=20):
        '''
        @param imageBuffer: An ImageBuffer object that has already been filled
        with the appropriate number of images. (Provide a full buffer...so a few
        frames of initialization will be required in most cases to fill up a
        newly created buffer.)     
        @param thresh: A noise threshold to remove very small differences from
         the background model 
        '''
        self._imageBuffer = imageBuffer
        self._threshold = thresh
            
    def _getMedianVals(self):
        '''
        @return: A scipy matrix representing the gray-scale median values of the image stack.
           If you want a pyvision image, just wrap the result in pv.Image(result).
        '''
        self._imageStack = self._imageBuffer.asStackBW()
        medians = sp.median(self._imageStack, axis=0) #median of each pixel jet in stack
        return medians
    
    def getForegroundMask(self):
        '''
        @return: a version of imagebuffer.getLast() with background subtraction
            via subtracting the median values from the buffer. Note, one will likely
            have to perform additional morphological operations on the foreground
            mask prior to use.
        '''
        imgGray = self._imageBuffer.getLast().asMatrix2D()
        imgBG = self._getMedianVals()
        diff = abs(imgGray - imgBG)
        mask = (diff > self._threshold)
        return pv.Image( mask * 255.0)    
            
            
class ApproximateMedianFilter(MedianFilter):
    '''
    Approximates the median pixels via an efficient incremental algorithm that
    would converge to the true median in a perfect world. It initializes a
    median image based on the images in the initial image buffer, but
    then only updates the median image using the last (newest) image in the
    buffer.
    '''
    def __init__(self, imageBuffer, thresh=20):
        '''
        @param imageBuffer: An ImageBuffer object that has already been filled
        with the appropriate number of images. (Provide a full buffer...so a few
        frames of initialization will be required in most cases to fill up a
        newly created buffer.)     
        @param thresh: A noise threshold to remove very small differences from
         the background model 
        '''
        if not imageBuffer.isFull():
            raise ValueError("Image Buffer must be full before initializing Approx. Median Filter.")
        MedianFilter.__init__(self, imageBuffer, thresh)
        self._medians = self._getMedianVals()
        
    def _updateMedian(self):
        curImg = self._imageBuffer.getLast()
        curMat = curImg.asMatrix2D()
        median = self._medians
        up = (curMat > median)*1.0
        down = (curMat < median)*1.0
        self._medians = self._medians + up - down
            
    def getForegroundMask(self):
        self._updateMedian()
        imgGray = self._imageBuffer.getLast().asMatrix2D()
        imgBG = self._medians
        diff = abs(imgGray - imgBG)
        mask = (diff > self._threshold)
        return pv.Image( mask * 255.0)  
                 
    