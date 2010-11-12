'''
Created on Nov 9, 2010
@author: svohara
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

import pyvision as pv
from pyvision.surveillance.BackgroundSubtraction import *
import cv

class MotionDetector(object):
    '''
    Uses background subtraction from an image buffer to detect
    areas of motion in a video.
    
    The general process is to update the image buffer and then
    call the MotionDetector's detect() method.
    '''
    
    def __init__(self, imageBuff=None, thresh=20, method=BG_SUBTRACT_AMF, minArea=400, 
                 rectFilter=None, buffSize=5):
        '''
        Constructor
        @param imageBuff: a pv.ImageBuffer object to be used in the background subtraction
            step of the motion detection. If None, then this object will create an empty
            5-frame buffer, and until the buffer is full, the results of the motion detection
            will be nothing.
        @param thresh: Used by the background subtraction to eliminate noise.  
        @param method: Select background subtraction method. See constants defined in
          BackgroundSubtraction module
        @param minArea: minimum foreground contour area required for detection
        @param rectFilter: a function reference that takes a list of rectangles and
          returns a list filtered in some way. This allows the user to arbitrarily
          define rules to further limit motion detection results based on the geometry
          of the bounding boxes.
        @param buffSize: Only used if imageBuff==None. This controls the size of the
          internal image buffer.
        @note: Until the image buffer is full, the result of the motion detection will be
          nothing. See documentation on the detect(img) method of this class.
        '''
        #initialize object variables
        self._fgMask = None        
        self._minArea = minArea
        self._filter = rectFilter
        self._threshold = 20
        
        if imageBuff == None:
            self._imageBuff = pv.ImageBuffer(N=buffSize)
        else:
            self._imageBuff = imageBuff
        
        self._method = method      
        self._bgSubtract = None  #can't initialize until buffer is full...so done in detect()  
        
    def _initBGSubtract(self):
        if self._method==BG_SUBTRACT_FD:
            self._bgSubtract = pv.FrameDifferencer(self._imageBuff, self._threshold)
        elif self._method==BG_SUBTRACT_MF:
            self._bgSubtract = pv.MedianFilter(self._imageBuff, self._threshold)
        elif self._method==BG_SUBTRACT_AMF:
            self._bgSubtract = pv.ApproximateMedianFilter(self._imageBuff, self._threshold)
        else:
            raise ValueError("Unknown Background Subtraction Method specified.")
                  
    def _computeContours(self):
        cvMask = self._fgMask.asOpenCVBW()
        cvdst = cv.CloneImage(cvMask)  #because cv.FindContours may alter source image
        contours = cv.FindContours(cvdst, cv.CreateMemStorage(), cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)
        self._contours = contours
            
    def detect(self, img):
        '''
        You call this method to update detection results, given the new
        image in the stream. After updating detection results, use one
        of the getX() methods, such as getRects() to see the results in the
        appropriate format.
        
        @param img: A pv.Image() to be added to the buffer as the most recent image,
        and that triggers the new motion detection. Note that, depending on the
        background subtraction method, this may not be the "key frame" for the 
        detection. The Frame Differencer returns a background model based on the
        middle image, but Median and Approx. Median Filters return a background
        model based on the most recent (last) image in the buffer. 
        
        @return: The number of detected components in the current image. To get
        more details, use the various getX() methods, like getForegroundMask(),
        after calling detect().
        
        @note: Until the image buffer is full, this method will make no detections.
        In which case, the return value will be -1, indicating this status. Also,
        the getKeyFrame() method should be used to retrieve the key frame from
        the buffer, which is not always the most recent image, depending on background
        subtraction method.
        '''
        self._imageBuff.add(img)
        if not self._imageBuff.isFull():
            return -1
        
        #initialize background subtraction object only after buffer is full.
        if self._bgSubtract == None:
            self._initBGSubtract()

        mask = self._bgSubtract.getForegroundMask()
        cvBinary = mask.asOpenCVBW()
        cv.Dilate(cvBinary, cvBinary, None, 3)
        cv.Erode(cvBinary, cvBinary, None, 1)
        
        #update the foreground mask
        self._fgMask = pv.Image(cvBinary)
        
        #update the detected foreground contours
        self._computeContours()
        
        #update current annotation image from buffer, as appropriate for
        # the different methods
        if self._method==BG_SUBTRACT_FD:
            self._annotateImg = self._imageBuff.getMiddle()
        elif self._method==BG_SUBTRACT_MF:
            self._annotateImg = self._imageBuff.getLast()
        elif self._method==BG_SUBTRACT_AMF:
            self._annotateImg = self._imageBuff.getLast()
            
        return len(self._contours)

    def getKeyFrame(self):
        '''
        @return: The "key frame" of the motion detector's buffer. This is the image
        upon which detected motion rectangles, for example, should be overlaid. This
        is not always the last image in the buffer because some background subtraction
        methods (notably N-Frame Differencer) use the middle frame of the buffer.
        '''
        return self._annotateImg  #computed already by the detect() method    
    
    def getForegroundMask(self):
        '''
        @return: a binary pv.Image representing the foreground pixels
        as determined by the selected background subtraction method.
        @note: You must call the detect() method before getForegroundMask() to
        get the updated mask.
        '''
        return self._fgMask
    
    def getForegroundPixels(self):
        '''
        @return: The full color foreground pixels on a black background.
        @note: You must call detect() before getForegroundPixels() to
        get updated information.
        '''
        #binary mask selecting foreground regions
        mask = self._fgMask.asOpenCVBW()
        
        #full color source image
        image = self._annotateImg.copy().asOpenCV()
        
        #dest image, full color, but initially all zeros (black/background)
        # we will copy the foreground areas from image to here.
        dest = cv.CloneImage(image)
        cv.SetZero(dest)
        
        cv.Copy(image,dest,mask) #copy only pixels from image where mask != 0
        return pv.Image(dest)
            
    def getRects(self):
        '''
        @return: the bounding boxes of the external contours of the foreground mask.
        @note: You must call detect() before getRects() to see updated results.
        '''
        #create a list of the top-level contours found in the contours (cv.Seq) structure
        rects = []
        if len(self._contours) < 1: return(rects)
        seq = self._contours
        while not (seq == None):
            (x, y, w, h) = cv.BoundingRect(seq) 
            if (cv.ContourArea(seq) > self._minArea):
                r = pv.Rect(x,y,w,h)
                rects.append(r)
            seq = seq.h_next()
        
        if self._filter != None:
            rects = self._filter(rects)
        
        return rects
    
    def getAnnotatedImage(self, showContours=False):
        '''
        @return: the annotation image with bounding boxes
        and optionally contours drawn upon it.
        @note: You must call detect() prior to getAnnotatedImage()
        to see updated results.
        '''
        rects = self.getRects()
        outImg = self._annotateImg.copy()  #deep copy, so can freely modify the copy
        
        #draw contours in green
        if showContours:
            cvimg = outImg.asOpenCV()
            cv.DrawContours(cvimg, self._contours, cv.RGB(0, 255, 0), cv.RGB(255,0,0), 2)
        
        #draw bounding box in yellow
        for r in rects:
            outImg.annotateRect(r,"yellow")
        
        return outImg        
        
    def getForegroundTiles(self):
        '''
        @return: a list of "tiles", where each tile is a small pv.Image
        representing the clipped area of the annotationImg based on
        the motion detection. The foreground mask will be used to show
        only the foreground pixels within each tile.
        @note: You must call detect() prior to getForegroundTiles() to get
        updated information.
        '''
        
        #binary mask selecting foreground regions
        mask = self._fgMask.asOpenCVBW()
        
        #full color source image
        image = self._annotateImg.copy().asOpenCV()
        
        #dest image, full color, but initially all zeros (black/background)
        # we will copy the foreground areas from image to here.
        dest = cv.CloneImage(image)
        cv.SetZero(dest)
        
        cv.Copy(image,dest,mask) #copy only pixels from image where mask != 0
        dst = pv.Image(dest)
        
        rects = self.getRects()
        
        tiles = []
        for r in rects:
            #for every rectangle, crop from dest image
            t = dst.crop(r)
            tiles.append(t)
            
        return tiles