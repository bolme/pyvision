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
    
    def __init__(self, imageBuff, thresh=20, method=BG_SUBTRACT_FD):
        '''
        Constructor
        @param imageBuff: a pv.ImageBuffer object, already full, to be used
          in the background subtraction step of the motion detection.
        @param thresh: Used by the background subtraction to eliminate noise.  
        '''
        if method==BG_SUBTRACT_FD:
            self._bgSubtract = pv.FrameDifferencer(imageBuff, thresh)
        elif method==BG_SUBTRACT_MF:
            self._bgSubtract = pv.MedianFilter(imageBuff, thresh)
        elif method==BG_SUBTRACT_AMF:
            self._bgSubtract = pv.ApproximateMedianFilter(imageBuff, thresh)
        else:
            raise ValueError("Unknown Background Subtraction Method specified.")
        
    def detect(self, minArea=400, annotateImg=None, rectFilter=None):
        '''
        @param minArea: Any contours with less than this area in pixels will be dropped
        @param annotateImg: The pv.Image on which to annotate the detection, use None for
         no annotations.
        @param rectFilter: A function to be applied to the list of rectangles to filter
          out those that don't meet some specification, such as being too thin, etc. 
        '''
        mask = self._bgSubtract.getForegroundMask()
        cvBinary = mask.asOpenCVBW()
        #cvdst = cv.CreateImage(binaryImg.size, cv.IPL_DEPTH_8U, 1)
        cv.Dilate(cvBinary, cvBinary, None, 3)
        cv.Erode(cvBinary, cvBinary, None, 1)
        self._filter = rectFilter
        
        return self._getRects(cvBinary, annotateImg, minArea) 
        
    def _getRects(self, cvBinary, annotateImg, minArea=400):
        '''
        Finds the external contours in binary image and then annotates
        the annotateImg with the bounding boxes of those contours
        @param cvBinary: Binary image containing "blobs" to detect contours on.
            Of type OpenCV single channel 8 bit image.
        @param annotateImg: The pv.Image on which to annotate the detection 
        @param minArea: Any contours with less than this area in pixels will be dropped
        '''
        cvdst = cv.CloneImage(cvBinary)  #because cv.FindContours may alter source image
        contours = cv.FindContours(cvdst, cv.CreateMemStorage(), cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)
        
        #create a list of the top-level contours found in the contours (cv.Seq) structure
        rects = []
        if len(contours) < 1: return(annotateImg,rects)
        seq = contours
        while not (seq == None):
            (x, y, w, h) = cv.BoundingRect(seq) 
            if (cv.ContourArea(seq) > minArea):
                r = pv.Rect(x,y,w,h)
                rects.append(r)
            seq = seq.h_next()
        
        if self._filter != None:
            rects = self._filter(rects)
        
        #print "Found %d rects"%len(rects)
        if annotateImg != None:
            #draw bounding box in yellow
            for r in rects:
                annotateImg.annotateRect(r,"yellow")
            #draw contours in green
            cvimg = pv.Image(annotateImg.asAnnotated()).asOpenCV()
            cv.DrawContours(cvimg, contours, cv.RGB(0, 255, 0), cv.RGB(255,0,0), 2)
            return (pv.Image(cvimg), rects)
        else:
            return(None,rects) 