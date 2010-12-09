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
import numpy as np

#TODO: Consider Renaming the following constants with a class-specific prefix
# so that they are easier to find in code completion. Recommend
# MD_BOUNDING_RECTS AND MD_STANDARDIZED_RECTS. Change will break existing code.
BOUNDING_RECTS     = "BOUNDING_RECTS"
STANDARDIZED_RECTS = "STANDARDIZED_RECTS"


class MotionDetector(object):
    '''
    Uses background subtraction from an image buffer to detect
    areas of motion in a video.
    
    The general process is to update the image buffer and then
    call the MotionDetector's detect() method.
    '''
    
    def __init__(self, imageBuff=None, thresh=20, method=BG_SUBTRACT_AMF, minArea=400, 
                 rectFilter=None, buffSize=5, soft_thresh = False,rect_type=BOUNDING_RECTS,rect_sigma=2.0):
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
        @param soft_thresh: Specify if the background subtraction method should
          use a soft threshold, in which case the returned mask is no longer a binary
          image, but represents weighted values. NOTE: NOT CURRENTLY IMPLEMENTED. 
          SOFT THRESHOLD WILL BE IGNORED, HARD THRESHOLD ONLY IN THIS VERSION.
        @note: Until the image buffer is full, the result of the motion detection will be
          nothing. See documentation on the detect(img) method of this class.
        '''
        #initialize object variables
        self._fgMask = None        
        self._minArea = minArea
        self._filter = rectFilter
        self._threshold = 20
        self._softThreshold = False #soft_thresh
        
        if imageBuff == None:
            self._imageBuff = pv.ImageBuffer(N=buffSize)
        else:
            self._imageBuff = imageBuff
        
        self._method = method      
        self._bgSubtract = None  #can't initialize until buffer is full...so done in detect()  
        self._contours = []
        self._annotateImg = None
        self._rect_type = rect_type
        self._rect_sigma = rect_sigma
        
    def _initBGSubtract(self):
        if self._method==BG_SUBTRACT_FD:
            self._bgSubtract = pv.FrameDifferencer(self._imageBuff, self._threshold, 
                                                   soft_thresh = self._softThreshold)
        elif self._method==BG_SUBTRACT_MCFD:
            self._bgSubtract = pv.MotionCompensatedFrameDifferencer(self._imageBuff, self._threshold, 
                                                   soft_thresh = self._softThreshold)
        elif self._method==BG_SUBTRACT_MF:
            self._bgSubtract = pv.MedianFilter(self._imageBuff, self._threshold,
                                               soft_thresh = self._softThreshold)
        elif self._method==BG_SUBTRACT_AMF:
            self._bgSubtract = pv.ApproximateMedianFilter(self._imageBuff, self._threshold,
                                                          soft_thresh = self._softThreshold)
        else:
            raise ValueError("Unknown Background Subtraction Method specified.")
                  
    def _computeContours(self):
        cvMask = self._fgMask.asOpenCVBW()
        cvdst = cv.CloneImage(cvMask)  #because cv.FindContours may alter source image
        contours = cv.FindContours(cvdst, cv.CreateMemStorage(), cv.CV_RETR_EXTERNAL , cv.CV_CHAIN_APPROX_SIMPLE)
        self._contours = contours
            
    def _computeConvexHulls(self):
        hulls = []
        seq = self._contours
        while not (seq == None) and len(seq) != 0:
            cvxHull = cv.ConvexHull2(seq, cv.CreateMemStorage(), return_points=True)
            hulls.append(cvxHull)
            seq = seq.h_next()           
            
        self._convexHulls = hulls
        
            
    def detect(self, img, ConvexHulls=False):
        '''
        You call this method to update detection results, given the new
        image in the stream. After updating detection results, use one
        of the get*() methods, such as getRects() to see the results in the
        appropriate format.
        
        @param img: A pv.Image() to be added to the buffer as the most recent image,
        and that triggers the new motion detection. Note that, depending on the
        background subtraction method, this may not be the "key frame" for the 
        detection. The Frame Differencer returns a background model based on the
        middle image, but Median and Approx. Median Filters return a background
        model based on the most recent (last) image in the buffer. 
        
        @param ConvexHulls: If true, then the detected foreground pixels are
        grouped into convex hulls, which can have the effect of removing internal
        "holes" in the detection.
        
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

        #update current annotation image from buffer, as appropriate for
        # the different methods
        if self._method==BG_SUBTRACT_FD:
            self._annotateImg = self._imageBuff.getMiddle()
        if self._method==BG_SUBTRACT_MCFD:
            self._annotateImg = self._imageBuff.getMiddle()
        elif self._method==BG_SUBTRACT_MF:
            self._annotateImg = self._imageBuff.getLast()
        elif self._method==BG_SUBTRACT_AMF:
            self._annotateImg = self._imageBuff.getLast()

        mask = self._bgSubtract.getForegroundMask()
#        if self._softThreshold:
#            cvWeights = mask.asOpenCVBW()
#            scale = (1.0/255.0)  #because weights are 0-255 in mask image
#            cvCurImg = self._annotateImg.copy().asOpenCVBW()
#            cvDst = cv.CreateImage(cv.GetSize(cvWeights), cv.IPL_DEPTH_8U, 1)
#            cv.Mul(cvWeights, cvCurImg, cvDst, scale)
#            cv.Smooth(cvDst, cvDst)
#            #update the foreground mask
#            self._fgMask = pv.Image(cvDst) 
#        else:    

        cvBinary = mask.asOpenCVBW()
        cv.Smooth(cvBinary, cvBinary)
        cv.Dilate(cvBinary, cvBinary, None, 3)
        cv.Erode(cvBinary, cvBinary, None, 1)
        
        #update the foreground mask
        self._fgMask = pv.Image(cvBinary)
        
        #update the detected foreground contours
        self._computeContours()
        self._computeConvexHulls()
            
        if ConvexHulls:
            for hull in self._convexHulls:
                cv.FillConvexPoly(cvBinary, hull, cv.RGB(255,255,255))
            #k = cv.CreateStructuringElementEx(15, 15, 7, 7, cv.CV_SHAPE_RECT)
            #cv.Dilate(mask, mask, element=k, iterations=1)
            
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
    
    def getForegroundPixels(self, bgcolor=None):
        '''
        @param bgcolor: The background color to use. Specify as an (R,G,B) tuple.
        Specify None for a blank/black background.
        @return: The full color foreground pixels on either a blank (black)
        background, or on a background color specified by the user.
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
        if bgcolor==None:
            cv.SetZero(dest)
        else:
            cv.Set(dest, cv.RGB(*bgcolor))

        cv.Copy(image,dest,mask) #copy only pixels from image where mask != 0               
        return pv.Image(dest)
            
    def getRects(self): 
        '''
        @return: the bounding boxes of the external contours of the foreground mask. The
        boxes will either be the bounding rectangles of the contours, or a box fitted to
        the contours based on the center of mass and n-sigma deviations in x and y. This
        preference is selected when initializing the MotionDetector object.
        
        @note: You must call detect() before getRects() to see updated results.
        '''
        if self._rect_type == BOUNDING_RECTS:
            return self.getBoundingRects()
        
        elif self._rect_type == STANDARDIZED_RECTS:
            return self.getStandardizedRects()
        
        else:
            raise ValueError("Unknown rect type: "+self._rect_type)
        
            
    def getBoundingRects(self):
        '''
        @return: the bounding boxes of the external contours of the foreground mask.
        @note: You must call detect() before getBoundingRects() to see updated results.
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
    
    def getStandardizedRects(self):
        '''
        @return: the boxes centered on the target center of mass +- n_sigma*std
        @note: You must call detect() before getStandardizedRects() to see updated results.
        '''
        #create a list of the top-level contours found in the contours (cv.Seq) structure
        rects = []
        if len(self._contours) < 1: return(rects)
        seq = self._contours
        while not (seq == None):
            (x, y, w, h) = cv.BoundingRect(seq) 
            if (cv.ContourArea(seq) > self._minArea): # and  self._filter(rect)
                r = pv.Rect(x,y,w,h)
                moments = cv.Moments(seq)
                m_0_0 = cv.GetSpatialMoment(moments, 0, 0)
                m_0_1 = cv.GetSpatialMoment(moments, 0, 1)
                m_1_0 = cv.GetSpatialMoment(moments, 1, 0)
                mu_2_0 = cv.GetCentralMoment(moments, 2, 0)
                mu_0_2 = cv.GetCentralMoment(moments, 0, 2)
                
                cx = m_1_0/m_0_0
                cy = m_0_1/m_0_0
                w = 2.0*self._rect_sigma*np.sqrt(mu_2_0/m_0_0)
                h = 2.0*self._rect_sigma*np.sqrt(mu_0_2/m_0_0)
                
                r = pv.CenteredRect(cx,cy,w,h)

                rects.append(r)
            seq = seq.h_next()
        
        if self._filter != None:
            rects = self._filter(rects)
        
        return rects
    
    def getPolygons(self,return_all=False):
        '''
        @param return_all: return all contours regardless of min area.
        @return: the polygon contours of the foreground mask. The polygons are
        compatible with pv.Image annotatePolygon() method.
        @note: You must call detect() before getPolygons() to see updated results.
        '''
        #create a list of the top-level contours found in the contours (cv.Seq) structure
        polys = []
        if len(self._contours) < 1: return(polys)
        seq = self._contours
        while not (seq == None):

            if return_all or (cv.ContourArea(seq) > self._minArea):
                poly = [ pv.Point(*each) for each in seq ]
                poly.append(poly[0])
                
                polys.append(poly)
                
            seq = seq.h_next()
                
        return polys
    
    def getConvexHulls(self):
        '''
        @return: the convex hulls of the contours of the foreground mask.
        @note: You must call detect() before getConvexHulls() to see updated results.
        '''
        return self._convexHulls
        
    
    def getAnnotatedImage(self, showRects=True, showContours=False, 
                          showConvexHulls=False, showFlow=False):
        '''
        @return: the annotation image with selected objects drawn upon it. showFlow will
        only work if the BG subtraction method was MCFD.
        @note: You must call detect() prior to getAnnotatedImage() to see updated results.
        '''
        rects = self.getRects()
        outImg = self._annotateImg.copy()  #deep copy, so can freely modify the copy
        
        
        #draw optical flow information in white
        if showFlow and (self._method == pv.BG_SUBTRACT_MCFD):
            flow = self._bgSubtract.getOpticalFlow()
            flow.annotateFrame(outImg)
            
        if showContours or showConvexHulls:
            cvimg = outImg.asOpenCV()
        
        #draw contours in green   
        if showContours:
            cv.DrawContours(cvimg, self._contours, cv.RGB(0, 255, 0), cv.RGB(255,0,0), 2)
        
        #draw hulls in cyan    
        if showConvexHulls:
            cv.PolyLine(cvimg, self._convexHulls, True, cv.RGB(0,255,255))
        
        #draw bounding box in yellow
        if showRects:
            for r in rects:
                outImg.annotateRect(r,"yellow")
        
        return outImg        
        
    def annotateFrame(self, key_frame, rect_color='yellow', 
                      contour_color='#00FF00', flow_color='white'):
        '''
        Draws detection results on an image (key_frame) specified by the user. Specify
        None as the color for any aspect you wish not drawn.
        @return: Renders annotations onto key frame that shows detection information.
        @note: You must call detect() prior to annotateFrame() to see updated results.
        @note: Optical flow is only shown if method was MCFD
        '''
        #key_frame = md.getKeyFrame()
        
        if key_frame != None:
            
            if contour_color != None:
                for poly in self.getPolygons():
                    key_frame.annotatePolygon(poly,color=contour_color,width=1)
                    
            if rect_color != None:   
                for rect in self.getRects():
                    key_frame.annotatePolygon(rect.asPolygon(),width=2,color=rect_color)
                
            if (flow_color != None) and (self._method == pv.BG_SUBTRACT_MCFD):
                flow = self._bgSubtract.getOpticalFlow()
                flow.annotateFrame(key_frame, type="TRACKING", color=flow_color)
                
            #for rect in rects:
            #    key_frame.annotatePolygon(rect.asPolygon(),width=2)
            #    key_frame.annotatePoint(rect.center())                
        
            #ilog(key_frame)

        
    def getForegroundTiles(self, bgcolor=None):
        '''
        @param bgcolor: The background color to use. Specify as an (R,G,B) tuple.
        Specify None for a blank/black background.
        @return: a list of "tiles", where each tile is a small pv.Image
        representing the clipped area of the annotationImg based on
        the motion detection. Only the foreground pixels are copied, so
        the result are tiles with full-color foreground pixels on the
        specified background color (black by default).
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
        if bgcolor==None:
            cv.SetZero(dest)
        else:
            cv.Set(dest, cv.RGB(*bgcolor))
            
        cv.Copy(image,dest,mask) #copy only pixels from image where mask != 0
        dst = pv.Image(dest)
        
        rects = self.getRects()
        
        tiles = []
        for r in rects:
            #for every rectangle, crop from dest image
            t = dst.crop(r)
            tiles.append(t)
            
        return tiles