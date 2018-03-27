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

'''
Created on Dec 10, 2009

@author: bolme
'''
import pyvision as pv
#import cv2
import numpy as np


class OpticalFlow:
    '''
    This class tracks the motion of the camera.  This can be used for things like
    video stablization or to better understand camera motion across multiple
    frames.
    
    This uses the LK optical flow algorithm to track point correspondences from one
    frame to another.  
    
    It can output the frame to frame camera motion as a homography.
    '''
    
    def __init__(self,tile_size="ORIG"):
        '''
        Create the camera tracker
        
        @param tile_size: Frames will be resized to these dimensions before tracking.
        '''
        self.frame_size = None
        self.prev_frame = None
        self.tile_size = tile_size
        self.tracks = []
        self.bad_points = []
        
        # Divide the image in to gridXgrid regions
        self.grid = 5
        
        # Require a minimum number of points_b per grid
        self.min_points = 5
        
        self.n = 0
        
        self.homography = None
        self.homography_rev = None
        
        self.prev_input = None
        
    def update(self,frame):
        '''
        This tracks the points_b to the next frame using the LK tracker.
        Add more good points_b to track.
        
        @param frame: update optical flow for the frame.
        @type frame: pv.Image
        '''
        if self.frame_size == None:
            self.frame_size = frame.size
            
        if self.prev_frame == None:
            #Initialize the tracker
            
            # Pick a frame size
            if self.tile_size == "AUTO":
                # Rescale the image so that the largest demenion is between 256 and 512
                w,h = frame.size
                n = max(w,h)
                while n > 512:
                    n = n/2
                
                scale = n/float(max(w,h))
                
                w = int(round(scale*w))
                h = int(round(scale*h))
                
                self.tile_size = (w,h)
                
                
            if self.tile_size == "ORIG":
                self.tile_size = frame.size

            # allocate memory
            w,h = self.tile_size
            self.frame = cv.CreateImage((w,h), 8, 1)
            self.prev_frame = cv.CreateImage((w,h), 8, 1) 
            
            # Resize the frame
            cvim = frame.asOpenCVBW()
            cv.Resize(cvim,self.frame)
            
            # Find good features to track
            self._selectTrackingPoints(self.frame)
            
            # Init transform
            
            
        else:
            # Resize the frame
            cvim = frame.asOpenCVBW()
            cv.Resize(cvim,self.frame)
        
        
            # Track the points_b (optical flow)
            new_tracks = self._opticalFlow()
        
            
            # Compute the transform
            assert len(new_tracks) == len(self.tracks)
            n = len(new_tracks)
            src_points = cv.CreateMat(n,2,cv.CV_32F)
            dst_points = cv.CreateMat(n,2,cv.CV_32F)
    
            # Copy data into matricies
            for i in range(len(new_tracks)):
                src_points[i,0] = self.tracks[i].X()
                src_points[i,1] = self.tracks[i].Y()

                dst_points[i,0] = new_tracks[i].X()
                dst_points[i,1] = new_tracks[i].Y()
    
            # Create homography matrix
            self.homography = cv.CreateMat(3,3,cv.CV_32F)
            self.homography_rev = cv.CreateMat(3,3,cv.CV_32F)
            mask = cv.CreateMat(1,n,cv.CV_8U)
            cv.FindHomography(dst_points,src_points,self.homography_rev,cv.CV_LMEDS,1.5,mask)
            cv.FindHomography(src_points,dst_points,self.homography,cv.CV_LMEDS,1.5,mask)
            
            # Drop bad points_b
            self.tracks = []
            self.bad_points = []
            for i in range(n):
                if mask[0,i]:
                    self.tracks.append(new_tracks[i])
                else:
                    self.bad_points.append(new_tracks[i])
                    
            self.n = len(self.tracks)
            
            # Add new points_b
            self._selectTrackingPoints(self.frame)
            
            self.prev_input.to_next = self.asHomography(forward = False)
            frame.to_prev = self.asHomography(forward = True)
            
        self.prev_input = frame

        cv.Copy(self.frame,self.prev_frame)
            
    def _selectTrackingPoints(self,frame):
        '''
        This uses the OpenCV get good features to track to initialize a set of tracking points_b.
        '''
        quality = 0.01
        min_distance = 15
        
        w,h = self.tile_size
        tw = w//self.grid
        th = h//self.grid

        for i in range(self.grid):
            for j in range(self.grid):
                ul = pv.Point(i*tw,j*th)
                rect = pv.Rect(i*tw,j*th,tw,th)
                count = 0
                for pt in self.tracks:
                    if rect.containsPoint(pt):
                        count += 1
                    
                if count < self.min_points:
                    gray = cv.CreateImage ((tw,th), 8, 1)
                    
                    faceim = cv.GetSubRect(frame, rect.asOpenCV())
                    cv.Resize(faceim,gray)

                    eig = cv.CreateImage ((tw,th), 32, 1)
                    temp = cv.CreateImage ((tw,th), 32, 1)
                
                    # search the good points_b
                    points_b = cv.GoodFeaturesToTrack (gray, eig, temp, 2*self.min_points, quality, min_distance, None, 3, 0, 0.04)
                    
                    for pt in points_b:
                        self.tracks.append(ul+pv.Point(pt))
                                        
        
    def _opticalFlow(self):
        '''
        Compute the optical flow between frames using cv.CalcOpticalFlow
        
        @returns: a list of tracks for the new image
        @rtype: list of pv.Point()
        '''
        flags = 0
        
        grey = self.frame
        prev_grey = self.prev_frame
    
        pyramid = cv.CreateImage (cv.GetSize (grey), 8, 1)
        prev_pyramid = cv.CreateImage (cv.GetSize (grey), 8, 1)
    
        cv_points = []
        for each in self.tracks:
            cv_points.append((each.X(),each.Y()))
        
        points_b, _, _,= cv.CalcOpticalFlowPyrLK (
                    prev_grey, 
                    grey, 
                    prev_pyramid, 
                    pyramid,
                    cv_points,
                    (5,5),
                    3,#pyr number
                    (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 10, 0.01),
                    flags)
                
        result = []
        for pt in points_b:
            result.append(pv.Point(pt))
    
        return result

                
    def asHomography(self,forward=True):
        '''
        Get the transformation as a homography.
        
        @keyword forward: switch between the forward and reverse transform.
        @ktype forward: True|False
        '''
        fw,fh = self.frame_size
        tw,th = self.frame_size
        if forward:
            if self.homography == None:
                matrix = np.eye(3)
            else:
                matrix = np.linalg.inv(pv.OpenCVToNumpy(self.homography))
            matrix = np.dot(np.diag([tw/fw,th/fh,1.0]),matrix)
            perspective = pv.PerspectiveTransform(matrix,self.frame_size)
        else:
            if self.homography_rev == None:
                matrix = np.eye(3)
            else:
                matrix = np.linalg.inv(pv.OpenCVToNumpy(self.homography_rev))
            matrix = np.dot(np.diag([tw/fw,th/fh,1.0]),matrix)
            perspective = pv.PerspectiveTransform(matrix,self.frame_size)
            
        return perspective

        
    def annotateFrame(self,frame,color='white'):
        '''
        Renders optical flow information to the frame.
        
        @param frame: the frame that will be annotated
        @type frame: pv.Image
        @keyword type:
        @ktype type: "TRACKING" 
        
        '''
        w,h = self.tile_size
        rect = pv.Rect(0,0,w,h)
        affine = pv.AffineFromRect(rect,frame.size)
        for pt in affine.transformPoints(self.tracks[:self.n]):
            frame.annotatePoint(pt,color=color)

        for pt in affine.transformPoints(self.tracks[self.n:]):
            frame.annotatePoint(pt,color='red')
            
        for pt in affine.transformPoints(self.bad_points):
            frame.annotatePoint(pt,color='gray')
            
        
        if self.homography != None:
            matrix = pv.OpenCVToNumpy(self.homography)
            perspective = pv.PerspectiveTransform(matrix,frame.size)

            for pt in self.tracks[:self.n]:
                # Transform the point multiple times to amplify the track 
                # for visualization
                old = perspective.invertPoint(pt)
                old = perspective.invertPoint(old)
                old = perspective.invertPoint(old)
                frame.annotateLine(pt,old,color=color)
        


