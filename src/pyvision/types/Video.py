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

import opencv
import opencv.highgui
import time
#import highgui

from pyvision.types.Image import Image
from pyvision.edge.canny import canny
from pyvision.analysis.ImageLog import ImageLog

class Webcam:
    def __init__(self,camera_num=1,size=(640,480)):
        self.cv_capture = opencv.highgui.cvCreateCameraCapture( camera_num )
        self.size = size
    
    def query(self):
        '''
        The returned image also include a field named orig_frame which returns 
        the original image returned before rescaling.
        
        @returns: the frame rescaled to a given size.
        '''
        frame = opencv.highgui.cvQueryFrame( self.cv_capture );
        im = Image(self.resize(frame))
        im.orig_frame = Image(frame)
        im.capture_time = time.time()
        return im
    
    def grab(self):
        return opencv.highgui.cvGrabFrame( self.cv_capture );
    
    def retrieve(self):
        '''
        The returned image also include a field named orig_frame which returns 
        the original image returned before rescaling.
        
        @returns: the frame rescaled to a given size.
        '''
        frame = opencv.highgui.cvRetrieveFrame( self.cv_capture );
        im = Image(self.resize(frame))
        im.orig_frame = Image(frame)
        return im
        
    def resize(self,frame):
        if self.size == None:
            return frame
        else:
            depth = frame.depth
            channels = frame.nChannels
            w,h = self.size
            resized = opencv.cvCreateImage( opencv.cvSize(w,h), depth, channels )
            opencv.cvResize( frame, resized, opencv.CV_INTER_NN )
            return resized

class Video:
    def __init__(self,filename,size=None):
        self.filename = filename
        self.cv_capture = opencv.highgui.cvCreateFileCapture( filename );
        self.size = size
        self.n_frames = opencv.highgui.cvGetCaptureProperty(self.cv_capture,opencv.highgui.CV_CAP_PROP_FRAME_COUNT)
        self.current_frame = 0
        
    def query(self):
        if self.current_frame >= self.n_frames:
            return None
        self.current_frame += 1
        frame = opencv.highgui.cvQueryFrame( self.cv_capture );
        return Image(self.resize(frame))
    
    def grab(self):
        return opencv.highgui.cvGrabFrame( self.cv_capture );
    
    def retrieve(self):
        frame = opencv.highgui.cvRetrieveFrame( self.cv_capture );
        return Image(self.resize(frame))
        
    def resize(self,frame):
        if self.size == None:
            return frame
        else:
            depth = frame.depth
            channels = frame.nChannels
            w,h = self.size
            resized = opencv.cvCreateImage( opencv.cvSize(w,h), depth, channels )
            opencv.cvResize( frame, resized, opencv.CV_INTER_LINEAR )
            return resized
    
    def __iter__(self):
        ''' Return an iterator for this video '''
        return Video(self.filename,self.size)
        
    def next(self):
        frame = self.query()
        if frame == None:
            raise StopIteration("End of video sequence")
        return frame
        
                
        
