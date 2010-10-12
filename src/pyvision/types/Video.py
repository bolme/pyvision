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


# TODO: This will probably not yet work with OpenCV 2.0

import time
import os
import pyvision as pv
import cv
#from scipy import weave

# TODO: The default camera on linux appears to be zero and 1 on MacOS
# Video capture is an alterative for windows http://videocapture.sourceforge.net/
# An option for linux http://code.google.com/p/python-video4linux2/
# On linux it may be possible to use something like v4lctl to capture in a seperate process.

class Webcam:
    def __init__(self,camera_num=0,size=(640,480)):

        self.cv_capture = cv.CreateCameraCapture( camera_num )        
        
        self.size = size
    
    def query(self):
        '''
        The returned image also include a field named orig_frame which returns 
        the original image returned before rescaling.
        
        @returns: the frame rescaled to a given size.
        '''
        # TODO: Video capture is unreliable under linux.  This may just be a timing issue when running under parallels.
        frame = cv.QueryFrame( self.cv_capture )
        im = pv.Image(self.resize(frame))
        im.orig_frame = pv.Image(frame)
        im.capture_time = time.time()
        return im
    
    def grab(self):
        return cv.GrabFrame( self.cv_capture );
    
    def retrieve(self):
        '''
        The returned image also include a field named orig_frame which returns 
        the original image returned before rescaling.
        
        @returns: the frame rescaled to a given size.
        '''
        frame = cv.RetrieveFrame( self.cv_capture );
        im = pv.Image(self.resize(frame))
        im.orig_frame = pv.Image(frame)
        return im
        
    def resize(self,frame):
        if self.size == None:
            return frame
        else:
            depth = frame.depth
            channels = frame.nChannels
            w,h = self.size
            resized = cv.CreateImage( (w,h), depth, channels )
            cv.Resize( frame, resized, cv.CV_INTER_NN )
            return resized

class Video:
    def __init__(self,filename,size=None):
        self.filename = filename
        self.cv_capture = cv.CaptureFromFile( filename );
        #print self.cv_capture, self.cv_capture.__hash__, dir(self.cv_capture), repr(self.cv_capture)
        self.size = size
        #print filename
        #print self.size
        #self.n_frames = cv.GetCaptureProperty(self.cv_capture,cv.CV_CAP_PROP_FRAME_COUNT)
        #print cv.GetCaptureProperty(self.cv_capture,cv.CV_CAP_PROP_FRAME_WIDTH)
        #print cv.GetCaptureProperty(self.cv_capture,cv.CV_CAP_PROP_FRAME_HEIGHT)
        #print cv.GetCaptureProperty(self.cv_capture,cv.CV_CAP_PROP_POS_FRAMES)
        #print cv.GetCaptureProperty(self.cv_capture,cv.CV_CAP_PROP_FRAME_COUNT)
        #while True:
        #    print cv.QueryFrame(self.cv_capture)
        #    print cv.GetCaptureProperty(self.cv_capture,cv.CV_CAP_PROP_POS_FRAMES),
        #    print cv.GetCaptureProperty(self.cv_capture,cv.CV_CAP_PROP_POS_AVI_RATIO),
        #    print cv.GetCaptureProperty(self.cv_capture,cv.CV_CAP_PROP_FRAME_COUNT)
        #print self.n_frames
        self.current_frame = 0
        
    #def __del__(self):
        #opencv.highgui.cvReleaseCapture(self.cv_capture)
        

        # cvReleaseCapture interface does not work so use weave this may be fixed in release 1570
        # TODO: This should be removed when the opencv bug is fixed
        #capture = self.cv_capture.__int__()
        #cv.ReleaseCapture(self.cv_capture)
        #weave.inline(
        #    '''
        #    CvCapture* tmp = (CvCapture*) capture;
        #    cvReleaseCapture(&tmp);
        #    ''',
        #    arg_names=['capture'],
        #    type_converters=weave.converters.blitz,
        #    include_dirs=['/usr/local/include'],
        #    headers=['<opencv/cv.h>','<opencv/highgui.h>'],
        #    library_dirs=['/usr/local/lib'],
        #    libraries=['cv','highgui']
        #)

    def query(self):
        if self.current_frame > 0 and cv.GetCaptureProperty(self.cv_capture,cv.CV_CAP_PROP_POS_AVI_RATIO) == 1.0:
            return None
        self.current_frame += 1
        frame = cv.QueryFrame( self.cv_capture );
        return pv.Image(self.resize(frame))
    
    #def grab(self):
    #    return cv.GrabFrame( self.cv_capture );
    
    #def retrieve(self):
    #    frame = cv.RetrieveFrame( self.cv_capture );
    #    return pv.Image(self.resize(frame))
        
    def resize(self,frame):
        if self.size == None:
            return frame
        else:
            depth = frame.depth
            channels = frame.channels
            w,h = self.size
            resized = cv.CreateImage( (w,h), depth, channels )
            cv.Resize( frame, resized, cv.CV_INTER_LINEAR )
            return resized
    
    def __iter__(self):
        ''' Return an iterator for this video '''
        return Video(self.filename,self.size)
        
    def next(self):
        frame = self.query()
        if frame == None:
            raise StopIteration("End of video sequence")
        return frame
        
                
        
class FfmpegIn:
    # TODO: there may be a bug with the popen interface
    
    def __init__(self,filename,size=None,aspect=None,options=""):
        self.filename = filename
        self.size = size
        self.aspect = aspect
        
        # Open a pipe
        args = "/opt/local/bin/ffmpeg -i %s %s -f yuv4mpegpipe - "%(filename,options)
        #print args
        
        self.stdin, self.stdout, self.stderr = os.popen3(args)
        #popen = subprocess.Popen(args,executable="/opt/local/bin/ffmpeg")
        
        line = self.stdout.readline()
        print line
        #self.stdout.seek(0,os.SEEK_CUR)
        
        format,w,h,f,t1,aspect,t2,t3 = line.split()
        
        # I am not sure what all this means but I am checking it anyway
        assert format=='YUV4MPEG2'
        #assert t1=='Ip'
        assert t2=='C420mpeg2'
        assert t3=='XYSCSS=420MPEG2'

        # get the width and height
        assert w[0] == "W"
        assert h[0] == "H"
        
        self.w = int(w[1:])
        self.h = int(h[1:])
        
        # Create frame caches        
        if size == None and self.aspect != None:
            h = self.h
            w = int(round(self.aspect*h))
            size = (w,h)
            #print size
        
        self.size = size
        
        self.frame_y = cv.CreateImage( (self.w,self.h), cv.IPL_DEPTH_8U, 1 )
        self.frame_u2 = cv.CreateImage( (self.w/2,self.h/2), cv.IPL_DEPTH_8U, 1 )
        self.frame_v2 = cv.CreateImage( (self.w/2,self.h/2), cv.IPL_DEPTH_8U, 1 )

        self.frame_u = cv.CreateImage( (self.w,self.h), cv.IPL_DEPTH_8U, 1 )
        self.frame_v = cv.CreateImage( (self.w,self.h), cv.IPL_DEPTH_8U, 1 )
        self.frame_col = cv.CreateImage( (self.w,self.h), cv.IPL_DEPTH_8U, 3 )

        
        if self.size != None:
            w,h = self.size
            self.frame_resized = cv.CreateImage( (w,h),cv.IPL_DEPTH_8U,3)

        
        
    def frame(self):
        line = self.stdout.readline()
        #print line
        #print self.w,self.h
        y = self.stdout.read(self.w*self.h)
        u = self.stdout.read(self.w*self.h/4)
        v = self.stdout.read(self.w*self.h/4)
        if len(y) < self.w*self.h:
            raise EOFError
        
        self.frame_y.imageData=y
        self.frame_u2.imageData=u
        self.frame_v2.imageData=v

        cv.Resize(self.frame_u2,self.frame_u)
        cv.Resize(self.frame_v2,self.frame_v)
        
        cv.Merge(self.frame_y,self.frame_u,self.frame_v,None,self.frame_col)
        cv.CvtColor(self.frame_col,self.frame_col,cv.CV_YCrCb2RGB)
        
        out = self.frame_col
        
        if self.size != None:
            cv.Resize(self.frame_col,self.frame_resized)
            out = self.frame_resized

        return pv.Image(self.frame_y),pv.Image(self.frame_u),pv.Image(self.frame_v),pv.Image(out)
        
class VideoFromImages:
    '''
    This class allows the user to treat a directory of images as a video. It is assumed that
    the files in the directory are named as follows:
    {prefix}{num}.{ext}
    where
     prefix is any string that is constant for all the files,
     ext is the file extension/type like jpg, png, etc.
     num is a zero-padded number like 0001, 0002, ...
         note: the amount of padded zeros is the minimum required based on the length
         (num frames) in the video. So if you only had 120 frames, then it would be 001, 002,...120.
         We assume the frames are sequential with no gaps, and start at number startnum (with 
         appropriate padding). If you have extra zero padding, then you can put the prefix zeros
         as part of the prefix string.
    '''
    def __init__(self,dirname,numframes,prefix="frame",ext="jpg", startnum=1, size=None):
        self.dirname = dirname
        self.numframes = numframes
        self.prefix = prefix
        self.ext = ext
        self.size = size  #the optional width,height to resize the input frames
        self.startnum = startnum
        self.current_frame = startnum  #we start at frame 1 by default
        
        #check that directory exists
        if not os.path.exists(dirname):
            print "Error. Directory: %s does not exist."%dirname
            raise IOError
        
    def query(self):      
        if self.current_frame <= self.numframes:  
            pad = len(str(self.numframes))
            num = str(self.current_frame).zfill(pad)
            filename = self.prefix + num + "." + self.ext
            f = os.path.join(self.dirname, filename)
            frame = pv.Image(f)
            self.current_frame += 1
            return(self.resize(frame))
        else:
            return None
       
    def resize(self,frame):
        if self.size == None:
            return frame
        else:
            depth = frame.depth
            channels = frame.channels
            w,h = self.size
            resized = cv.CreateImage( (w,h), depth, channels )
            cv.Resize( frame, resized, cv.CV_INTER_LINEAR )
            return resized
                
    def next(self):
        frame = self.query()
        if frame == None:
            raise StopIteration("End of video sequence")
        return frame
        
    def __iter__(self):
        ''' Return an iterator for this video '''
        return VideoFromImages(self.dirname, self.numframes, self.prefix, self.ext, self.startnum, self.size) 
        
        
    