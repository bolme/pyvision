# PyVision License
#
# Copyright (c) 2006-2008 David S. Bolme, Stephen O'Hara
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
        #cv.SetCaptureProperty(self.cv_capture,cv.CV_CAP_PROP_FRAME_WIDTH,1600.0)
        #cv.SetCaptureProperty(self.cv_capture,cv.CV_CAP_PROP_FRAME_HEIGHT,1200.0)
        #print cv.GetCaptureProperty(self.cv_capture,cv.CV_CAP_PROP_FRAME_WIDTH)
        # print cv.GetCaptureProperty(self.cv_capture,cv.CV_CAP_PROP_FRAME_HEIGHT)
        
        
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
        self._numframes = cv.GetCaptureProperty(self.cv_capture,cv.CV_CAP_PROP_FRAME_COUNT)
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
        frame = cv.QueryFrame( self.cv_capture )
        if frame == None:
            raise StopIteration("End of video sequence")
        self.current_frame += 1
        frame = cv.CloneImage(frame);
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
        
    def play(self, window="Input", pos=None, delay=20, imageBuffer=None, onNewFrame=None, onNewFrame_kwargs={} ):
        '''
        Plays the video, calling the onNewFrame function after loading each
         frame from the video. The user may interrupt video playback by
         hitting (sometimes repeatedly) the spacebar, upon which they are
         given a text menu in the console to abort program, quit playback,
         continue playback, or step to the next frame.
        @param window: The window name used to display the video. If None,
        then the video won't be shown, but onNewFrame will be called at
        each frame.
        @param pos: A tuple (x,y) where the output window should be located
        on the users screen. None indicates default openCV positioning algorithm
        will be used.
        @param delay: The delay in ms between window updates. This allows the user
        to control the playback frame rate. A value of 0 indicates that the video
        will wait for keyboard input prior to advancing to the next frame. This
        delay is used by the pauseAndPlay interface, so it will affect the rate
        at which onNewFrame is called as well.
        @param imageBuffer: An optional pyvision ImageBuffer object to contain the
        most recent frames. This is useful if a buffer is required for background
        subtraction, for example. The buffer contents is directly modified each
        time a new image is captured from the video, and a reference to the buffer
        is passed to the onNewFrame function (defined below).
        @param onNewFrame: A python callable object (function) with a
        signature of 'foo( pvImage, frameNum, key=None, buffer=None )', where key is
        the key pressed by the user (if any) during the pauseAndPlay interface, and
        buffer is a reference to the optional image buffer provided to the play method.
        @param onNewFrame_kwargs: Optional keyword arguments that should be passed
        onto the onNewFrame function.
        @return: The final frame number of the video, or the frame number at which
        the user terminated playback using the 'q'uit option.
        '''
        fn = -1
        vid = self
        if delay==0:
            delayObj = {'wait_time':20, 'current_state':'PAUSED'}
        else:
            delayObj = {'wait_time':delay, 'current_state':'PLAYING'}
        key=''
        for fn, img in enumerate(vid):
            if imageBuffer != None:
                imageBuffer.add(img)
            
            if window != None:
                pt = pv.Point(10, 10)
                img.annotateLabel(label="Frame: %d"%(fn+1), point=pt, color="white", background="black")
                img.show(window=window,pos=pos)
            
            if onNewFrame != None:
                onNewFrame( img, fn, key=key, buffer=imageBuffer, **onNewFrame_kwargs )
                
            key = self._pauseAndPlay(delayObj)
            if key == 'q':
                #user selected quit playback
                return(fn)
        
        return(fn)
    
    def _pauseAndPlay(self,delayObj={'wait_time':20, 'current_state':'PLAYING'}):
        '''
        This function is intended to be used in the playback loop of a video.
        It allows the user to interrupt the playback to pause the video, to 
        step through it one frame at a time, and to register other keys/commands
        that the user may select.
        @param delayObj: The "delay object", which is just a dictionary that
        specifies the wait_time (the delay in ms between frames), and
        the current_state of either 'PLAYING' or 'PAUSED'
        '''
        state = delayObj['current_state']
        wait = delayObj['wait_time']
        #print state, wait
        
        if state=="PAUSED":
            print "PAUSED: Select <a>bort program, <q>uit playback, <c>ontinue playback, or <s>tep to next frame."
            wait = 0
            
        c = cv.WaitKey(wait)
        c = c & 127 #bit mask to get only lower 8 bits
        
        #sometimes a person has to hold down the spacebar to get the input
        # recognized by the cv.WaitKey() within the short time limit. So
        # we need to 'soak up' these extra inputs when the user is still
        # holding the spacebar, but we've gotten into the pause state.
        while c==ord(' '):
            print "PAUSED: Select <a>bort program, <q>uit playback, <c>ontinue playback, or <s>tep to next frame."
            c = cv.WaitKey(0)
            c = c & 127 #bit mask to get only lower 8 bits
        
        #At this point, we have a non-spacebar input, so process it.
        if c == ord('a'):   #abort
            print "User Aborted Program."
            raise SystemExit
        elif c == ord('q'): #quit video playback
            return 'q'
        elif c == ord('c'): #continue video playback
            delayObj['current_state'] = "PLAYING"
            return 'c'
        elif c == ord('s'): #step to next frame, keep in paused state
            delayObj['current_state'] = "PAUSED"
            return 's'
        else:   #any other keyboard input is just returned
            #delayObj['current_state'] = "PAUSED"
            return chr(c)
        
class FFMPEGVideo:
    # TODO: there may be a bug with the popen interface
    
    def __init__(self,filename,size=None,aspect=None,options=""):
        self.filename = filename
        self.size = size
        self.aspect = aspect
        self.options = options 
        
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
        #assert t2=='C420mpeg2'
        #assert t3=='XYSCSS=420MPEG2'

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

        cv.SetData(self.frame_y,y)
        cv.SetData(self.frame_u2,u)
        cv.SetData(self.frame_v2,v)

        cv.Resize(self.frame_u2,self.frame_u)
        cv.Resize(self.frame_v2,self.frame_v)
        
        cv.Merge(self.frame_y,self.frame_u,self.frame_v,None,self.frame_col)
        cv.CvtColor(self.frame_col,self.frame_col,cv.CV_YCrCb2RGB)
        
        out = self.frame_col
        
        if self.size != None:
            cv.Resize(self.frame_col,self.frame_resized)
            out = self.frame_resized

        return pv.Image(self.frame_y),pv.Image(self.frame_u),pv.Image(self.frame_v),pv.Image(out)
    
    
    def __iter__(self):
        ''' Return an iterator for this video '''
        return FFMPEGVideo(self.filename,size=self.size,aspect=self.aspect,options=self.options)

        
    def next(self):
        try:
            _,_,_,frame = self.frame()
        except EOFError:
            raise StopIteration("End of video sequence")
        return frame

class VideoFromFileList(Video):
    '''
    Given a sorted list of filenames (including full path), this will
    treat the list as a video sequence.
    '''
    def __init__(self, filelist, size=None):
        '''
        @param filelist: a list of full file paths to the images that comprise the video.
        They must be files capable of being loaded into a pv.Image() object, and should
        be in sorted order for playback.
        @param size: Optional tuple to indicate the desired playback window size.
        '''
        self.filelist = filelist
        self.idx = 0
        self.size = size
        
    def resize(self,frame):
        if self.size == None:
            return frame
        else:
            depth = frame.depth
            channels = frame.channels
            w,h = self.size
            resized = cv.CreateImage( (w,h), depth, channels )
            cv.Resize( frame.asOpenCV(), resized, cv.CV_INTER_LINEAR )
            return pv.Image(resized)
            
    def query(self):
        if self.idx >= len(self.filelist): return None
        f = self.filelist[self.idx]
        frame = pv.Image(f)
        self.idx += 1
        return self.resize(frame)
        
                    
    def next(self):
        frame = self.query()
        if frame == None:
            raise StopIteration("End of video sequence")
        return frame
        
    def __iter__(self):
        ''' Return an iterator for this video '''
        return VideoFromFileList(self.filelist) 
 
        
class VideoFromImages(Video):
    '''
    This class allows the user to treat a directory of images as a video. It is assumed that
    the files in the directory are named as follows:
    {prefix}{num}.{ext}
    where
    prefix is any string that is constant for all the files,
    ext is the file extension/type like jpg, png, etc.
    num is a zero-padded number like 0001, 0002, ...
         
    note: the amount of padded zeros is the minimum required based on the length
    (num frames) in the video, unless a specific padding is specified. So if you only had
    120 frames, then it would be 001, 002,...120.
    
    We assume the frames are sequential with no gaps, and start at number startnum (with 
    appropriate padding).
    '''
    def __init__(self,dirname,numframes,prefix="frame",ext="jpg", pad=None, startnum=1, size=None):
        '''
        The file names are of the format {prefix}{zero-padded num}.{ext}, the amount of
        zero-padding is determined automatically based on numframes. If there is additional
        zero-padding required, put it in the prefix.
        Example: a directory with images: vid_t1_s1_f001.jpg, ..., vid_t1_s1_f999.jpg
        would have prefix="vid_t1_s1_f", startnum=1, numframes=999, ext="jpg"

        @param dirname: directory where the images comprising the video exist 
        @param numframes: the number of frames in the video...0 to numframes will be read.
        specify None to read all images in directory, in which case you must specify
        a value for the pad parameter.
        @param prefix: a string which remains as a constant prefix to all frames in video
        @param ext: the extension of the images, like jpg, png, etc. Do not include the dot.
        @param pad: the padding (like string.zfill(x)) used on the sequential numbering of
        the input files. Specify None, and the padding will be determined based on length
        of numframes. (So if numframes = 1234, then pad=4, 0001,0002,...1234) 
        @param startnum: the starting number of the first frame, defaults to 1
        @param size: the optional width,height to resize the input frames
        '''
        self.dirname = dirname
        if numframes == None:
            #user wants to read all frames, so padding must be specified
            assert(pad != None and pad>0)
        
        if pad == None:
            pad = len(str(numframes))
            
        self.pad = pad                        
        self.maxframes = numframes
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
        numstr = str(self.current_frame).zfill(self.pad)
        filename = self.prefix + numstr + "." + self.ext
        f = os.path.join(self.dirname, filename)
        
        if (self.maxframes == None) or (self.current_frame <= self.maxframes):
            #then we query the next in the sequence until file not exists
            if os.path.exists(f):
                frame = pv.Image(f)
                self.current_frame += 1
                return(self.resize(frame))
            else:
                print "Image file %s does not exist. Stopping VideoFromImages."%f
        
        return None
       
    def resize(self,frame):
        if self.size == None:
            return frame
        else:
            depth = frame.depth
            channels = frame.channels
            w,h = self.size
            resized = cv.CreateImage( (w,h), depth, channels )
            cv.Resize( frame.asOpenCV(), resized, cv.CV_INTER_LINEAR )
            return pv.Image(resized)
                
    def next(self):
        frame = self.query()
        if frame == None:
            raise StopIteration("End of video sequence")
        return frame
        
    def __iter__(self):
        ''' Return an iterator for this video '''
        return VideoFromImages(self.dirname, self.maxframes, self.prefix, self.ext, self.pad, self.startnum, self.size) 
        
                
