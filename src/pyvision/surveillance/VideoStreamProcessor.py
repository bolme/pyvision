'''
Created on Mar 18, 2011
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
import cv
'''
This module implements various Video Stream Processors, or VSPs for short.
A VSP is designed to encapsulate a per-frame operation that can be
applied to a video stream. Examples include displaying the image while
overlaying the frame number (SimpleVSP), writing the output of a video
stream to a video file (VideoWriterVSP), and performing motion detection
on each video frame (MotionDetectionVSP).

The general idea is to chain together a VSP sequence, and then attach
the head of the chain to a video's play method. We hope that users will
create and/or contribute many useful subclasses of AbstractVSP.

For example:
import pyvision as pv
vsp_write = pv.VideoWriterVSP('tmp.avi',size=(640,480))
vsp_disp = pv.SimpleVSP(window="Display", nextModule=vsp_write)
vid = pv.Video(sourceFile)
vid.play(window=None, delay=25, onNewFrame=vsp_disp)
'''


class AbstractVSP():
    '''AbstractVSP is the abstract class definition of a
    Video Stream Processor (VSP) object. VSP's are designed to be chained
    together to accomplish processing on a video stream.
    '''
    
    def __init__(self, window=None, nextModule=None):
        ''' Constructor
        @param window: The window name to use when displaying this VSP's
        output. Specify None to suppress showing the output, but note that
        if you modify the current image with annotations, those will be
        persisted "downstream" to later processors.
        @param nextModule: A Video Stream Processor object that should be
        invoked on every frame after this processor has finished.
        '''
        self._windowName = window
        self._nextModule = nextModule
    
    def __call__(self, img, fn, **kwargs):
        newImg = self._onNewFrame(img, fn, **kwargs)
        if self._nextModule != None:
            if newImg != None:
                #we have a new image to replace the current one instream
                kwargs['orig_img']=img #add a new keyword arg to allow access to origininal img
                self._nextModule(newImg, fn, **kwargs)
            else:
                self._nextModule(img, fn, **kwargs)
            
    def _onNewFrame(self, img, fn, **kwargs):
        ''' Override this abstract method with the processing your object
        performs on a per-frame basis. It is recommended that you do not
        directly call this method. Rather, the VSP is a callable object,
        and so the __call__ method takes care of invoking this method as
        well as calling the next module, if any.
        '''
        raise NotImplemented
        
class FrameNumberVSP(AbstractVSP):
    '''A simple VSP object simply displays the input video frame with
    some simple annotation to show the frame number in upper left corner.
    '''        
    def _onNewFrame(self, img, fn, **kwargs):
        pt = pv.Point(10, 10)
        img.annotateLabel(label="Frame: %d"%(fn+1), point=pt, color="white", background="black")
        if self._windowName != None: img.show(window=self._windowName, delay=1)
 
#TODO: There seems to be a bug in the video writing output when writing
# frames from some source video objects in some output sizes. The symptom
# appears as an output video that is "slanted" and grainy.        
class VideoWriterVSP(AbstractVSP):
    '''
    A video stream processor that outputs to a new movie file.
    If you want to display the frame number in the output, chain this VSP
    after a SimpleVSP object in the series.
    '''
    def __init__(self, filename, window="Input", nextModule=None, fourCC_str="XVID", fps=15, size=None, bw=False, 
                 no_annotations = False):
        '''
        Constructor
        @param filename: The full output filename. Include the extension, such as .avi.
        @param window: The window name to use when displaying this VSP's
        output. Specify None to suppress showing the output, but note that
        if you modify the current image with annotations, those will be
        persisted "downstream" to later processors.
        @param nextModule: A Video Stream Processor object that should be
        invoked on every frame after this processor has finished.
        @param fourCC_str:  The "Four CC" string that is used to specify the encoder.
        @param fps: Frames per second. Not all codecs allow you to specify arbitrary frame rates, however.
        @param size: A tuple (w,h) representing the size of the output frames.
        @param bw: Specify true if you wish for a black-and-white only output.
        @param no_annotations: set to True to output the original, non-annotated version of the image
        '''
        cvFourCC = cv.CV_FOURCC(*fourCC_str)
        if bw:
            colorFlag = cv.CV_LOAD_IMAGE_GRAYSCALE
        else:
            colorFlag = cv.CV_LOAD_IMAGE_UNCHANGED
        self._bw = bw
        self._out = cv.CreateVideoWriter(filename, cvFourCC, fps, size, colorFlag)
        self._no_annotations = no_annotations
        AbstractVSP.__init__(self, window=window, nextModule=nextModule)
        
    def addFrame(self, img):
        '''
        @param img: A pyvision img to write out to the video.        
        '''
        if self._no_annotations:
            img2 = img
        else:
            img2 = pv.Image(img.asAnnotated())
            
        if self._bw:
            cv.WriteFrame(self._out, img2.asOpenCVBW())    
        else:
            cv.WriteFrame(self._out, img2.asOpenCV())        
           
    def _onNewFrame(self, img, fn, **kwargs):
        self.addFrame(img)

class ResizerVSP(AbstractVSP):
    '''This VSP resizes each frame of video. Subsequent VSPs in a chain
    will see the resized image instead of the original.
    '''
    def __init__(self, new_size=(320,240), window="Resized Image", nextModule=None):
        self._newSize = new_size
        AbstractVSP.__init__(self, window=window, nextModule=nextModule)
    
    def _onNewFrame(self, img, fn, **kwargs):
        img = img.resize(self._newSize)
        if self._windowName != None: img.show(window=self._windowName, delay=1)
        return img
        
class MotionDetectionVSP(AbstractVSP):
    ''' This VSP uses an existing motion detection object to apply motion
    detection to each frame of video.
    '''
    def __init__(self, md_object, window="Motion Detection", nextModule=None):
        ''' Constructor
        @param md_object: The pyvision motion detection object to be used by
        this VSP
        @param window: The name of the output window. Use None to suppress output.
        @param nextModule: The next VSP, if any, to be called by this VSP.
        '''
        self._md = md_object
        AbstractVSP.__init__(self, window=window, nextModule=nextModule)
        
    def _onNewFrame(self, img, fn, **kwargs):
        ''' Performs motion detection using this object's md object,
        displays the foreground pixels to a window.
        '''
        md = self._md
        rc = md.detect(img)
        if rc > -1:
            md.annotateFrame(img, rect_color="yellow", contour_color=None, flow_color=None)
            if self._windowName != None: img.show(window=self._windowName, delay=1)
            #img_fg = md.getForegroundPixels()
            #img_fg.show("Foreground")
            
class PeopleDetectionVSP(AbstractVSP):
    ''' This Video Stream Processor applies the OpenCV HOG people detector
    to each frame of video, annotating the detections with red rectangles.
    '''
    def _onNewFrame(self, img, fn, **kwargs):
        rects = self._detectPeople(img)
        for r in rects: img.annotateRect(r)
        if self._windowName != None: img.show(window=self._windowName, delay=1)
        
    def _detectPeople(self, img):
        cvim = img.asOpenCV()  #convert to OpenCV format before using OpenCV functions
        rect_list = []
        try:
            found = list(cv.HOGDetectMultiScale(cvim, cv.CreateMemStorage(0)))
            rect_list = [ pv.Rect(x,y,w,h) for ((x,y),(w,h)) in found]  #python list comprehension            
        except:
            #cv.HOGDetectMultiScale can throw exceptions, so return empty list
            return []
        
        return rect_list
        
        
        
        