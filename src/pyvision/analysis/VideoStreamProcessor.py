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

class AbstractVSP():
    '''AbstractVSP is the abstract class definition of a
    Video Stream Processor (VSP) object. VSP's are designed to be chained
    together to accomplish processing on a video stream.
    '''
    
    def __init__(self, nextModule=None):
        ''' Constructor
        @param nextModule: A Video Stream Processor object that should be
        invoked on every frame after this processor has finished.
        '''
        self._nextModule = nextModule
    
    def __call__(self, img, fn, **kwargs):
        self._onNewFrame(img, fn, **kwargs)
        if self._nextModule != None:
            self._nextModule(img, fn, **kwargs)
            
    def _onNewFrame(self, img, fn, key=None, buffer=None, prevModule=None):
        ''' Override this abstract method with the processing your object
        performs on a per-frame basis. It is recommended that you do not
        directly call this method. Rather, the VSP is a callable object,
        and so the __call__ method takes care of invoking this method as
        well as calling the next module, if any.
        '''
        raise NotImplemented
        
class SimpleVSP(AbstractVSP):
    '''A simple VSP object simply displays the input video frame with
    some simple annotation to show the frame number in upper left corner.
    '''
    def _onNewFrame(self, img, fn, key=None, buffer=None, prevModule=None):
        pt = pv.Point(10, 10)
        img.annotateLabel(label="Frame: %d"%(fn+1), point=pt, color="white")
        img.annotateLabel(label="Key: %s"%key, point=pv.Point(10,20), color="white")
        img.show("Input Video")
        
class MotionDetectionVSP(AbstractVSP):
    def __init__(self, md_object ,nextModule=None):
        ''' Constructor
        @param md_object: The pyvision motion detection object to be used by
        this VSP
        @param nextModule: The next VSP, if any, to be called by this VSP.
        '''
        self._md = md_object
        AbstractVSP.__init__(self, nextModule)
        
    def _onNewFrame(self, img, fn, key=None, buffer=None, prevModule=None):
        ''' Performs motion detection using this object's md object,
        displays the foreground pixels to a window.
        '''
        md = self._md
        rc = md.detect(img)
        if rc > -1:
            img_fg = md.getForegroundPixels()
            img_fg.show("Foreground")
            