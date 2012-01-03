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

class ImageBuffer:
    '''
    Stores a limited number of images from a video (or any other source)
    Makes it easy to do N-frame-differencing, for example, by easily being
    able to get the current (middle) frame, plus the first and last frames of the
    buffer. With an ImageBuffer of size N, as images are added, eventually the
    buffer fills, and older items are dropped off the end. This is convenient
    for streaming input sources, as the user can simply keep adding images
    to this buffer, and internally, the most recent N will be kept available.
    '''

    def __init__(self, N=5):
        '''
        @param N: how many image frames to buffer
        '''
        self._data = [None for _ in xrange(N)]
        self._count = 0
        self._max = N
            
    def __getitem__(self, key):
        return self._data[key]
        
    def __len__(self):
        '''
        This is a fixed-sized ring buffer, so length is always the number
        of images that can be stored in the buffer (as initialized with Nframes)
        '''
        return self._max
    
    def isFull(self):
        if self._count == self._max:
            return True
        else:
            return False
            
    def clear(self):
        self._data = [None for _ in xrange(self._max)]
        self._count = 0
            
    def getCount(self):
        '''
        Note that getCount() differs from __len__() in that this method returns the number of
        image actually stored in the ImageBuffer, while __len__() returns the size of the buffer,
        defined as the number of images the buffer is allowed to store.
        '''
        return self._count
    
    def getBuffer(self):
        return self._data
            
    def getFirst(self):
        return self._data[0]
    
    def getLast(self):
        return self._data[-1]
    
    def getMiddle(self):
        mid = int(self._count/2)
        return self._data[mid]
            
    def add(self, image):
        '''
        add an image to the buffer, will kick out the oldest of the buffer is full
        @param  image: image to add to buffer
        '''
        self._data.pop(0)  #remove last, if just beginning, this will be None
        self._data.append(image)
        self._count += 1
        if(self._count > self._max):
            self._count = self._max
            
    def fillBuffer(self, vid):
        '''
        If buffer is empty, you can use this function to spool off the first
        N frames of the video to initialize/fill the buffer.
        @param vid: an iterator of images, typically a pv.Video object or similar.
        @note: Will cause an assertion exception if buffer is already full.
        '''
        assert not self.isFull()
        
        while not self.isFull():
            im = vid.next()
            self.add(im)

        return

    def asStackBW(self, size=None):
        '''
        Outputs an image buffer as a 3D numpy array ("stack") of grayscale images.
        @param size: A tuple (w,h) indicating the output size of each frame.
        If None, then the size of the first image in the buffer will be used.
        @return: a 3D array (stack) of the gray scale version of the images
        in the buffer. The dimensions of the stack are (N,w,h), where N is
        the number of images (buffer size), w and h are the width and height
        of each image.        
        '''
        if size==None:
            img0 = self[0]        
            (w,h) = img0.size
        else:
            (w,h) = size
            
        f = self.getCount()
        stack = sp.zeros((f,w,h))
        for i,img in enumerate(self._data):
            #if img is not (w,h) in size, then resize first
            sz = img.size
            if (w,h) != sz:
                img2 = img.resize((w,h))
                mat = img2.asMatrix2D()
            else:
                mat = img.asMatrix2D()
            stack[i,:,:] = mat
            
        return stack
    
    def asMontage(self, layout, tileSize=None, **kwargs):
        (w,h) = self[0].size
        if tileSize == None:
            tw = w/5
            th = h/5
            if tw < 32: tw=32
            if th < 24: th=24
            tileSize = (tw,th)
            
        im = pv.ImageMontage(self._data, layout=layout, tileSize=tileSize, **kwargs)
        return im
    
    def show(self, N=10, window="Image Buffer", pos=None, delay=0):
        '''
        @param N: The number of images in the buffer to display at once
        @param window: The window name
        @param pos: The window position
        @param delay: The window display duration 
        '''
        if self[0] == None: return
        
        if N <= self._count:
            im = self.asMontage(layout=(1,N))
        else:
            im = self.asMontage(layout=(1,self._count))
        im.show(window, pos, delay)
        #img = im.asImage()
        #img.show(window, pos, delay)
        