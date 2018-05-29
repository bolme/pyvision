# PyVision License
#
# Copyright (c) 2006-2011 David S. Bolme
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
Created on Oct 31, 2011

@author: bolme
'''

import pyvision as pv
#import cv2
import PIL.Image as pil
from collections import defaultdict

def null_callback(*args,**kwargs):
    pass

class SingletonCallback(object):
    def __init__(self):
        self.my_callback = None
    
    def __call__(self,*args,**kwargs):
        if self.my_callback is not None:
            self.my_callback(*args,**kwargs)
        else:
            print("pyvision.analysis.gui_tools.SingletonCallback: warning no callback function.")
            
    def set(self,callback):
        self.my_callback = callback
        
    def release(self):
        self.my_callback = None

CALLBACK_DICT = defaultdict(SingletonCallback)

class CaptureClicks:
    '''
    This object handles the data mangagement and display of the capture clicks window.
    '''
    
    def __init__(self,im,default_points=[],keep_window_open = False,window="PyVision Capture Points"):
        '''
        Initialize the data.
        '''
        import cv2
        self.window = window
        self.im = im.copy()
        self.keep_window_open = keep_window_open
        self.reset()
        for pt in default_points:
            self.mouseCallback(cv2.EVENT_LBUTTONDOWN,pt.X(),pt.Y(),None,None)
            
    def display(self):
        '''
        Display the window and run the main event loop.
        '''
        import cv2
        global CALLBACK_DICT
        # Setup the mouse callback to handle mause events (optional)
        cv2.namedWindow(self.window)
        
        CALLBACK_DICT[self.window].set(self.mouseCallback)
        cv2.setMouseCallback(self.window, CALLBACK_DICT[self.window])
        
        while True:
            key_press = self.im.show(self.window,delay=100)
            key_press = key_press%128
            # Handle key press events.
            if key_press == ord(' '):
                break
            
            if key_press == ord('r'):
                self.reset()
                
        CALLBACK_DICT[self.window].release()
                
        if not self.keep_window_open:
            cv2.destroyWindow(self.window)
            
        return self.points
                
    def reset(self):
        '''
        Clear the points and start over.
        '''
        self.im = self.im.copy()
        self.im.annotateLabel(pv.Point(10,10), "Click anywhere in the image to select a point.",color='yellow')
        self.im.annotateLabel(pv.Point(10,20), "Press the 'r' to reset.",color='yellow')
        self.im.annotateLabel(pv.Point(10,30), "Press the space bar when finished.",color='yellow')
        self.points = []
        
            
    def mouseCallback(self, event, x, y, flags, param):
        '''
        Call back function for mouse events.
        '''
        import cv2
        if event in [cv2.EVENT_LBUTTONDOWN]:
            point = pv.Point(x,y)
            self.im.annotateLabel(point,str(len(self.points)),mark='below')
            self.points.append(point)
        
class CaptureClicksVideo:
    '''
    This object handles the data mangagement and display of the capture clicks window.
    '''
    
    def __init__(self, video, buffer_size = 60, callback = None, keep_window_open=False):
        '''
        Initialize the data.
        '''
        self.callback = callback
        self.video = video
        self.points = {}
        self.buffer = []
        self.frame = -1
        self.buffer_index = -1
        self.buffer_size = buffer_size
        self.keep_window_open = keep_window_open
        next(self)
        
        
    def display(self):
        '''
        Display the window and run the main event loop.
        '''
        import cv2
        # Setup the mouse callback to handle mause events (optional)
        cv.NamedWindow("PyVision Capture Points")
        
        # This next line creates a memory leak where 'self' is never released
        # and the window cannot be closed.
        cv.SetMouseCallback("PyVision Capture Points", self.mouseCallback)
        
        
        while True:
            key_press = self.im.show("PyVision Capture Points",delay=100)
            key_press = key_press%128
            
            # Handle key press events.
            if key_press == ord('r'):
                self.reset()

            if key_press == ord('p'):
                self.prev()

            if key_press == ord('P'):
                for _ in range(10):
                    self.prev()
                
            if key_press == ord(' ') or key_press == ord('n'):
                next(self)

            if key_press == ord('N'):
                for _ in range(10):
                    next(self)
                
            if key_press == ord('q'):
                break
                
        # Reduce but does not eliminate the memory leak.
        del self.buffer
        
        if not self.keep_window_open:
            cv2.DestroyWindow("PyVision Capture Points")
        
        return self.points
            
    def reset(self):
        if self.frame in self.points:
            del self.points[self.frame]
            self.render()
            
    def render(self):
        '''
        Clear the points and start over.
        '''
        im = self.buffer[self.buffer_index]
        w,h = im.size
        nim = pil.new('RGB',(w,h+100))
        nim.paste(im.asPIL(),(0,0))
        self.im = pv.Image(nim)
        
        if self.callback != None:
            self.callback(self.im,self.frame)
        
        self.im.annotateLabel(pv.Point(10,h+10), "Frame: %d"%self.frame,color='yellow')
        self.im.annotateLabel(pv.Point(10,h+20), "Click anywhere in the image to select a point.",color='yellow')
        self.im.annotateLabel(pv.Point(10,h+30), "Press 'r' to reset.",color='yellow')
        self.im.annotateLabel(pv.Point(10,h+40), "Press the space bar or 'n' for the next frame.",color='yellow')
        self.im.annotateLabel(pv.Point(10,h+50), "Press 'p' for the previous frame.",color='yellow')
        self.im.annotateLabel(pv.Point(10,h+60), "Press 'N' or 'P' to skip 10 frames.",color='yellow')
        self.im.annotateLabel(pv.Point(10,h+70), "Press 'q' when finished.",color='yellow')
        if self.frame in self.points:
            points = self.points[self.frame]
            for i in range(len(points)):
                pt = points[i]
                self.im.annotateLabel(pt,'%d'% i,mark='below')
        
    def __next__(self):
        if self.buffer_index == -1:
            try:
                self.buffer.append(next(self.video))
                self.frame += 1
            except StopIteration:
                print("End of video.")
            self.buffer = self.buffer [-self.buffer_size:]
        else:
            self.buffer_index += 1
            self.frame += 1

        print(self.buffer_index,self.frame,len(self.buffer),self.points)
        self.render()
        
    
    def prev(self):
        if self.buffer_index == -len(self.buffer):
            print("Buffer exceed. Cannot display previous frame")
        else:
            self.buffer_index -= 1
            self.frame -= 1
        self.render()
        
            
    def mouseCallback(self, event, x, y, flags, param):
        '''
        Call back function for mouse events.
        '''
        import cv2
        if event in [cv2.CV_EVENT_LBUTTONDOWN]:
            if self.frame not in self.points:
                self.points[self.frame] = []
            points = self.points[self.frame]
            point = pv.Point(x,y)
            self.im.annotateLabel(point,str(len(points)),mark='below')
            points.append(point)
        
def capturePointsFromMouse(im,*args,**kwargs):
    '''
    This function opens a high gui window that displays the image.  Any 
    points that are clicked will be returned after the user presses the 
    space bar.
    
    @param im: An image to display.
    @param default_points: Some default points to display.
    @type default_points: list of pv.Point
    @type default_points: list
    @param keep_window_open: keep the window open after point were captured
    @type True|False
    @param window: The name of the window
    @type window: string
    @returns: a list of points that were clicked by the user.
    '''
    if isinstance(im, pv.Image):
        cap = CaptureClicks(im,*args,**kwargs)
        clicks = cap.display()
    else:
        cap = CaptureClicksVideo(im,*args,**kwargs)
        clicks = cap.display()
    return clicks


if __name__ == '__main__':
    #im = pv.Image(pv.TAZ_IMAGE)
    #pv.capturePointsFromMouse(im)
    
    video = pv.Video(pv.TAZ_VIDEO)
    ccv = capturePointsFromMouse(video)
