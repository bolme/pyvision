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
import cv


class CaptureClicks:
    '''
    This object handles the data mangagement and display of the capture clicks window.
    '''
    
    def __init__(self,im):
        '''
        Initialize the data.
        '''
        self.im = im.copy()
        self.reset()
        
    def display(self):
        '''
        Display the window and run the main event loop.
        '''
        # Setup the mouse callback to handle mause events (optional)
        cv.NamedWindow("PyVision Capture Points")
        cv.SetMouseCallback("PyVision Capture Points", self.mouseCallback)
        
        while True:
            key_press = self.im.show("PyVision Capture Points",delay=100)
            
            # Handle key press events.
            if key_press == ord(' '):
                break
            
            if key_press == ord('r'):
                self.reset()
                
        cv.DestroyWindow("PyVision Capture Points")
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
        if event in [cv.CV_EVENT_LBUTTONDOWN]:
            print "Mouse Event:",event,x,y
            point = pv.Point(x,y)
            self.im.annotateLabel(point,str(len(self.points)),mark='below')
            self.points.append(point)
        
def capturePointsFromMouse(im):
    '''
    This function opens a high gui window that displays the image.  Any 
    points that are clicked will be returned after the user presses the 
    space bar.
    
    @param im: An image to display.
    @returns: a list of points that were clicked by the user.
    '''
    cap = CaptureClicks(im)
    clicks = cap.display()
    return clicks
        