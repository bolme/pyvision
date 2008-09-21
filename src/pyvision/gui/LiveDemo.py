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

#!/usr/bin/env python
import wx
import os
import os.path
import pyvision as pv
from pyvision.types.Image import Image
from pyvision.types.Point import Point
from pyvision.edge.canny import canny
from pyvision.point.DetectorHarris import DetectorHarris
from pyvision.point.DetectorDOG import DetectorDOG
from pyvision.face.CascadeDetector import CascadeDetector
from pyvision.types.Video import Webcam
import opencv
import time
from PIL.Image import FLIP_LEFT_RIGHT

'''
This is a simple live demo gui that loads an image from a webcam and displays it on the screen.
'''

class FrameTimer(wx.Timer):
    def __init__(self,target):
        wx.Timer.__init__(self)
        self.target = target
        
    
    def Notify(self):
        self.target.onFrame()
        
class RenderHarris:
    def __init__(self):
        self.harris = DetectorHarris()
        
    def __call__(self,im):
        points = self.harris.detect(im)
        for score,pt,radius in points:
            im.annotatePoint(pt)
        return im
        
class RenderDOG:
    def __init__(self):
        self.dog = DetectorDOG()
        
    def __call__(self,im):
        tmp = im.asPIL()
        tmp = tmp.resize((160,120))
        tmp = Image(tmp)
        points = self.dog.detect(tmp)
        for score,pt,radius in points:
            pt = Point(pt.X()*4,pt.Y()*4)
            im.annotateCircle(pt,radius*4)
        return im


class RenderFace:
    
    def __init__(self):
        self.face = CascadeDetector()
        
    # --------------------------------------------
    def __call__(self,im):
        faces = self.face(im)
        for face in faces:
            im.annotateRect(face)
        return im


class RenderAffine:
    
    def __init__(self):
        self.face = CascadeDetector()
        
    # --------------------------------------------
    def __call__(self,im):
        faces = self.face(im)
        for face in faces:
            im.annotateRect(face)
        return im

class RenderPerspective:
    
    def __init__(self):
        src = [pv.Point(0,0),pv.Point(639,0),pv.Point(639,479),pv.Point(0,479)]
        dst = [pv.Point(50,25),pv.Point(620,120),pv.Point(610,300),pv.Point(40,460)]
        self.transform = pv.PerspectiveFromPoints(src,dst,(640,480))
        
    # --------------------------------------------
    def __call__(self,im):
        return self.transform.transformImage(im)



DEMO_DEFAULTS = {'Canny':canny,
                 'Harris':RenderHarris(),
                 'DOG':RenderDOG(),
                 'Face':RenderFace(),
                 'Perspective':RenderPerspective(),
                }

class LiveDemoFrame(wx.Frame):
    
    def __init__(self,parent,id,name,demos=DEMO_DEFAULTS,size=(800,550)):
        wx.Frame.__init__(self,parent,id,name,size=size)
        
        # ---------------- Basic Data -------------------
        self.webcam = Webcam()
        self.harris = DetectorHarris()
        self.dog = DetectorDOG(n=100,selector='best')
        self.face = CascadeDetector()
        self.demos = demos
        
        # ------------- Other Components ----------------
        self.CreateStatusBar()
        
        # ------------------- Menu ----------------------
        
        # Creating the menubar.
        
        # ----------------- Image List ------------------
        
        # --------------- Image Display -----------------
        self.static_bitmap = wx.StaticBitmap(self,wx.NewId(), bitmap=wx.EmptyBitmap(640, 480))
        
        self.radios = wx.RadioBox(self,wx.NewId(),'Demos',
                                 choices=['None'] + self.demos.keys(),
                                 style=wx.RA_SPECIFY_ROWS)
        
        self.mirror = wx.CheckBox(self,wx.NewId(),'Mirror')
        self.mirror.SetValue(True)
        
        # --------------- Window Layout -----------------
        grid = wx.FlexGridSizer(2,2)
        grid.Add(self.static_bitmap)
        grid.Add(self.radios)
        grid.Add(self.mirror)

        self.SetAutoLayout(True)
        self.SetSizer(grid)
        self.Layout()
        
        # -----------------------------------------------
        self.timer = FrameTimer(self)
        self.timer.Start(200)
        # -------------- Event Handleing ----------------
        wx.EVT_SIZE(self.static_bitmap, self.onBitmapResize)
        wx.EVT_LEFT_DOWN(self.static_bitmap, self.onClick)
        wx.EVT_TIMER(self,-1,self.onTmp)
        
        #wx.EVT_CLOSE(self,self.onClose)        
                
    def onTmp(self):
        print "Notify"
        
    def onFrame(self,event=None):
        self.timer.Stop()        
        starttime = time.time()
        img = self.webcam.query()
        
        selection = self.radios.GetStringSelection()
        if selection == 'None':
            pass
        else:
            for key,func in self.demos.iteritems():
                if key == selection:
                    img = func(img)

        print "Displaying Annotated Frame:", selection
        im = img.asAnnotated()
        if self.mirror.GetValue():
            im = im.transpose(FLIP_LEFT_RIGHT)
        wxImg = wx.EmptyImage(im.size[0], im.size[1])
        wxImg.SetData(im.tostring())
        bm = wxImg.ConvertToBitmap()
            
        self.static_bitmap.SetBitmap(bm)
        print "Frame Time:",time.time() - starttime
        self.timer.Start(milliseconds = 1, oneShot = 1)

    
    def onBitmapResize(self,event):
        w = event.GetSize().GetWidth()
        h = event.GetSize().GetHeight()

        self.static_bitmap.SetSize(event.GetSize())
              
    # ------------- Event Handlers ---------------      
    def onClick(self,event):
        self.onFrame(event)
           
    #def onClose(self,event):
    #    pass        

def runDemos(name='PyVision Live Demo', demos=DEMO_DEFAULTS):
    app = wx.PySimpleApp()
    frame = LiveDemoFrame(None, wx.ID_ANY, name, demos)
    frame.Show(True)
    app.MainLoop()
    
if __name__ == '__main__':
    runDemos()    