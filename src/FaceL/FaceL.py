import sys
import os
import os.path
import traceback

# TODO: FaceL Todo list
# * Try to track down svm training bug. (It possibly stops the timer?)
# * About Dialog
# * Add Horizontal scroll bar to exception dialog

# Requests
# SVM Tuning Grid
# Save training data
# Read and enroll from a video/image (Labeled Faces in the Wild)



ABOUT_MESSAGE = '''Welcome to Facile Face Labeling (FaceL)
                  
FaceL is a simple application that detects, registers, and labels faces.

About the system...

The source code for FaceL is available as part of the PyVision library and can be downloaded from:

http://pyvision.sourceforge.net.

* Face detection uses the OpenCV Cascade Face Detector.

    Paul Viola and Michael Jones. "Robust Real-time Face 
    Detection." International Journal of Computer Vision. 
    Vol 57. No 4. 2004

* Eye localization uses Average of Synthetic Exact Filters. 
    
    David Bolme, Bruce Draper, and Ross Beveridge.
    "Average of Synthetic Exact Filters". Proceedings of
    IEEE Conference on Computer Vision and Pattern 
    Recognition. 2009.
    
* Labeling is performed using the libsvm Support Vector Classifier. 
    
    Chih Chung Chang and Chih Jen Lin. LIBSVM: a library for 
    support vector machines. 2001. Software available at 
    http://www.csie.ntu.edu.tw/~cjlin/libsvm

                  
Created by David S. Bolme and J. Ross Beveridge
Colorado State University
Fort Collins, Colorado, USA'''

LICENSE_MESSAGE = '''FaceL is released under the PyVision License

PyVision License

Copyright (c) 2006-2009 David S. Bolme
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
3. Neither name of copyright holders nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 
 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
'''

CAMERA_ERROR_CODE = 10
CAMERA_ERROR = '''An error occured while FaceL was connecting to the camera. Please check the camera and try again. 

* If using an external camera make sure that the camera is connected and turned on.

* Make sure there are no other applications using the camera.  Quit any other applications that may be using the iSight camera such as iChat or Photo Booth.

If this problem persists please submit a bug report to bolme@cs.colostate.edu with the text "FaceL Bug" in the subject line.  Please copy and paste the following information into the email message.
'''

UNKNOWN_ERROR_CODE = 11
UNKNOWN_ERROR = '''An unknown exception occurred while running FaceL.  Details can be found bellow.

If this problem persists please submit a bug report to bolme@cs.colostate.edu with the text "FaceL Bug" in the subject line.  Please copy and paste the following information into the email message.'''

SVM_MANUAL    = "SVM_MANUAL"
SVM_AUTOMATIC = "SVM_AUTOMATIC"


if os.environ.has_key('RESOURCEPATH'):
    ##################################################################
    # Use these settings as an application bundle
    ##################################################################
    sys.path = [os.path.join(os.environ['RESOURCEPATH'], 'lib', 'python2.5', 'lib-dynload')] + sys.path
    sys.path = [os.path.join(os.environ['RESOURCEPATH'], 'lib', 'python2.5')] + sys.path
    
    # Resource Locations
    CASCADE_NAME = os.path.join(os.environ['RESOURCEPATH'], "haarcascade_frontalface_alt.xml")
    FEL_NAME = os.path.join(os.environ['RESOURCEPATH'], "EyeLocatorASEF128x128.fel")
    ARIAL_BLACK_NAME = os.path.join(os.environ['RESOURCEPATH'], "Arial Black.ttf")
    FACEL_LOGO = os.path.join(os.environ['RESOURCEPATH'], "LogoFaceL.png")
    CSU_LOGO = os.path.join(os.environ['RESOURCEPATH'], "ColoStateLogo.png")

else:
    ##################################################################
    # Use these settings when run using eclipse
    ##################################################################
    
    # Resource Locations
    CASCADE_NAME = "haarcascade_frontalface_alt.xml"
    FEL_NAME = "EyeLocatorASEF128x128.fel"
    ARIAL_BLACK_NAME = "Arial Black.ttf"
    FACEL_LOGO = "LogoFaceL.png"
    CSU_LOGO = "ColoStateLogo.png"

from pyvision.types.Video import Webcam
from pyvision.face.CascadeDetector import CascadeDetector
from pyvision.analysis.FaceAnalysis.FaceDetectionTest import is_success
import pyvision as pv
import pickle
import wx
import wx.lib.plot as plot
import time
from PIL.Image import FLIP_LEFT_RIGHT
import PIL
import PIL.ImageFont
from pyvision.face.SVMFaceRec import SVMFaceRec
from pyvision.face.FilterEyeLocator import loadFilterEyeLocator
import opencv as cv
import math


class FrameTimer(wx.Timer):
    '''
    This timer is used to control the Video capture process from the iSight camera.
    '''
    def __init__(self,target):
        wx.Timer.__init__(self)
        self.target = target
        
    
    def Notify(self):
        self.target.onFrame()


class VideoWindow(wx.Frame):
    '''
    This is the main FaceL window which includes the webcam video and enrollment and training controls.
    '''
    
    def __init__(self,parent,id,name,size=(640,672)):
        '''
        Create all the windows and controls used for the window and 
        '''
        wx.Frame.__init__(self,parent,id,name,size=size)
        
        self.CenterOnScreen(wx.HORIZONTAL)
        self.timing_window = None # Initialize timing window
        
        # ------------- Face Processing -----------------
        self.face_detector = CascadeDetector(cascade_name=CASCADE_NAME,image_scale=0.5)
        self.fel = loadFilterEyeLocator(FEL_NAME)
        self.face_rec = SVMFaceRec()
        
        self.svm_mode  = SVM_AUTOMATIC
        self.svm_C     = 4.000e+00
        self.svm_Gamma = 9.766e-04
        
        self.current_faces = []
        self.enrolling    = None
        self.enroll_count = 0
        self.enroll_max   = 32
        self.enroll_list  = []
        
        self.previous_time = time.time()
        
        self.arialblack24 = PIL.ImageFont.truetype(ARIAL_BLACK_NAME, 24)

        # ---------------- Basic Data -------------------
        try:
            self.webcam = Webcam()
        except SystemExit:
            raise
        except:
            trace = traceback.format_exc()
            message = TraceBackDialog(None, "Camera Error", CAMERA_ERROR, trace)
            message.ShowModal()
            
            sys.stderr.write("FaceL Error: an error occurred while trying to connect to the camera.  Details follow.\n\n")
            sys.stderr.write(trace)
            sys.exit(CAMERA_ERROR_CODE)

        # ------------- Other Components ----------------
        self.CreateStatusBar()
        
        # ------------------- Menu ----------------------
        # Creating the menubar.
        
        # Menu IDs
        license_id = wx.NewId()
        
        mirror_id = wx.NewId()
        face_id = wx.NewId()
        svm_tune_id = wx.NewId()
        performance_id = wx.NewId()
        
        # Menu Items
        self.file_menu = wx.Menu();

        self.file_menu.Append( wx.ID_ABOUT, "&About..." )
        self.file_menu.Append( license_id, "FaceL License..." )
        self.file_menu.AppendSeparator();
        self.file_menu.Append( wx.ID_EXIT, "E&xit" )

        self.options_menu = wx.Menu();
        self.face_menuitem = self.options_menu.AppendCheckItem( face_id, "Face Processing" )
        self.mirror_menuitem = self.options_menu.AppendCheckItem( mirror_id, "Mirror Video" )
        self.options_menu.AppendSeparator()
        self.options_menu.Append( svm_tune_id, "SVM Tuning..." )
        self.options_menu.Append( performance_id, "Performance..." )
        
        # Create Menu Bar
        self.menu_bar = wx.MenuBar();
        self.menu_bar.Append( self.file_menu, "&File" )
        self.menu_bar.Append( self.options_menu, "&Options" )

        self.SetMenuBar( self.menu_bar )
        
        # Menu Events
        wx.EVT_MENU(self, wx.ID_ABOUT, self.onAbout )
        wx.EVT_MENU(self, license_id, self.onLicense )

        wx.EVT_MENU(self, mirror_id, self.onNull )
        wx.EVT_MENU(self, face_id, self.onNull )
        wx.EVT_MENU(self, svm_tune_id, self.onSVMTune )
        wx.EVT_MENU(self, performance_id, self.onTiming )
        
        # Set up menu checks
        self.face_menuitem.Check(True)
        self.mirror_menuitem.Check(True)
        
        
        # ----------------- Image List ------------------
        
        # --------------- Image Display -----------------
        self.static_bitmap = wx.StaticBitmap(self,wx.NewId(), bitmap=wx.EmptyBitmap(640, 480))
        
        self.controls_box = wx.StaticBox(self, wx.NewId(), "Controls")

        self.facel_logo = wx.StaticBitmap(self,wx.NewId(), bitmap=wx.Bitmap(FACEL_LOGO))
        self.csu_logo = wx.StaticBitmap(self,wx.NewId(), bitmap=wx.Bitmap(CSU_LOGO))
#        self.performance_box = wx.StaticBox(self, wx.NewId(), "Performance")
        
        self.enroll_chioce_label = wx.StaticText(self, wx.NewId(), "Enrollment Count:", style=wx.ALIGN_LEFT)
        self.enroll_choice = wx.Choice(self,wx.NewId(),wx.Point(0,0),wx.Size(-1,-1),['16','32','48','64','128','256'])
        self.enroll_choice.Select(3)
        
        self.train_button = wx.Button(self,wx.NewId(),'Train Labeler')
        self.reset_button = wx.Button(self,wx.NewId(),'Clear Labels')
        
        # --------------- Instrumentation ---------------
        
        
          
        self.enroll_label = wx.StaticText(self, wx.NewId(), "Click a face in the video to enroll.", style=wx.ALIGN_LEFT)

        self.ids_label = wx.StaticText(self, wx.NewId(), "Labels:", size=wx.Size(-1,16), style=wx.ALIGN_LEFT)
        self.ids_text = wx.StaticText(self, wx.NewId(), size = wx.Size(30,16), style= wx.ALIGN_RIGHT )  
        
        self.faces_label = wx.StaticText(self, wx.NewId(), "Faces:", size=wx.Size(-1,16), style=wx.ALIGN_LEFT)
        self.faces_text = wx.StaticText(self, wx.NewId(), size = wx.Size(30,16), style= wx.ALIGN_RIGHT )          
        

        # --------------- Window Layout -----------------
        enroll_sizer = wx.BoxSizer(wx.HORIZONTAL)
        enroll_sizer.Add(self.ids_label, flag = wx.ALIGN_CENTER | wx.ALL, border=4)
        enroll_sizer.Add(self.ids_text, flag = wx.ALIGN_CENTER | wx.ALL, border=4)
        enroll_sizer.AddSpacer(20)
        enroll_sizer.Add(self.faces_label, flag = wx.ALIGN_CENTER | wx.ALL, border=4)
        enroll_sizer.Add(self.faces_text, flag = wx.ALIGN_CENTER | wx.ALL, border=4)

        training_sizer = wx.BoxSizer(wx.HORIZONTAL)
        training_sizer.Add(self.train_button, flag = wx.ALIGN_CENTER | wx.ALL, border=4)
        training_sizer.Add(self.reset_button, flag = wx.ALIGN_CENTER | wx.ALL, border=4)

    
        enroll_choice_sizer = wx.BoxSizer(wx.HORIZONTAL)
        enroll_choice_sizer.Add(self.enroll_chioce_label, flag = wx.ALIGN_CENTER | wx.ALL, border=0)
        enroll_choice_sizer.Add(self.enroll_choice, flag = wx.ALIGN_CENTER | wx.ALL, border=0)

        controls_sizer = wx.StaticBoxSizer(self.controls_box,wx.VERTICAL) #wx.BoxSizer(wx.VERTICAL)
        controls_sizer.Add(self.enroll_label, flag = wx.ALIGN_LEFT | wx.ALL, border=0)
        controls_sizer.Add(enroll_sizer, flag = wx.ALIGN_LEFT | wx.ALL, border=0)
        controls_sizer.Add(enroll_choice_sizer, flag = wx.ALIGN_LEFT | wx.ALL, border=4)
        controls_sizer.Add(training_sizer, flag = wx.ALIGN_LEFT | wx.ALL, border=0)

        bottom_sizer = wx.BoxSizer(wx.HORIZONTAL)
        bottom_sizer.Add(self.facel_logo, flag = wx.ALIGN_CENTER | wx.ALL, border=0)
        bottom_sizer.Add(controls_sizer, flag = wx.ALIGN_TOP | wx.ALL, border=4)
        bottom_sizer.Add(self.csu_logo, flag = wx.ALIGN_CENTER | wx.ALL, border=0)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.static_bitmap, flag = wx.ALIGN_CENTER | wx.ALL, border=0)
        main_sizer.Add(bottom_sizer, flag = wx.ALIGN_CENTER | wx.ALL, border=4)
        

        self.SetAutoLayout(True)
        self.SetSizer(main_sizer)
        self.Layout()
        
        # -----------------------------------------------
        self.timer = FrameTimer(self)
        self.timer.Start(200)
        
        # -------------- Event Handleing ----------------
        wx.EVT_SIZE(self.static_bitmap, self.onBitmapResize)
        wx.EVT_LEFT_DOWN(self.static_bitmap, self.onClick)
                
        self.Bind(wx.EVT_BUTTON, self.onTrain, id=self.train_button.GetId())
        self.Bind(wx.EVT_BUTTON, self.onReset, id=self.reset_button.GetId())
        
        # --------------- Setup State -------------------
        self.setupState()
                
    def onTrain(self,event=None):
        '''
        Start the SVM training process.
        '''
        print "Train"
        #progress = wx.ProgressDialog(title="SVM Training", message="Training the Face Recognition Algorithm. Please Wait...")
        if self.svm_mode == SVM_AUTOMATIC:
            # Train with automatic tuning.
            self.face_rec.train() #callback=progress.Update)
            self.svm_C = self.face_rec.svm.C
            self.svm_Gamma = self.face_rec.svm.gamma
        else:
            # Train with manual tuning.
            self.face_rec.train( C=[self.svm_C] , Gamma=[self.svm_Gamma])# ,callback=progress.Update)
        
        #progress.Destroy()
        
    def onReset(self,event=None):
        '''
        Clear the enrollment data for the SVM.
        '''
        self.face_rec.reset() 
        self.setupState()

        
    def onFrame(self,event=None):
        '''
        Retrieve and process a video frame.
        '''
        self.timer.Stop()        
        starttime = time.time()
        self.detect_time = 0.0
        self.eye_time = 0.0
        self.label_time = 0.0
        img = self.webcam.query()
        
        face_processing = self.face_menuitem.IsChecked()
        
        names = []
        
        if face_processing:
            faces = self.findFaces(img)
            if self.enrolling != None:
                success = None
                for rect,leye,reye in faces:
                    img.annotateRect(self.enrolling,color='yellow')
                    if (success == None) and is_success(self.enrolling,rect):
                        success = rect
                        img.annotateRect(rect,color='blue')
                        img.annotatePoint(leye,color='blue')
                        img.annotatePoint(reye,color='blue')
                        self.enroll_list.append([img,rect,leye,reye])

                    else:
                        img.annotateRect(rect,color='red')
                        img.annotatePoint(leye,color='red')
                        img.annotatePoint(reye,color='red')
                        img.annotateLine(pv.Point(rect.x,rect.y),pv.Point(rect.x+rect.w,rect.y+rect.h), color='red')
                        img.annotateLine(pv.Point(rect.x+rect.w,rect.y),pv.Point(rect.x,rect.y+rect.h), color='red')

                if success == None:
                    rect = self.enrolling
                    img.annotateLine(pv.Point(rect.x,rect.y),pv.Point(rect.x+rect.w,rect.y+rect.h), color='yellow')
                    img.annotateLine(pv.Point(rect.x+rect.w,rect.y),pv.Point(rect.x,rect.y+rect.h), color='yellow')
                else:
                    #enroll in the identification algorithm
                    pass
            else:
                for rect,leye,reye in faces:
                    img.annotateRect(rect,color='blue')
                    img.annotatePoint(leye,color='blue')
                    img.annotatePoint(reye,color='blue')
                    
            
            if self.face_rec.isTrained():
                self.label_time = time.time()
                for rect,leye,reye in faces:
                    label = self.face_rec.predict(img,leye,reye)
                    names.append([0.5*(leye+reye),label])
                self.label_time = time.time() - self.label_time


        # Displaying Annotated Frame
        im = img.asAnnotated()
        if self.mirror_menuitem.IsChecked():
            im = im.transpose(FLIP_LEFT_RIGHT)
            
        if self.enrolling != None:
            draw = PIL.ImageDraw.Draw(im)
            x,y = self.enrolling.x,self.enrolling.y
            if self.mirror_menuitem.IsChecked():
                x = 640 - (x + self.enrolling.w)
            self.enroll_count += 1
            draw.text((x+10,y+10), "Enrolling: %2d of %2d"%(self.enroll_count,self.enroll_max), fill='yellow', font=self.arialblack24)
            del draw
            
            if self.enroll_count >= self.enroll_max:
                print "Count:",self.enroll_count
                
                if len(self.enroll_list) == 0:
                    warning_dialog = wx.MessageDialog(self,
                                                      "No faces were detected during the enrollment process.  Please face towards the camera and keep your face in the yellow rectangle during enrollment.",
                                                      style=wx.ICON_EXCLAMATION | wx.OK,
                                                      caption="Enrollment Error")
                    warning_dialog.ShowModal()
                else:
                    name_dialog = wx.TextEntryDialog(self, "Please enter a name to associate with the face. (%d faces captured)"%len(self.enroll_list), caption = "Enrollment ID")
                    result = name_dialog.ShowModal()
                    sub_id = name_dialog.GetValue()
                    if result == wx.ID_OK:
                        if sub_id == "":
                            print "Warning: Empty Subject ID"
                            warning_dialog = wx.MessageDialog(self,
                                                              "A name was entered in the previous dialog so this face will not be enrolled in the database.  Please repeat the enrollment process for this person.",
                                                              style=wx.ICON_EXCLAMATION | wx.OK,
                                                              caption="Enrollment Error")
                            warning_dialog.ShowModal()
                        else:
                            for data,rect,leye,reye in self.enroll_list:
                                self.face_rec.addTraining(data,leye,reye,sub_id)
                                self.setupState()

                                
                self.enroll_count = 0
                self.enrolling    = None
                self.enroll_list  = []
            
            
        if len(names) > 0:
            draw = PIL.ImageDraw.Draw(im)
            for pt,name in names:
                x,y = pt.X(),pt.Y() 
                w,h = draw.textsize(name,font=self.arialblack24)
                if self.mirror_menuitem.IsChecked():
                    x = 640 - x - 0.5*w
                else:
                    x = x - 0.5*w
                draw.text((x,y-20-h), name, fill='green', font=self.arialblack24)
            del draw

            
            
        wxImg = wx.EmptyImage(im.size[0], im.size[1])
        wxImg.SetData(im.tostring())
        bm = wxImg.ConvertToBitmap()
            
        self.static_bitmap.SetBitmap(bm)
        
        # Update timing gauges
        full_time = time.time() - starttime
        if self.timing_window != None:
            self.timing_window.update(self.detect_time,self.eye_time,self.label_time,full_time)
               
        self.ids_text.SetLabel("%d"%(self.face_rec.n_labels,))
        self.faces_text.SetLabel("%d"%(self.face_rec.n_faces,))
        

        self.timer.Start(milliseconds = 1, oneShot = 1)


    
    def setupState(self):
        #print "state",self.face_rec.n_labels,self.IsEnabled()
        if self.face_rec.n_labels >= 2:
            self.train_button.Enable()
        else:
            self.train_button.Disable()
            
        
    def onBitmapResize(self,event):
        w = event.GetSize().GetWidth()
        h = event.GetSize().GetHeight()

        self.static_bitmap.SetSize(event.GetSize())
              
              
    def onClick(self,event):
        '''
        Process a click in the Video window which starts the enrollment process.
        '''
        x = event.GetX()
        y = event.GetY()
        
        if self.mirror_menuitem.IsChecked():
            x = 640-x
            
        for rect,leye,reye in self.current_faces:
            if rect.containsPoint(pv.Point(x,y)):
                self.enrolling = rect
                self.enroll_count = 0
                self.enroll_max = int(self.enroll_choice.GetStringSelection())
                

    def findFaces(self,im):
        faces = []
        
        self.detect_time = time.time()
        rects = self.face_detector.detect(im)
        self.detect_time = time.time() - self.detect_time

        cvim = im.asOpenCV()
        cvtile = cv.cvCreateMat(128,128,cv.CV_8UC3)
        bwtile = cv.cvCreateMat(128,128,cv.CV_8U)
        
        self.eye_time = time.time()
        for rect in rects:
            faceim = cv.cvGetSubRect(cvim, rect.asOpenCV())
            cv.cvResize(faceim,cvtile)
            
            affine = pv.AffineFromRect(rect,(128,128))

            cv.cvCvtColor( cvtile, bwtile, cv.CV_BGR2GRAY )
            
            leye,reye,lcp,rcp = self.fel.locateEyes(bwtile)
            leye = pv.Point(leye)
            reye = pv.Point(reye)
            
            leye = affine.invertPoint(leye)
            reye = affine.invertPoint(reye)
            
            faces.append([rect,leye,reye])
        self.eye_time = time.time() - self.eye_time

        self.current_faces = faces

        return faces
    
    def onAbout(self,event):
        wx.MessageBox( ABOUT_MESSAGE,
                  "About FaceL", wx.OK | wx.ICON_INFORMATION )
        
    def onLicense(self,event):
        wx.MessageBox( LICENSE_MESSAGE,
                  "FaceL License", wx.OK | wx.ICON_INFORMATION )

    def onNull(self,*args,**kwargs):
        pass
    
    def onSVMTune(self,event):
        dialog = SVMTuningDialog(self, self.svm_mode, self.svm_C, self.svm_Gamma)
        dialog.CenterOnParent()

        result = dialog.ShowModal()
        if result == wx.ID_OK:
            self.svm_mode  = dialog.mode
            self.svm_C     = dialog.C
            self.svm_Gamma = dialog.Gamma
        
        print "SVM Tuning Info <MODE:%s; C:%0.2e; Gamma:%0.2e>"%(self.svm_mode,self.svm_C,self.svm_Gamma)
        
        dialog.Destroy()
        
        
    def onTiming(self,event):
        if self.timing_window == None:
            self.timing_window = TimingWindow(self, wx.NewId(),"Performance")
            self.timing_window.CenterOnParent()
            self.timing_window.Show(True)
            self.timing_window.Bind(wx.EVT_CLOSE, self.onCloseTiming, id=self.timing_window.GetId())

        else:
            self.timing_window.Show(True)
            self.timing_window.Raise()
        
        
    def onCloseTiming(self,event):
        self.timing_window.Destroy()
        self.timing_window = None
        
        
class TimingWindow(wx.Frame):
    '''
    This window displays the timing information for FaceL.
    '''
    
    def __init__(self,parent,id,name,size=(400,200)):
        wx.Frame.__init__(self,parent,id,name,size=size)
        
        # Setup timing gauges
        self.process_label = wx.StaticText(self, wx.NewId(), "Process", style=wx.ALIGN_LEFT)
        self.time_label = wx.StaticText(self, wx.NewId(), "Time", style=wx.ALIGN_LEFT)
        self.relative_label = wx.StaticText(self, wx.NewId(), "Relative CPU Usage", style=wx.ALIGN_LEFT)
        
        
        self.detect_gauge = wx.Gauge(self,wx.NewId(), 100, size=wx.Size(150,20) )
        self.eye_gauge = wx.Gauge(self,wx.NewId(), 100, size=wx.Size(150,20) )
        self.label_gauge = wx.Gauge(self,wx.NewId(), 100, size=wx.Size(150,20) )
        self.other_gauge = wx.Gauge(self,wx.NewId(), 100, size=wx.Size(150,20) )
        
        self.detect_label = wx.StaticText(self, wx.NewId(), "Face Detect Time:", style=wx.ALIGN_LEFT)
        self.eye_label = wx.StaticText(self, wx.NewId(), "Eye Locate Time:", style=wx.ALIGN_LEFT)
        self.label_label = wx.StaticText(self, wx.NewId(), "Face Label Time:", style=wx.ALIGN_LEFT)
        self.other_label = wx.StaticText(self, wx.NewId(), "Other Processing:", style=wx.ALIGN_LEFT)
        
        self.detect_text = wx.StaticText(self, wx.NewId(), size = wx.Size(50,20), style= wx.ALIGN_RIGHT)    
        self.eye_text = wx.StaticText(self, wx.NewId(), size = wx.Size(50,20), style= wx.ALIGN_RIGHT)    
        self.label_text = wx.StaticText(self, wx.NewId(), size = wx.Size(50,20), style= wx.ALIGN_RIGHT)    
        self.other_text = wx.StaticText(self, wx.NewId(), size = wx.Size(50,20), style= wx.ALIGN_RIGHT)  

        self.total_label = wx.StaticText(self, wx.NewId(),"Total Time:", size = wx.Size(-1,16), style=wx.ALIGN_LEFT)
        self.total_text = wx.StaticText(self, wx.NewId(), size = wx.Size(30,16), style= wx.ALIGN_RIGHT)  
        
        self.fps_label = wx.StaticText(self, wx.NewId(), "Frames/Sec:", size = wx.Size(-1,16), style=wx.ALIGN_LEFT)
        self.fps_text = wx.StaticText(self, wx.NewId(), size = wx.Size(30,16), style= wx.ALIGN_RIGHT)  

        # Setup Sizers
        timing_sizer = wx.FlexGridSizer(4,3,4,4)

        timing_sizer.Add(self.process_label)
        timing_sizer.Add(self.time_label)
        timing_sizer.Add(self.relative_label)
        
        timing_sizer.Add(self.detect_label)
        timing_sizer.Add(self.detect_text)
        timing_sizer.Add(self.detect_gauge)
        
        timing_sizer.Add(self.eye_label)
        timing_sizer.Add(self.eye_text)
        timing_sizer.Add(self.eye_gauge)
        
        timing_sizer.Add(self.label_label)
        timing_sizer.Add(self.label_text)
        timing_sizer.Add(self.label_gauge)
        
        timing_sizer.Add(self.other_label)
        timing_sizer.Add(self.other_text)
        timing_sizer.Add(self.other_gauge)
                
        fps_sizer = wx.BoxSizer(wx.HORIZONTAL)

        fps_sizer.Add(self.total_label, flag = wx.ALIGN_CENTER | wx.ALL, border=4)
        fps_sizer.Add(self.total_text, flag = wx.ALIGN_CENTER | wx.ALL, border=4)
        fps_sizer.AddSpacer(20)
        fps_sizer.Add(self.fps_label, flag = wx.ALIGN_CENTER | wx.ALL, border=4)
        fps_sizer.Add(self.fps_text, flag = wx.ALIGN_CENTER | wx.ALL, border=4)
 
        status_sizer = wx.BoxSizer(wx.VERTICAL)

        status_sizer.Add(timing_sizer, flag = wx.ALIGN_CENTER | wx.ALL, border=4)
        status_sizer.Add(fps_sizer, flag = wx.ALIGN_CENTER | wx.ALL, border=4)

        self.SetSizer(status_sizer)
                
    def update(self,detect_time,eye_time,label_time,full_time):
        '''
        Update the timing display.
        '''
        other_time = full_time - detect_time - eye_time - label_time
        
        self.detect_gauge.SetValue(100.0*detect_time/full_time)
        self.eye_gauge.SetValue(100.0*eye_time/full_time)
        self.label_gauge.SetValue(100.0*label_time/full_time)
        self.other_gauge.SetValue(100.0*other_time/full_time)

        self.detect_text.SetLabel("%0.1fms"%(1000.0*detect_time,))
        self.eye_text.SetLabel("%0.1fms"%(1000.0*eye_time,))
        self.label_text.SetLabel("%0.1fms"%(1000.0*label_time,))
        self.other_text.SetLabel("%0.1fms"%(1000.0*other_time,))

        self.total_text.SetLabel("%0.0fms"%(1000.0*full_time,))
        self.fps_text.SetLabel("%0.1ffps"%(1/full_time,))

        
        
        
class SVMTuningDialog(wx.Dialog):
    '''
    This window contains controls for tuning the SVM
    '''
    
    def __init__(self,parent, mode, C, Gamma):
        '''
        Create and initialize the window and controls.
        '''
        wx.Dialog.__init__(self,parent,wx.NewId(),"SVM Tuning",style=wx.CAPTION)
        
        assert mode in [SVM_MANUAL,SVM_AUTOMATIC]
        C = float(C)
        Gamma = float(Gamma)
        
        self.mode = mode
        self.C = C
        self.Gamma = Gamma
        
        # Add Controls
        self.svm_tuning = wx.RadioBox(self,wx.NewId(),'Tuning Mode', choices=['Manual','Automatic'],
                         style=wx.RA_SPECIFY_COLS)
        
        if self.mode == SVM_MANUAL:
            self.svm_tuning.SetStringSelection('Manual')
        else:
            self.svm_tuning.SetStringSelection('Automatic')

        
        self.C_label = wx.StaticText(self, wx.NewId(), "C", style=wx.ALIGN_LEFT)
        self.C_slider = wx.Slider(self, wx.NewId(), 5 , -5, 15, size = wx.Size(120,25))
        self.C_text = wx.StaticText(self, wx.NewId(), size = wx.Size(75,25), style= wx.TE_RIGHT )  

        self.G_label = wx.StaticText(self, wx.NewId(), "Gamma", style=wx.ALIGN_LEFT)
        self.G_slider = wx.Slider(self, wx.NewId(), -3 , -15, 3, size = wx.Size(120,25))
        self.G_text = wx.StaticText(self, wx.NewId(), size = wx.Size(75,25), style= wx.TE_RIGHT )  
        
        self.reset_button     = wx.Button(self, wx.NewId(), "Reset Defaults")
        
        self.ok_button     = wx.Button(self, wx.NewId(), "OK")
        self.cancel_button = wx.Button(self, wx.NewId(), "Cancel")
        
        # Add Sizers
        slider_sizer = wx.FlexGridSizer(2,3,4,4)
        slider_sizer.Add(self.C_label)
        slider_sizer.Add(self.C_slider)
        slider_sizer.Add(self.C_text)
        slider_sizer.Add(self.G_label)
        slider_sizer.Add(self.G_slider)
        slider_sizer.Add(self.G_text)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add(self.ok_button, flag = wx.ALIGN_CENTER | wx.ALL, border=4)
        button_sizer.Add(self.cancel_button, flag = wx.ALIGN_CENTER | wx.ALL, border=4)
        
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.svm_tuning,   flag = wx.ALIGN_CENTER | wx.ALL, border=4)
        main_sizer.Add(slider_sizer,      flag = wx.ALIGN_CENTER | wx.ALL, border=4)
        main_sizer.Add(self.reset_button, flag = wx.ALIGN_CENTER | wx.ALL, border=4)
        main_sizer.Add(button_sizer,      flag = wx.ALIGN_CENTER | wx.ALL, border=4)

        self.SetAutoLayout(True)
        self.SetSizer(main_sizer)
        self.Layout()

        # Connect Events
        self.Bind(wx.EVT_RADIOBOX, self.OnMode, id=self.svm_tuning.GetId())
        
        self.Bind(wx.EVT_SCROLL,   self.OnTune, id=self.C_slider.GetId())
        self.Bind(wx.EVT_SCROLL,   self.OnTune, id=self.G_slider.GetId())
    
        self.Bind(wx.EVT_BUTTON,   self.OnReset, id=self.reset_button.GetId())
        self.Bind(wx.EVT_BUTTON,   self.OnOk, id=self.ok_button.GetId())
        self.Bind(wx.EVT_BUTTON,   self.OnCancel, id=self.cancel_button.GetId())

        
        self.SetTuning(self.C,self.Gamma)
        
    def OnTune(self,event):
        '''
        Process changes in the sliders.
        '''
        self.C = 2.0 ** self.C_slider.GetValue()
        self.C_text.SetLabel("%8.2e"%self.C)

        self.Gamma = 2.0 ** self.G_slider.GetValue()
        self.G_text.SetLabel("%8.2e"%self.Gamma)
        
    def SetTuning(self,C = 4.000e+00, Gamma = 9.766e-04):
        '''
        Setup the tuning parameters.
        
        If no arguments are passed defaults are selected.
        '''
        self.C = C
        self.Gamma = Gamma
        
        self.C_slider.SetValue(int(round(math.log(C,2))))
        self.C_text.SetLabel("%8.2e"%C)

        self.G_slider.SetValue(int(round(math.log(Gamma,2.0))))
        self.G_text.SetLabel("%8.2e"%Gamma)

    def OnMode(self,event):
        '''
        Change the tuning mode for the SVM to MANUAL or AUTOMATIC.
        '''
        if self.svm_tuning.GetStringSelection() == "Manual":
            self.mode = SVM_MANUAL
        else:
            self.mode = SVM_AUTOMATIC

    def OnReset(self,event):
        '''
        Reset to default tuning.
        '''
        self.svm_tuning.SetStringSelection('Manual')
        self.mode = SVM_MANUAL
        self.SetTuning()
        
    def OnOk(self,event):
        '''
        Finish the modal behavior.
        '''
        self.EndModal(wx.ID_OK)

    def OnCancel(self,event):
        '''
        Finish the modal behavior.
        '''
        self.EndModal(wx.ID_CANCEL)



class TraceBackDialog(wx.Dialog):
    '''
    This dialog displays an error message and the trace back from an exception.
    '''
    def __init__(self,parent, title, message, trace):
        wx.Dialog.__init__(self,parent,wx.NewId(),title,style=wx.CAPTION,size=wx.Size(500,400))
        
        self.message = wx.StaticText(self, wx.NewId(), message, size=wx.Size(400,-1))
        
        self.trace_box = wx.TextCtrl(self, wx.NewId(), value = trace, size=wx.Size(600,200),
                   style = wx.HSCROLL | wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_LEFT)

        self.ok_button     = wx.Button(self, wx.NewId(), "OK")
       
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.message,flag=wx.ALIGN_CENTER | wx.ALL,border=20)
        main_sizer.Add(self.trace_box,   flag = wx.ALIGN_CENTER | wx.ALL, border=20)
        main_sizer.Add(self.ok_button,      flag = wx.ALIGN_CENTER | wx.ALL, border=20)

        self.SetAutoLayout(True)
        self.SetSizer(main_sizer)
        self.Layout()
        
        self.SetSize(self.GetEffectiveMinSize())
        self.Layout()

        # Connect Events
        self.Bind(wx.EVT_BUTTON,   self.OnOk, id=self.ok_button.GetId())

    def OnOk(self,event):
        self.EndModal(wx.ID_OK)



def runDemos(name='CSU FaceL'):
    '''
    Start FaceL
    '''
    app = wx.PySimpleApp()
    frame = VideoWindow(None, wx.ID_ANY, name)
    frame.Show(True)
    app.MainLoop()
    
if __name__ == '__main__':
    try:
        runDemos()
    except SystemExit:
        raise
    except:
        trace = traceback.format_exc()
        message = TraceBackDialog(None, "Unknown Error", UNKNOWN_ERROR, trace)
        message.ShowModal()
        
        sys.stderr.write("FaceL Error: an unknown error occurred.  Details follow.\n\n")
        sys.stderr.write(trace)
        sys.exit(UNKNOWN_ERROR_CODE)
        
        
        
        