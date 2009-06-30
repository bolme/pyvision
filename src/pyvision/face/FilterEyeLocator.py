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

'''
This is a simplified correlation filter implementation used to locate eyes
using ASEF correlation filters.  This file contains two classes: 
OpenCVFilterEyeLocator and FilterEyeLocator.  The first is the bare minimum 
required to locate eye and only requires opencv.  The second need is a wrapper
that includes a nice pyvision compatible interface.  This class is not integrated with 
PyVision.  PyVision supplies an interface to this class which cleans
up the interface and provides a bridge to many of the PyVision data
structures.
'''

import opencv as cv
import math
import struct
import array
import os.path
import pyvision as pv
import numpy as np
import sys

__author__ = "David S. Bolme - Colorado State Univeristy"
__version__ = "$Revision: 729 $"

if pv.WARN_COMMERCIAL_USE:
    warning = '''
    WARNING: A patent protection is anticipated for ASEF and 
             similar filters by the Colorado State University 
             Research Foundation (CSURF). 
       
             This module, "FilterEyeLocator.py", my not be 
             suitable for commercial use.
    
             Commercial and government users should contact 
             CSURF for additional details:
             http://www.csurf.org/tto/pdfs/ncs_forms/09-017_csurf_ncs.pdf
    '''
    sys.stderr.write(warning)
    print pv.WARN_COMMERCIAL_USE

#TODO: Unknown error - THis may be related version 1.0.0 of opencv
#Traceback (most recent call last):
#  File "/home/dbolme/ASEFFilters/python/csu/tools/face_scan.py", line 117, in ?
#    results = processFaces(im,face_detect,locate_eyes)
#  File "/home/dbolme/ASEFFilters/python/csu/tools/face_scan.py", line 57, in processFaces
#    eye1, eye2, corr1, corr2 = locate_eyes.locateEyes(cv_im)
#  File "/home/dbolme/ASEFFilters/python/csu/face/FilterEyeLocator.py", line 185, in locateEyes
#    leye = cv.cvMinMaxLoc(self.left_roi)[3]
#IndexError: list index out of range

#TODO: Add a quality estimate

def saveFilterEyeLocator(filename, el, comment="",copyright=""):
    '''
    File Format
        - Line 1: CFEL 
        - Line 2: <comment>
        - Line 3: <copyright>
        - Line 4: ROWS COLS
        - Line 5: LEFT_RECT
        - Line 6: RIGHT_RECT
        - Line 7: BYTE_ORDER(0x41424344 or 'ABCD')
        - Line 8: <binary data: two single precision floating point arrays of 4*WIDTH*HEIGHT bytes)
    '''
    r,c = el.left_filter.rows,el.left_filter.cols
    
    f = open(filename,'wb')
    f.write("CFEL\n")
    f.write(comment.strip()+"\n")
    f.write(copyright.strip()+"\n")
    f.write("%d %d\n"%(r,c))
    f.write("%d %d %d %d\n"%(el.left_rect.x,el.left_rect.y,el.left_rect.width,el.left_rect.height))
    f.write("%d %d %d %d\n"%(el.right_rect.x,el.right_rect.y,el.right_rect.width,el.right_rect.height))
    f.write("%s\n"%struct.pack("i",0x41424344))
    
    assert len(el.left_filter.imageData) == 4*r*c
    f.write(el.left_filter.imageData)
    
    assert len(el.right_filter.imageData) == 4*r*c
    f.write(el.right_filter.imageData)    
    

def loadFilterEyeLocator(filename,ilog=None):
    '''
    Loads the eye locator from a file.'
    '''
    
    # open the file
    f = open(filename,'rb')
    
    # Check the first line
    line = f.readline().strip()
    assert line == "CFEL"
    
    # read past the comment and copyright.
    f.readline()
    f.readline()
    
    # get the width and the height
    r,c = f.readline().split()
    r,c = int(r),int(c)
    
    # read in the left bounding rectangle
    x,y,w,h = f.readline().split()
    left_rect = cv.cvRect(int(x),int(y),int(w),int(h))
    
    # read in the right bounding rectangle
    x,y,w,h = f.readline().split()
    right_rect = cv.cvRect(int(x),int(y),int(w),int(h))
    
    # read the magic number
    magic_number = f.readline().strip()
    assert len(magic_number) == 4
    magic_number = struct.unpack('i',magic_number)[0]
    
    # Read in the filter data
    lf = array.array('f')
    rf = array.array('f')
    
    lf.fromfile(f,r*c)
    rf.fromfile(f,r*c)
    
    # Test the magic number and byteswap if necessary.
    if magic_number == 0x41424344:
        pass
    elif magic_number == 0x44434241:
        lf.byteswap()
        rf.byteswap()
    else:
        raise ValueError("Bad Magic Number: Unknown byte ordering in file")
    
    # Create the left and right filters
    left_filter  = cv.cvCreateMat(r,c,cv.CV_32F)
    right_filter = cv.cvCreateMat(r,c,cv.CV_32F)
    
    # Copy data into the left and right filters
    left_filter.imageData  = lf.tostring()
    right_filter.imageData = rf.tostring()
    
    tmp = pv.OpenCVToNumpy(left_filter)
    t1 = tmp.mean()
    t2 = tmp.std()
    cv.cvScale(left_filter,left_filter,1.0/t2,-t1*1.0/t2)

    tmp = pv.OpenCVToNumpy(right_filter)
    t1 = tmp.mean()
    t2 = tmp.std()
    cv.cvScale(right_filter,right_filter,1.0/t2,-t1*1.0/t2)

    #tmp = pv.OpenCVToNumpy(left_filter)
    #print tmp.mean(),tmp.std()
    
    if ilog != None:
        #lf = cv.cvCreateMat(r,c,cv.CV_8U)
        #rf = cv.cvCreateMat(r,c,cv.CV_8U)
        
        lf = pv.OpenCVToNumpy(left_filter)
        rf = pv.OpenCVToNumpy(right_filter)
        
        lf = np.fft.fftshift(lf).transpose()
        rf = np.fft.fftshift(rf).transpose()
        
        ilog.log(pv.Image(lf),label="LeftEyeFilter")
        ilog.log(pv.Image(rf),label="RightEyeFilter")
    
    # Return the eye locator
    return OpenCVFilterEyeLocator(left_filter,right_filter,left_rect,right_rect)
    
    
class OpenCVFilterEyeLocator:
    '''    
    This class is used for someone only interested in locating the eyes in an 
    image using correlation filters.  This class does not include any support
    for training correlation filters.  For training see ASEF.  This class 
    is written only using OpenCV and is much faster than the ASEF class.
    
    For details see the paper:
    
    David S. Bolme, Bruce A. Draper, and J. Ross Beveridge. Average of 
    Synthetic Exact Filters. Submitted to Computer Vision and Pattern 
    Recoginition. 2009.
    
    The class uses two ASEF filters to find the eyes.  The eyes are located by 
    first computing the correlation of the face tile with each filter.  The 
    max value from the correlation plain is returned as the eye coordinate.
    Also returned is the full correlation output from the image.
    
    The images are normalized by computing log transforming the pixel values
        
    To improve performance, this class is not thread safe.  The class reuses 
    data arrays allocated for each call to use this class for multiple threads
    you should create an instance for each threads.  Also note that each method
    call may overwrite arrays returned by this application.  So if you need 
    the returned data to persist be sure to create a copy. 
    
        - Left and right eyes are in relation to the location in the image.
    '''
    
    
    def __init__(self,left_filter,right_filter, left_rect, right_rect):
        '''
        @param left_filter: is in the Fourier domain where the left eye 
                corresponds to the real output and the right eye corresponds to 
                the imaginary output
        '''
        # Check the input to this function
        r,c = left_filter.rows,left_filter.cols
        
        assert left_filter.width == right_filter.width
        assert left_filter.height == right_filter.height
        assert left_filter.nChannels == 1
        assert right_filter.nChannels == 1
        
        # Create the arrays needed for the computation
        self.left_filter      = cv.cvCreateMat(r,c,cv.CV_32F)
        self.right_filter     = cv.cvCreateMat(r,c,cv.CV_32F)
        self.left_filter_dft  = cv.cvCreateMat(r,c,cv.CV_32F)
        self.right_filter_dft = cv.cvCreateMat(r,c,cv.CV_32F)
        self.image            = cv.cvCreateMat(r,c,cv.CV_32F)
        self.left_corr        = cv.cvCreateMat(r,c,cv.CV_32F)
        self.right_corr       = cv.cvCreateMat(r,c,cv.CV_32F)
        
        # Populate the spatial filters
        cv.cvConvertScale(left_filter,  self.left_filter)
        cv.cvConvertScale(right_filter, self.right_filter)

        # Compute the filters in the Fourier domain
        cv.cvDFT(self.left_filter,  self.left_filter_dft,  cv.CV_DXT_FORWARD)
        cv.cvDFT(self.right_filter, self.right_filter_dft, cv.CV_DXT_FORWARD)
        
        # Set up correlation region of interest
        self.left_rect = left_rect
        self.right_rect = right_rect

        self.left_roi = cv.cvGetSubRect(self.left_corr,self.left_rect)
        self.right_roi = cv.cvGetSubRect(self.right_corr,self.right_rect)
        
        # Create the look up table for the log transform
        self.lut = cv.cvCreateMat(256,1,cv.CV_32F)
        
        for i in range(256):
            self.lut[i,0] = math.log(i+1)


    def locateEyes(self,image_tile):
        '''
        @param image_tile: is an 32-bit gray scale opencv image tile of a face 
                that is the same size as the filter
        @type image_tile: 8-bit gray scale opencv image
        
        @returns: a tuple consisting of the location of the left and right eyes
                (opencv 2D points), and the complex correlation plain output
                
        @raises AssertionError: is raised if the image is not 8-bit or not the
                same size as the filter
        '''
        self.correlate(image_tile)
        
        leye = cv.cvMinMaxLoc(self.left_roi)[3]
        leye = cv.cvPoint(self.left_rect.x+leye.x,self.left_rect.y+leye.y)

        reye = cv.cvMinMaxLoc(self.right_roi)[3]
        reye = cv.cvPoint(self.right_rect.x+reye.x,self.right_rect.y+reye.y)
        
        return leye,reye,self.left_corr,self.right_corr

        
    def _preprocess(self,image_tile):
        '''
        preprocess an image tile.
        '''
        cv.cvLUT(image_tile,self.image,self.lut)
        
        return self.image
        
        
    def correlate(self,image_tile):
        '''
        Correlate the image with the left and right filters.
        '''
        self._preprocess(image_tile)
        
        cv.cvDFT(self.image,  self.image,  cv.CV_DXT_FORWARD)
        
        cv.cvMulSpectrums( self.image, self.left_filter_dft, self.left_corr, cv.CV_DXT_MUL_CONJ )
        cv.cvMulSpectrums( self.image, self.right_filter_dft, self.right_corr, cv.CV_DXT_MUL_CONJ )
        
        cv.cvDFT(self.left_corr,self.left_corr,cv.CV_DXT_INV_SCALE)
        cv.cvDFT(self.right_corr,self.right_corr,cv.CV_DXT_INV_SCALE)
        
        return self.left_corr,self.right_corr


class FilterEyeLocator:
    '''
    This class provides a PyVision interface to the ASEF eye locator.
    '''
    
    def __init__(self,filename=None,ilog=None):
        '''
        Load the eye detector from the file.
        '''
        if filename == None:
            filename = os.path.join(pv.__path__[0],"config","EyeLocatorASEF128x128.fel")
            
        self.fel = loadFilterEyeLocator(filename,ilog=ilog)
        
        self.bwtile = cv.cvCreateMat(128,128,cv.CV_8U)
            
        
    def __call__(self,im,face_rects,ilog=None):
        return self.locateEyes(im,face_rects,ilog=ilog)
        
        
    def locateEyes(self,im,face_rects,ilog=None):
        '''
        Finds the eyes in the image.  
        
        @param im: full sized image
        @param face_rects: list of rectangle which are the output from the cascade face detector.
        '''        
        cvim = im.asOpenCVBW()
        
        faces = []
        
        for rect in face_rects:
            faceim = cv.cvGetSubRect(cvim, rect.asOpenCV())
            cv.cvResize(faceim,self.bwtile)
            
            affine = pv.AffineFromRect(rect,(128,128))

            #cv.cvCvtColor( self.cvtile, self.bwtile, cv.CV_BGR2GRAY )
            
            leye,reye,lcp,rcp = self.fel.locateEyes(self.bwtile)
            le = pv.Point(leye)
            re = pv.Point(reye)
            
            leye = affine.invertPoint(le)
            reye = affine.invertPoint(re)
            
            faces.append([rect,leye,reye])
            
            if ilog != None:
                ilog.log(pv.Image(self.bwtile),label="FaceDetection")
                lcp = pv.OpenCVToNumpy(lcp).transpose()
                lcp = lcp*(lcp > 0.0)
                rcp = pv.OpenCVToNumpy(rcp).transpose()
                rcp = rcp*(rcp > 0.0)
                ilog.log(pv.Image(lcp),label="Left_Corr")
                ilog.log(pv.Image(rcp),label="Right_Corr")
                tmp = pv.Image(self.bwtile)
                tmp.annotatePoint(le)
                tmp.annotatePoint(re)
                ilog.log(tmp,"EyeLocations")
                
        return faces
        

#############################################################################
# Unit Tests
#############################################################################
import unittest
import pyvision.face.CascadeDetector as cd
from pyvision.analysis.FaceAnalysis.FaceDatabase import ScrapShotsDatabase
from pyvision.analysis.FaceAnalysis.EyeDetectionTest import EyeDetectionTest

class _TestFilterEyeLocator(unittest.TestCase):
    
    def test_ASEFEyeLocalization(self):
        '''
        This trains the FaceFinder on the scraps database.
        '''
        # Load a face database
        ssdb = ScrapShotsDatabase()
                
        # Create a face detector 
        face_detector = cd.CascadeDetector()

        # Create an eye locator
        eye_locator = FilterEyeLocator()
        
        # Create an eye detection test
        edt = EyeDetectionTest(name='asef_scraps')

        #print "Testing..."
        for face_id in ssdb.keys():
            face = ssdb[face_id]
            im = face.image

            # Detect the faces
            faces = face_detector.detect(im)
            
            # Detect the eyes
            pred_eyes = eye_locator(im,faces)
            
            truth_eyes = [[face.left_eye,face.right_eye]]
            pred_eyes = [ [leye,reye] for rect,leye,reye in pred_eyes]
            
            # Add to eye detection test
            edt.addSample(truth_eyes, pred_eyes, im=im, annotate=False)
        
        edt.createSummary()
        self.assertAlmostEqual( edt.face_rate ,   0.953757225434, places = 3 )
        
        self.assertAlmostEqual( edt.both25_rate , 0.797687861272, places = 3 )
        self.assertAlmostEqual( edt.both10_rate , 0.445086705202, places = 3 )
        self.assertAlmostEqual( edt.both05_rate , 0.346820809249, places = 3 )

        
    
    
        