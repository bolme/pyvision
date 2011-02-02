# PyVision License
#
# Copyright (c) 2011 David S. Bolme
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
Created on Jan 11, 2011

@author: bolme
'''

import pyvision as pv
import cv
import numpy as np

HIST_HS = "HIST_HS"
HIST_RGB = "HIST_RGB"

class Histogram:
    '''Basically a wrapper around an OpenCV histogram.'''
    
    def __init__(self,hist,hist_type,nbins1,nbins2,nbins3):
        '''
        nbins* are the number of bins and should be set to None if they do not exist
        @param hist: the OpenCV histogram
        @param hist_type
        '''
        self.hist = hist
        self.hist_type = hist_type
        self.nbins1 = nbins1
        self.nbins2 = nbins2
        self.nbins3 = nbins3
        
        assert isinstance(nbins1,int) and nbins1 > 0
        
    def asMatrix(self):
        if self.nbins2 == None:
            result = np.zeros([self.nbins1])
            for i in range(self.nbins1):
                result[i] = cv.QueryHistValue_1D(self.hist,i)
            return result
        elif self.nbins3 == None:
            result = np.zeros([self.nbins1,self.nbins2])
            for i in range(self.nbins1):
                for j in range(self.nbins2):
                    result[i,j] = cv.QueryHistValue_2D(self.hist,i,j)
            return result
        else:
            result = np.zeros([self.nbins1,self.nbins2,self.nbins3])
            for i in range(self.nbins1):
                for j in range(self.nbins2):
                    for k in range(self.nbins3):
                        result[i,j,k] = cv.QueryHistValue_3D(self.hist,i,j,k)
            return result

    def rescaleMax(self,value=255):
        ''' 
        Rescale the histogram such that the maximum equals the value.
        '''
        cv.NormalizeHist(self.hist,1)
        _,max_value,_,_ = cv.GetMinMaxHistValue(self.hist)
        if max_value == 0:
            max_value = 1.0
        cv.NormalizeHist(self.hist,255/max_value)

    def rescaleSum(self,value=1.0):
        ''' 
        Rescale the histogram such that the maximum equals the value.
        '''
        cv.NormalizeHist(self.hist,value)
        
    def backProject(self,im,bg_hist=None):
        if self.hist_type == pv.HIST_HS:
            return pv.hsBackProjectHist(im,self,bg_hist)
        if self.hist_type == pv.HIST_RGB:
            return pv.rgbBackProjectHist(im,self,bg_hist)
        else:
            raise NotImplementedError("backProject not implemented for type: %s"%self.hist_type)


def hsHist(im,h_bins=32,s_bins=32,mask=None,normalize=True):
    '''
    Compute the hue saturation histogram of an image. (Based on OpenCV example code).
    
    @param im: the image
    @type im: pv.Image
    @param h_bins: the number of bins for hue.
    @type h_bins: int
    @param s_bins: the number of bins for saturation.
    @type s_bins: int
    @param mask: an image containing a mask
    @type mask: cv.Image or np.array(dtype=np.bool)
    @return: an OpenCV histogram
    @rtype: pv.Histogram
    '''
    w,h = im.size
    hsv = im.asHSV()

    # Extract the H and S planes
    h_plane = cv.CreateImage((w,h), cv.IPL_DEPTH_8U, 1)
    s_plane = cv.CreateImage((w,h), cv.IPL_DEPTH_8U, 1)
    cv.Split(hsv, h_plane, s_plane, None, None)
    planes = [h_plane, s_plane]

    # set the histogram size
    hist_size = [h_bins, s_bins]
    
    # hue varies from 0 (~0 deg red) to 180 (~360 deg red again */
    h_ranges = [0, 180]
    
    # saturation varies from 0 (black-gray-white) to
    # 255 (pure spectrum color)
    s_ranges = [0, 255]
    
    ranges = [h_ranges, s_ranges]

    # Calculate the histogram    
    hist = cv.CreateHist(hist_size, cv.CV_HIST_ARRAY, ranges, 1)
    if mask != None:
        mask = mask.asOpenCVBW()
    cv.CalcHist(planes, hist,mask=mask)
    
    return pv.Histogram(hist,HIST_HS,h_bins,s_bins,None)


def rgbHist(im,r_bins=8,g_bins=8,b_bins=8,mask=None,normalize=True):
    '''
    Compute the hue saturation histogram of an image. (Based on OpenCV example code).
    
    @param im: the image
    @type im: pv.Image
    @param h_bins: the number of bins for hue.
    @type h_bins: int
    @param s_bins: the number of bins for saturation.
    @type s_bins: int
    @param mask: an image containing a mask
    @type mask: cv.Image or np.array(dtype=np.bool)
    @return: an OpenCV histogram
    @rtype: pv.Histogram
    '''
    w,h = im.size
    bgr = im.asOpenCV()

    # Extract the H and S planes
    b_plane = cv.CreateImage((w,h), cv.IPL_DEPTH_8U, 1)
    g_plane = cv.CreateImage((w,h), cv.IPL_DEPTH_8U, 1)
    r_plane = cv.CreateImage((w,h), cv.IPL_DEPTH_8U, 1)
    cv.Split(bgr, b_plane, g_plane, r_plane, None)
    planes = [b_plane, g_plane, r_plane]

    # set the histogram size
    hist_size = [b_bins, g_bins, r_bins]
    
    # pixel value ranges
    b_ranges = [0, 255]
    g_ranges = [0, 255]
    r_ranges = [0, 255]
    
    ranges = [b_ranges, g_ranges, r_ranges]

    # Calculate the histogram    
    hist = cv.CreateHist(hist_size, cv.CV_HIST_ARRAY, ranges, 1)
    if mask != None:
        mask = mask.asOpenCVBW()
    cv.CalcHist(planes, hist, mask=mask)
    
    return pv.Histogram(hist,HIST_RGB,b_bins,g_bins,r_bins)


def hsBackProjectHist(im,fg_hist,bg_hist=None):
    '''
    Compute the hue saturation histogram of an image. (Based on OpenCV example code).
    
    @param im: the image
    @type im: pv.Image
    @param fg_hist: the histogram
    @type fg_hist: pv.Histogram
    @param bg_hist:
    @type bg_hist: pv.Histogram
    @return: an OpenCV histogram
    @rtype: pv.Image
    '''
    w,h = im.size
    hsv = im.asHSV()
    
    if bg_hist != None:
        # set the histogram size
        hist_size = [fg_hist.nbins1, fg_hist.nbins2]
        
        # pixel value ranges
        h_ranges = [0, 180]
        s_ranges = [0, 255]
        ranges = [h_ranges, s_ranges]
        
        # Calculate the histogram    
        prob_hist = cv.CreateHist(hist_size, cv.CV_HIST_ARRAY, ranges, 1)
        
        fg_hist.rescaleMax(255)
        bg_hist.rescaleMax(255)
        
        cv.CalcProbDensity(bg_hist.hist, fg_hist.hist, prob_hist, scale=255) 
        
        fg_hist = pv.Histogram(prob_hist,pv.HIST_HS,fg_hist.nbins1, fg_hist.nbins2, None)
    
    # Extract the H and S planes
    h_plane = cv.CreateImage((w,h), cv.IPL_DEPTH_8U, 1)
    s_plane = cv.CreateImage((w,h), cv.IPL_DEPTH_8U, 1)
    cv.Split(hsv, h_plane, s_plane, None, None)
    planes = [h_plane, s_plane]

    output = cv.CreateImage((w,h), cv.IPL_DEPTH_8U, 1)
    
    # Normalize the histogram
    fg_hist.rescaleMax(255)

    cv.CalcBackProject(planes,output,fg_hist.hist)
    
    return pv.Image(output)

def rgbBackProjectHist(im,fg_hist,bg_hist=None):
    '''
    Compute the hue saturation histogram of an image. (Based on OpenCV example code).
    
    @param im: the image
    @type im: pv.Image
    @param fg_hist: the histogram
    @type fg_hist: pv.Histogram
    @param bg_hist:
    @type bg_hist: pv.Histogram
    @return: an OpenCV histogram
    @rtype: pv.Image
    '''
    w,h = im.size
    bgr = im.asOpenCV()

    if bg_hist != None:
        # set the histogram size
        hist_size = [fg_hist.nbins1, fg_hist.nbins2,fg_hist.nbins3]
        
        # pixel value ranges
        b_ranges = [0, 255]
        g_ranges = [0, 255]
        r_ranges = [0, 255]
        ranges = [b_ranges, g_ranges, r_ranges]
        
        # Calculate the histogram    
        prob_hist = cv.CreateHist(hist_size, cv.CV_HIST_ARRAY, ranges, 1)
        
        fg_hist.rescaleMax(255)
        bg_hist.rescaleMax(255)
        
        cv.CalcProbDensity(bg_hist.hist, fg_hist.hist, prob_hist, scale=255) 
        
        fg_hist = pv.Histogram(prob_hist,pv.HIST_HS,fg_hist.nbins1, fg_hist.nbins2, None)

    # Extract the H and S planes
    b_plane = cv.CreateImage((w,h), cv.IPL_DEPTH_8U, 1)
    g_plane = cv.CreateImage((w,h), cv.IPL_DEPTH_8U, 1)
    r_plane = cv.CreateImage((w,h), cv.IPL_DEPTH_8U, 1)
    cv.Split(bgr, b_plane, g_plane, r_plane, None)
    planes = [b_plane, g_plane, r_plane]

    output = cv.CreateImage((w,h), cv.IPL_DEPTH_8U, 1)
    
    # Normalize the histogram
    fg_hist.rescaleMax(255)

    cv.CalcBackProject(planes,output,fg_hist.hist)
    
    return pv.Image(output)

import unittest
import os.path
class ColorTest(unittest.TestCase):
    
    def test01_hsHist(self):
        '''Test hue saturation histogram.'''
        ilog = pv.ImageLog()
        im = pv.Image(os.path.join(pv.__path__[0],"data","misc","baboon.jpg"))
        mask = np.zeros((512,512),dtype=np.bool)
        mask[150:200,128:300] = True
        m = pv.Image(1.0*mask)
        ilog(im)
        ilog(m)
        hist = hsHist(im,mask=m)
        print hist
        print dir(hist)
        #print dir(hist.bins),hist.bins.channels
        #for i in range(32):
        #    for j in range(30):
        #        print i,j,cv.QueryHistValue_2D(hist,j,i)
        hist.rescaleMax(255)
        
        print hist.asMatrix()
        #print cv.SetHistBinRanges
        back = hist.backProject(im)
        ilog(back)
        
        ilog.show()
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main(testRunner = unittest.TextTestRunner(verbosity=2))