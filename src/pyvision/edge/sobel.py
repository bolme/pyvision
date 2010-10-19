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

'''
Created on Oct 18, 2010
@author: Stephen O'Hara
'''
import cv
import pyvision as pv

def sobel(im,xorder=1,yorder=0,aperture_size=3,sigma=None):
    '''
    void cv.Sobel(src, dst, xorder, yorder, apertureSize = 3) 
    @param im: Input image
    @param xorder: The order of the x derivative (see cv.Sobel openCV docs) 
    @param yorder: The order of the y derivative (see cv.Sobel openCV docs)
    @param aperture_size: How large a convolution window to use
    @param sigma: Optional smoothing parameter to be applied prior to detecting edges
    '''
    gray = im.asOpenCVBW()
    edges = cv.CreateImage(cv.GetSize(gray), 8, 1)
    
    if sigma!=None:
        cv.Smooth(gray,gray,cv.CV_GAUSSIAN,int(sigma)*4+1,int(sigma)*4+1,sigma,sigma)

    #sobel requires a destination image with larger bit depth... 
    #...so we have to convert it back to 8 bit for the pv Image...
    dst32f = cv.CreateImage(cv.GetSize(gray),cv.IPL_DEPTH_32F,1) 
    
    cv.Sobel(gray, dst32f, xorder, yorder, aperture_size)
    cv.Convert(dst32f, edges)    
    edges = pv.Image(edges)
        
    return edges
    

        
