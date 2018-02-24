# PyVision License
#
# Copyright (c) 2006-2010 David S. Bolme
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
Created on Sep 6, 2010

@author: bolme
'''


import pyvision as pv
import numpy as np
import scipy as sp
import cv
import PIL.Image

import os.path

# Unlike many other graphics libraries PyVision does not have its own native
# pixel format.  Instead it depends on other libraries to implement this array
# data, and PyVision acts as a translator.  PyVision therefore provides a
# common framework that allows PIL, SciPy and OpenCV to work together to
# solve problems.

# This tutorial looks at pv.Image as a way to convert between types used
# in the PyVision library.  pv.Images are used as a common format that can
# easily translate between PIL, Numpy, and OpenCV formats.  Under the hood
# a pv.Image is represented by one of these formats, and the image is 
# converted to new formats on demand. 

# pv.Image implements a number of translation methods that allows images to
# be obtained in many useful formats.  These methods are:
# * asMatrix2D() -> gray scale (one-channel) SciPy array
# * asMatrix3D() -> color (three-channel) SciPy array
# * asPIL() -> Python Imageing Library (PIL) image
# * asOpenCV() -> OpenCV Color Image (8-bit)
# * asOpenCVBW() -> OpenCV gray scale (Black/White) Image (8-bit)
# * asAnnotated() -> A PIL formated image which includes annotations made to this image.
# 
# The constructor for pv.Images can also take any of these formats as 
# arguments.  Therefore converting between types can be done by code such as:
# OPENCV_IMAGE = pv.Image(NUMPY_MATIRX).asOpenCV()

# In this tutorial we will demonstraite how to use pv.Image to convert images
# to different formates and in each format we will perform a simple image 
# processing task of thresholding an image to produce a black and white 
# equivalent.

if __name__ == "__main__":
    # Image logs are used to save images and other data to a directory
    # for later analysis.  These logs are valuable tools for understanding
    # the imagery and debugging algorithms.  Unless otherwise specified, 
    # ImageLogs are usually created in the directory "/tmp".
    ilog = pv.ImageLog()
    
    # Timers keep a record of the time required for algorithms to execute
    # and help determine runtimes and can determine which parts of algorithms
    # are to slow and need optimization.
    timer = pv.Timer()

    # The filename for the baboon image
    filename = os.path.join(pv.__path__[0],'data','misc','baboon.jpg')
    
    # If a string is passed a to the initializer it will assume that is a 
    # path and will read the image from that file.  The image is usually read
    # from disk using PIL and then stored as a PIL image.
    im = pv.Image(filename)
    
    # This command saves the image to an image log which provides good 
    # information for debugging.  It is often helpful to save many images
    # during a processing to make sure that each step is producing the
    # intended result.
    ilog(im,"OriginalImage")
    
    # The PIL tool box supports man image processing and graphics functions.
    # Typically PIL is used for things like reading in image files and 
    # rendering graphics on top of images for annotations.  It tends to
    # be slower than OpenCV and also lacks many of the more specialized
    # computer vision algorithms.
    
    # pv.Image objects are responsible for converting between image types.
    # This next call returns an image in PIL format that can be used with
    # the PIL library.
    pil = im.asPIL()
    
    # "mark" checks the "wall clock time" and logs program timing data. 
    timer.mark("Starting")
    
    # Convert from RGB to gray scale
    gray = pil.convert('L')
    
    # This applys the "lambda" function to each pixel which performs 
    # thresholding. Processing each pixel with a  python function is slow.
    thresh = PIL.Image.eval(gray, lambda x: 255*(x>127.5) )
    
    # Record the time for PIL thresholding.
    timer.mark("PIL")
    
    # pv.Images can also be initialized using a PIL image.  This command
    # also saves a copy of the image to the ImageLog.
    ilog(pv.Image(thresh),"PILThresh")
    
    # Scipy arrays are very easy to work with, and scipy has many image
    # processing, linear algebra, and data analysis routines. 
    
    # "asMatrix2D" returns the 2D array containing the gray scale pixel
    # values. The values in the matrix are indexed using standard pixel
    # mat[x,y] coordinates and mat.shape = (w,h).  This may be transposed
    # from what some people might expect.  Matricies are typically indexed 
    # using "rows" and "columns" so the matrix has been transposed to 
    # maintain the image-like x,y indexing.  There is also a method
    # called asMatrix3D() which returns color data in a 3D array.
    
    # A pv.Image will often maintain multiple representations of an image.
    # Calling a method like asMatrix2D will generate and return a Scipy format
    # copy of the image.  That copy will also be cached for future use so
    # multiple calls to asMatrix2D will only generate that image for the 
    # first call, and all subsequent calls will return the cached copy.  
    mat = im.asMatrix2D()    
    
    timer.mark("Starting")
    
    # Scipy syntax is often very simple and straight forward and fast.  
    # Because of this Scipy can be used to quickly prototype algorithms.
    thresh = mat > 127.5
    
    timer.mark("Scipy")
    
    # pv.Image cannot be initialized using boolean data so "1.0*" converts
    # to a floating point array.
    ilog(pv.Image(1.0*thresh),"ScipyThresh")
    
    # OpenCV code is often more complicated because images need to be 
    # allocated explicitly and function calls are more complicated, but
    # the often executes faster than scipy or PIL.  OpenCV also has many
    # useful image processing, machine learning, and computer vision 
    # algorithms that are not found in scipy or PIL.
    
    # This function returns an OpenCV gray scale image.  There is also a
    # function asOpenCV() which will return a color image.
    cvim = im.asOpenCVBW()
    
    timer.mark("Starting")
    
    # OpenCV often requires manual image/data allocation which increases code 
    # complexity but also offers more control for performance tuning and memory
    # management.
    dest = cv.CreateImage(im.size,cv.IPL_DEPTH_8U,1)
    
    # OpenCV has fast implementations for many common vision algorithms and 
    # image processing tasks. Syntax is often more confusing than PIL or Scipy.
    cv.CmpS(cvim,127.5,dest,cv.CV_CMP_GT)
    
    timer.mark("OpenCV")
    
    # Like before we convert to a pv.Image and save the image to the log.
    ilog(pv.Image(dest),"OpenCVThresh")
    
    # The timer collects a record of the time taken for each operation
    # and then displays that data in a nicely formated table.  "Current"
    # times are measured from the previous mark and show that Scipy
    # and OpenCV are very fast at this particular task.
    print(timer)
    #|---|---------------|---------------|-------------------|-----------------|-------|
    #|   | event         | time          | current           | total           | notes |
    #|---|---------------|---------------|-------------------|-----------------|-------|
    #| 0 | Timer Created | 1283833886.12 |               0.0 |             0.0 | None  |
    #| 1 | Starting      | 1283833886.21 |   0.0880968570709 | 0.0880968570709 | None  |
    #| 2 | PIL           | 1283833886.21 |   0.0013279914856 | 0.0894248485565 | None  |
    #| 3 | Starting      | 1283833886.25 |   0.0393881797791 |  0.128813028336 | None  |
    #| 4 | Scipy         | 1283833886.25 | 0.000727891921997 |  0.129540920258 | None  |
    #| 5 | Starting      | 1283833886.31 |    0.056421995163 |  0.185962915421 | None  |
    #| 6 | OpenCV        | 1283833886.31 | 0.000664949417114 |  0.186627864838 | None  |
    #|---|---------------|---------------|-------------------|-----------------|-------|

    # Timing data can also be saved to the log for further analysis
    ilog(timer,"TimingData")
    
    # This is a conveniant function that displays all images in the log.
    ilog.show()
    