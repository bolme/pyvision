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

from math import *
from pyvision.types.Image import Image
import scipy as sp
import numpy as np

def normalizeMeanStd(matrix):
    ''' TODO: deprecated please use meanStd.'''
    print '''normalizeMeanStd is deprecated.  Please call as normalize.meanStd'''
    return meanStd(matrix)

def meanStd(matrix):
    ''' zero mean, one standard deviation '''
    is_image = False
    if isinstance(matrix,Image):
        matrix = matrix.asMatrix2D()
        is_image = True
    # Otherwize, assume it is a numpy matrix
    matrix = matrix - matrix.mean()
    matrix = (1.0/matrix.std()) * matrix
    if is_image:
        return Image(matrix)
    return matrix

def meanUnit(matrix):
    ''' zero mean, unit length '''
    is_image = False
    if isinstance(matrix,Image):
        matrix = matrix.asMatrix2D()
        is_image = True
    matrix = matrix - matrix.mean()
    length = sqrt( (matrix*matrix).sum() )
    matrix = (1.0/length) * matrix
    if is_image:
        return Image(matrix)
    return matrix

def unit(matrix):
    ''' unit length '''
    is_image = False
    if isinstance(matrix,Image):
        matrix = matrix.asMatrix2D()
        is_image = True
    length = sqrt( (matrix*matrix).sum() )
    matrix = (1.0/length) * matrix
    if is_image:
        return Image(matrix)
    return matrix

def selfQuotient(matrix,sigma=1.5):
    '''
    Compute a self quotient image.
    
    Based on work by Wang et.al. "Self Quotient Image for Face Recognition" ICIP 2004
    '''
    is_image = False
    if isinstance(matrix,Image):
        matrix = matrix.asMatrix2D()
        is_image = True

    denom = sp.ndimage.gaussian_filter(matrix,sigma)

    matrix = matrix/denom

    if is_image:
        return Image(matrix)
    return matrix


def selfQuotientPinto(tile, ilog=None):
    '''
    Compute a 3x3 Norm quotient image.
    
    This was found in Pinto et.al. CVPR 2009.  It may also be related
    to Wang et.al. "Self Quotient Image for Face Recognition" ICIP 2004.
    Code adapted from Yui Man Lui.
    '''
    # get the matrix
    is_image = False
    if isinstance(tile,Image):
        mat = tile.asMatrix2D()
        is_image = True
    else:
        # assume this is a matrix
        mat = tile
    
    
    # compute the 3x3 square
    mat2 = mat*mat #square
    
    # compute the 3x3 sums
    weights = sp.array([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]])
    mat2sum = sp.ndimage.convolve(mat2,weights)
    
    # take the sqrt for the norm
    matnorm = np.sqrt(mat2sum)
    
    # elementwize division
    normed = mat/matnorm
    
    # if ilog, save out data
    if ilog != None:
        gray = mat/mat.max()
        ilog.log(pv.Image(gray),label="PintoV1_Norm3x3_Grayscale")
        #ilog.log(pv.Image(mat2),label="Norm3x3_Squared")
        #ilog.log(pv.Image(mat2sum),label="Norm3x3_Sum")
        #ilog.log(pv.Image(matnorm),label="Norm3x3_Sqrt")
        ilog.log(pv.Image(normed),label="PintoV1_Norm3x3_Normed")
        
    # return the result
    if is_image:
        return Image(normed)
    return normed

def highPassFilter(matrix,sigma):
    '''
    This function computes a high and low pass filter.  This can be used 
    to reduce the effect of lighting.
    
    A low pass image is first computed by convolving the image with a 
    Gausian filter of radius sigma.  Second, a high pass image is computed
    by subtracting the low pass image from the original image.  This means that 
    the original image can be reconstructed by adding a low pass image and a high
    pass image.
    
    @returns: high_pass_image
    '''
    is_image = False
    if isinstance(matrix,Image):
        matrix = matrix.asMatrix2D()
        is_image = True

    matrix = matrix - sp.ndimage.gaussian_filter(matrix,sigma)
    
    if is_image:
        return Image(matrix)
    return matrix

def lowPassFilter(matrix,sigma):
    '''
    This function computes a low pass filter.  It basically smoothes the image 
    by convolving with a Gaussian.  This is often used to reduce the effect of 
    noise in images or to reduce the effect of small registration errors.  
 
    @returns: an pv.Image set from a numpy matrix if input was an image or a numpy 
                matrix otherwize. 
    '''
    is_image = False
    if isinstance(matrix,Image):
        matrix = matrix.asMatrix2D()
        is_image = True

    matrix = sp.ndimage.gaussian_filter(matrix,sigma)
    
    if is_image:
        return Image(matrix)
    return matrix


def bandPassFilter(matrix,sigma_low, sigma_high):
    '''
    This function computes a high and low pass filter.  This can be used 
    to reduce the effect of lighting.
    
    A low pass image is first computed by convolving the image with a 
    Gausian filter of radius sigma.  Second, a high pass image is computed
    by subtracting the low pass image from the original image.  This means that 
    the original image can be reconstructed by adding a low pass image and a high
    pass image.
    
    @returns: high_pass_image
    '''
    assert sigma_low > sigma_high
    is_image = False
    if isinstance(matrix,Image):
        matrix = matrix.asMatrix2D()
        is_image = True

    matrix = sp.ndimage.gaussian_filter(matrix,sigma_high) - sp.ndimage.gaussian_filter(matrix,sigma_low)
    
    if is_image:
        return Image(matrix)
    return matrix

