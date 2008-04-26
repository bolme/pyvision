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

__author__ = "$Author: bolme $"
__version__ = "$Rev: 625 $"
__info__ = "$Id: PhaseCorrelation.py 625 2008-03-24 20:24:37Z bolme $"
__copyright__ = "Copyright 2006 David S Bolme"

from numpy import *
from numpy.fft import fft2, ifft2
from numpy.linalg import lstsq

from pyvision.types.Image import Image
from pyvision.other.normalize import meanUnit


def PhaseCorrelation(tile1, tile2):
    '''
    Uses phase correlation to estimate the best integer displacement to align the images.
    Also fits a quadradic to the correltaion surface to determine a sub pixel estimate of 
    displacement.
    
    Returns four values as a tuple: max_corr, max_displacement, est_corr, est_displacement
    max_corr         - maximum correlation value.
    max_displacement - displacement needed to obtain the maximum correlation. (full pixel)
    est_corr         - estimated corelation value if subpixel displacement is used.
    est_displacement - estimated displacement (subpixel)
    
    see http://en.wikipedia.org/wiki/Phase_correlation
    '''
    if isinstance(tile1,Image):
        tile1 = tile1.asMatrix2D()
    else:
        tile1 = Image(tile1).asMatrix2D()
        raise TypeError("Please pass in a numpy array or a pyvision image.")

    if isinstance(tile2,Image):
        tile2 = tile2.asMatrix2D()
    else:
        tile2 = Image(tile2).asMatrix2D()    
    
    if tile1.shape != tile2.shape:
        raise ValueError("Image tiles must have the same shape. [tile1 %s] != [tile2 %s]"%(tile1.shape,tile2.shape))

    # copy the data
    tile1 = tile1.copy()
    tile2 = tile2.copy()

    # normalize the image tiles
    tile1 = meanUnit(tile1)
    tile2 = meanUnit(tile2)

    # compute the fft
    Ia = fft2(tile1)
    Ib = conjugate(fft2(tile2))

    # build the normalized cross-power spectrum
    ncs = Ia*Ib

    # build the power spectrum
    pc = real(ifft2(ncs))

    max_corr = pc.max()
    max_elem = (pc == max_corr).nonzero()
    max_elem = max_elem[0][0],max_elem[1][0]

    max_point = list(max_elem)
    if max_elem[0]*2 > tile1.shape[0]:
        max_point[0] = -tile1.shape[0] + max_elem[0]
    if max_elem[1]*2 > tile1.shape[1]:
        max_point[1] = -tile1.shape[1] + max_elem[1]

    est_corr, est_point = QuadradicEstimation(pc,max_point)

    return max_corr, max_point, est_corr, est_point



def QuadradicEstimation(pc, max_point):
    '''
    Estimate the subpixel displacement based on fitting a quadradic surface
    to area surrounding the max point.

    returns the estimated power spectrum at the subpixel point, and the displacement.
    '''
    # fit a surface to the area surrounding the point
    # ax2 + by2 + cx + dy + e = z
    def create_row(x,y):
        return [x*x, y*y, x, y, 1]

    A = []
    b = []

    x = max_point[0]
    y = max_point[1]
    A.append(create_row(x,y))
    b.append([pc[x,y]])

    x = max_point[0]+1
    y = max_point[1]
    A.append(create_row(x,y))
    b.append([pc[x,y]])

    x = max_point[0]-1
    y = max_point[1]
    A.append(create_row(x,y))
    b.append([pc[x,y]])

    x = max_point[0]
    y = max_point[1]+1
    A.append(create_row(x,y))
    b.append([pc[x,y]])

    x = max_point[0]
    y = max_point[1]-1
    A.append(create_row(x,y))
    b.append([pc[x,y]])

    A = array(A)
    b = array(b)

    lss = lstsq(A,b)
    assert lss[2] == 5 #make sure the matrix was of full rank
    x = lss[0]

    a = x[0,0]
    b = x[1,0]
    c = x[2,0]
    d = x[3,0]
    e = x[4,0]

    # find the derivitive with respect to x and y solve for f`(x,y) = 0 to find the max
    # 2ax + c = 0
    # 2by + d = 0
    x = -c/(2*a)
    y = -d/(2*b)

    # estimate the max at subpixel resolution
    emax = a*x*x + b*y*y + c*x + d*y + e

    return emax, [x,y]
    
# TODO: Add UnitTests
import unittest

class _TestPhaseCorrelation(unittest.TestCase):
    
    def test_Registration1(self):
        import os.path
        import pyvision
        from pyvision.types.Image import Image

        filename_a = os.path.join(pyvision.__path__[0],'data','test','registration1a.jpg')
        filename_b = os.path.join(pyvision.__path__[0],'data','test','registration1b.jpg')
        
        im_a = Image(filename_a)
        im_b = Image(filename_b)

        out = PhaseCorrelation(im_a,im_b)
        
        self.assertAlmostEqual(out[0],0.87382160686468002)
        self.assertEqual(out[1][0],20)
        self.assertEqual(out[1][1],20)
        self.assertAlmostEqual(out[2],0.87388092414032315)
        self.assertAlmostEqual(out[3][0],19.978881341260816)
        self.assertAlmostEqual(out[3][1],19.986396178942329)
        




