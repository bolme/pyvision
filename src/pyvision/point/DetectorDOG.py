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
Purpose: 
    Find regions of interest using the multiscale differance of 
    gaussians opereator.

Inspired by: 
    M. Brown and D. Lowe. Invariant features from interest point groups. 
    British Machine Vision Conference. 2002.

Contributed by: 
    David Bolme 2008.
'''


from pyvision.point.DetectorROI import DetectorROI
import unittest
import os.path
import pyvision
from pyvision.types.Image import Image
from pyvision.types.Point import Point
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter,maximum_filter,minimum_filter
from scipy.linalg import lstsq

from pyvision.analysis.ImageLog import ImageLog
from numpy import sqrt,array,nonzero
from numpy.lib.polynomial import polyfit


EXTREMA_SIZE=3
DEFAULT_SIGMA=sqrt(2) # from Lowe99
DEFAULT_TYPE='d'
DOG_SCALES = 2
MAX_EXTREMA = 1024

class DetectorDOG(DetectorROI):
    def __init__(self, sigma=DEFAULT_SIGMA, scales=DOG_SCALES, min_size=20, min_contrast=0.03, max_curvature_ratio=10, **kwargs):
        '''
        min_size - Image pyramid terminates when one of the image demensions reaches this size.
        '''
        DetectorROI.__init__(self,**kwargs)
        
        self.min_size = min_size
        self.scales = scales
        self.sigma = sigma
        self.min_contrast = min_contrast
        self.max_curvature_ratio = max_curvature_ratio
        
        
    
    def _detect(self,im):

        mat = im.asMatrix2D()
        
        levels = []
        while mat.shape[0] > self.min_size and mat.shape[1] > self.min_size:
            levels.append(mat)
            mat = zoom(mat,0.5)

        gaussians = []
        k = 2.0**(1.0/self.scales)
        for level in levels:
            gs = []
            sigma = self.sigma
            for each in range(self.scales+3):
                g = gaussian_filter(level,sigma)
                gs.append(g)
                sigma = k*sigma
            gaussians.append(gs)
                    
        dogs = []
        for gs in gaussians:
            ds = []
            for i in range(len(gs)-1):
                d = gs[i]-gs[i+1]
                ds.append(d)
            ds = array(ds)
            dogs.append(ds)
        #dogs = array(dogs,'d')
        
        points = []
        sigma = 2*k*self.sigma # approx 95% bounds       
        extrema = []
        scale = 1
        for ds in dogs:
            # find extrema
            mins = minimum_filter(ds,(3,3,3))
            maxs = maximum_filter(ds,(3,3,3))
                    
            # Find the extrema but not on the edges of the image
            minima = nonzero(mins[1:-1,1:-1,1:-1] == ds[1:-1,1:-1,1:-1])
            maxima = nonzero(maxs[1:-1,1:-1,1:-1] == ds[1:-1,1:-1,1:-1])    

            for i in range(len(minima[0])):
                # Correct for removing the edges in the previous step
                s = minima[0][i]+1
                x = minima[1][i]+1
                y = minima[2][i]+1
                
                # Get a 3 by 3 block
                block = ds[s-1:s+2,x-1:x+2,y-1:y+2]
                params = TaylorFit(block)
                ts,tx,ty = TaylorSubpixel(params)
                td = TaylorApprox(params,ts,tx,ty)
                ts -= 1.0
                tx -= 1.0
                ty -= 1.0
                                         
                # Only select extrema with high contrast
                if abs(td) < self.min_contrast: continue

                # Chech the ratios of the principal curvatures (see Lowe 2004 Sec 4.1:
                Dxx=2*params[1]
                Dyy=2*params[2]
                Dxy=params[3]
                TrH = Dxx+Dyy
                DetH = Dxx*Dyy-Dxy*Dxy
                
                r = self.max_curvature_ratio
                if DetH < 0: continue # discard because curvatures have different signs
                if r*TrH > DetH*(r+1)*(r+1): continue # Ratio of curvatuers is greater than R

                if not (-1.0 < tx and tx < 1.0 and -1.0 < ty and ty < 1.0): continue
                
                extrema.append([-td,scale*(x+tx),scale*(y+ty),(k**((s+ts)-1))*sigma])                    

            for i in range(len(maxima[0])):
                s = maxima[0][i]+1
                x = maxima[1][i]+1
                y = maxima[2][i]+1
                
                # Get a 3 by 3 block
                block = ds[s-1:s+2,x-1:x+2,y-1:y+2]
                params = TaylorFit(block)
                ts,tx,ty = TaylorSubpixel(params)
                td = TaylorApprox(params,ts,tx,ty)
                ts -= 1.0
                tx -= 1.0
                ty -= 1.0
                                         
                # Only select extrema with high contrast
                if abs(td) < self.min_contrast: continue

                # Chech the ratios of the principal curvatures (see Lowe 2004 Sec 4.1:
                Dxx=2*params[1]
                Dyy=2*params[2]
                Dxy=params[3]
                TrH = Dxx+Dyy
                DetH = Dxx*Dyy-Dxy*Dxy
                
                r = self.max_curvature_ratio
                if DetH < 0: continue # discard because curvatures have different signs
                if r*TrH > DetH*(r+1)*(r+1): continue # Ratio of curvatuers is greater than R

                if not (-1.0 < tx and tx < 1.0 and -1.0 < ty and ty < 1.0): continue
                
                extrema.append([td,scale*(x+tx),scale*(y+ty),(k**((s+ts)-1))*sigma])                    
           
            sigma = (k**2.0)*sigma
            scale *= 2
        
        return extrema


def TaylorFit(block):
    '''
    a*s*s + b*x*x + c*y*y + d*x*y + e*s + f*x + g*y + h
    '''
    A = []
    b = []
    for s in range(3):
        for x in range(3):
            for y in range(3):
                z = block[s,x,y]
                row = [s*s,x*x,y*y,x*y,s,x,y,1]
                A.append(row)
                b.append([z])
    params,resids,rank,s = lstsq(A,b)
    return params.flatten()
    
def TaylorSubpixel(params):
    '''
    a*s*s + b*x*x + c*y*y + d*x*y + e*s + f*x + g*y + h
    '''
    a,b,c,d,e,f,g,h = params
    s = -e/(2*a)
    x = -f/(2*b)
    y = -g/(2*c)
    return s,x,y

def TaylorApprox(params,s,x,y):
    '''
    a*s*s + b*x*x + c*y*y + d*x*y + e*s + f*x + g*y + h
    '''
    a,b,c,d,e,f,g,h = params
    return a*s*s + b*x*x + c*y*y + d*x*y + e*s + f*x + g*y + h
    


class DetectorCornerTestCase(unittest.TestCase):
    def setUp(self):
        pass
    
        
    def testDetectorCorner1(self):
        detector = DetectorDOG(selector='best',n=100)
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_1.jpg')
        im = Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        print len(points)
        for score,pt,radius in points:
            im.annotateCircle(pt,radius)
        im.show()
        #self.assertEquals(len(points),0)

    def dtestDetectorCorner2(self):
        detector = DetectorDOG(selector='best')
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_19.jpg')
        im = Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for score,pt,radius in points:
            im.annotatePoint(pt)
            
        self.assertEquals(len(points),0)

    def dtestDetectorCorner3(self):
        detector = DetectorDOG()
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_22.jpg')
        im = Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for score,pt,radius in points:
            im.annotatePoint(pt)
            
        self.assertEquals(len(points),0)

    def dtestDetectorCorner4(self):
        detector = DetectorDOG()
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_37.jpg')
        im = Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for score,pt,radius in points:
            im.annotatePoint(pt)
            
        self.assertEquals(len(points),0)

    def dtestDetectorCorner5(self):
        detector = DetectorDOG(selector='best')
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_37.jpg')
        im = Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for score,pt,radius in points:
            im.annotatePoint(pt)
            
        self.assertEquals(len(points),0)
        
    def dtestDetectorCorner6(self):
        detector = DetectorDOG(selector='all')
        filename = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_37.jpg')
        im = Image(filename,bw_annotate=True)
        
        points = detector.detect(im)
        for score,pt,radius in points:
            im.annotatePoint(pt)
            
        self.assertEquals(len(points),0)
        

  



