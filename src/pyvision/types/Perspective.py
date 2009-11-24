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


##
# This class Performs a perspective transform on an image.  Primarily 
# implemented using OpenCV: cvWarpPerspective
##

import unittest
import os.path
import math

import opencv as cv
import numpy as np
import numpy.linalg as la
import pyvision as pv


def logPolar(im,center=None,radius=None,M=None,size=(64,128)):
    '''
    Produce a log polar transform of the image.  See OpenCV for details.
    The scale space is calculated based on radius or M.  If both are given 
    M takes priority.
    '''
    #M=1.0
    w,h = im.size
    if radius == None:
        radius = 0.5*min(w,h)
        
    if M == None:
        #rho=M*log(sqrt(x2+y2))
        #size[0] = M*log(r)
        M = size[0]/np.log(radius)

    if center == None:
        center = pv.Point(0.5*w,0.5*h)
    src = im.asOpenCV()
    dst = cv.cvCreateImage( cv.cvSize(size[0],size[1]), 8, 3 );
    cv.cvLogPolar( src, dst, center.asOpenCV(), M, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS )
    return pv.Image(dst)
    
##
# Create a transform or homography that optimally maps one set of points to 
# the other set of points.  Requires at least four points.
# A B C   x1  =  x2
# D E F   y1  =  y2
# G H 1   w1  =  w2
#
def PerspectiveFromPointsOld(source, dest, new_size):
    '''
    Python/Scipy implementation implementation which finds a perspective 
    transform between points.
    
    Most users should use PerspectiveFromPoints instead.  This method
    may be eliminated in the future.
    '''
    assert len(source) == len(dest)
    
    src_nrm = pv.AffineNormalizePoints(source)
    source = src_nrm.transformPoints(source)
    dst_nrm = pv.AffineNormalizePoints(dest)
    dest   = dst_nrm.transformPoints(dest)
    
    A = []
    for i in range(len(source)):
        src = source[i]
        dst = dest[i]
        
        # See Hartley and Zisserman Ch. 4.1, 4.1.1, 4.4.4
        row1 = [0.0,0.0,0.0,-dst.w*src.x,-dst.w*src.y,-dst.w*src.w,dst.y*src.x,dst.y*src.y,dst.y*src.w]
        row2 = [dst.w*src.x,dst.w*src.y,dst.w*src.w,0.0,0.0,0.0,-dst.x*src.x,-dst.x*src.y,-dst.x*src.w]
        #row3 = [-dst.y*src.x,-dst.y*src.y,-dst.y*src.w,dst.x*src.x,dst.x*src.y,dst.x*src.w,0.0,0.0,0.0]
        A.append(row1)
        A.append(row2)
        #A.append(row3)
    A = np.array(A)
    U,D,Vt = la.svd(A)
    H = Vt[8,:].reshape(3,3)

    matrix = np.dot(dst_nrm.inverse,np.dot(H,src_nrm.matrix))

    return PerspectiveTransform(matrix,new_size)


def PerspectiveFromPoints(source, dest, new_size, method=0, ransacReprojThreshold=2.0):
    '''
    Calls the OpenCV function: cvFindHomography.  This method has
    additional options to use the CV_RANSAC or CV_LMEDS methods to
    find a robust homography.  Method=0 appears to be similar to 
    PerspectiveFromPoints.  
    '''
    assert len(source) == len(dest)
    
    n_points = len(source)
    
    s = cv.cvCreateMat(n_points,2,cv.CV_32F)
    d = cv.cvCreateMat(n_points,2,cv.CV_32F)
    p = cv.cvCreateMat(3,3,cv.CV_32F)
    
    for i in range(n_points):
        s[i,0] = source[i].X()
        s[i,1] = source[i].Y()
    
        d[i,0] = dest[i].X()
        d[i,1] = dest[i].Y()
        
    results = cv.cvFindHomography(s,d,p)
    
    matrix = pv.OpenCVToNumpy(p)

    return PerspectiveTransform(matrix,new_size)



    

##
# The PerspectiveTransform class is used to transform images and points back and
# and forth between different coordinate systems.  
# 
#
#
# @param matrix a 3-by-3 matrix that defines the transformation.
# @param new_size the size of any new images created by this affine transform.
# @keyparam filter the image filtering function used for interpolating between pixels.
# @return an AffineTransform object
class PerspectiveTransform:

    def __init__(self,matrix,new_size,filter=None):
        self.matrix = matrix
        self.inverse = la.inv(matrix)
        self.size = new_size
        self.filter = filter
    
    ##
    # Transforms an image into the new coordinate system.
    #
    # @param im an pv.Image object
    def transformImage(self,im):
        ''' Transform an image. '''
        matrix = pv.NumpyToOpenCV(self.matrix)
        src = im.asOpenCV()
        dst = cv.cvCreateImage( cv.cvSize(self.size[0],self.size[1]), cv.IPL_DEPTH_8U, src.nChannels );
        cv.cvWarpPerspective( src, dst, matrix)                    
        return pv.Image(dst)

        
    ##
    # Transforms a link.Point into the new coordinate system.
    #
    # @param pt a link.Point
    def transformPoint(self,pt):
        ''' Transform a point from the old image to the new image '''
        vec = np.dot(self.matrix,pt.asVector2H())
        return pv.Point(x=vec[0,0],y=vec[1,0],w=vec[2,0])
        
    ##
    # Transforms a list of link.Points into the new coordinate system.
    #
    # @param pts a list of link.Points
    def transformPoints(self,pts):
        ''' Transform a point from the old image to the new image '''
        return [ self.transformPoint(pt) for pt in pts ]
        
    ##
    # Transforms a link.Point from the new coordinate system to
    # the old coordinate system.
    #
    # @param pts a list of link.Points
    def invertPoint(self,pt):
        ''' Transform a point from the old image to the new image '''
        vec = np.dot(self.inverse,pt.asVector2H())
        return pv.Point(x=vec[0,0],y=vec[1,0],w=vec[2,0])
        
    ##
    # Transforms a list of link.Points from the new coordinate system to
    # the old coordinate system.
    #
    # @param pts a list of link.Points
    def invertPoints(self,pts):
        ''' Transform a point from the old image to the new image '''
        return [ self.invertPoint(pt) for pt in pts ]
        
    ##
    # @return the transform as a 3 by 3 matrix
    def asMatrix(self):
        ''' Return the transform as a 3 by 3 matrix '''
        return self.matrix
    
    ##
    # Used to concatinate transforms.  For example:
    # <pre>
    # # This code first scales and then translates
    # S = AffineScale(2.0)
    # T = AffineTranslate(4,5)
    # A = T*S
    # new_im = A.transformImage(old_im)
    # </pre>
    # @return a single link.AffineTransform which is the the same as 
    #         both affine transforms.
    def __mul__(self,affine):
        return PerspectiveTransform(dot(self.matrix,affine.matrix),self.size,self.filter)

# TODO: Add unit tests
class _PerspectiveTest(unittest.TestCase):
    
    def setUp(self):
        fname_a = os.path.join(pv.__path__[0],'data','test','perspective1a.jpg')
        fname_b = os.path.join(pv.__path__[0],'data','test','perspective1b.jpg')
        
        self.im_a = pv.Image(fname_a)
        self.im_b = pv.Image(fname_b)
        
        #corners clockwize: upper left, upper right, lower right, lower left
        self.corners_a = (pv.Point(241,136),pv.Point(496,140),pv.Point(512,343),pv.Point(261,395))
        self.corners_b = (pv.Point(237,165),pv.Point(488,177),pv.Point(468,392),pv.Point(222,347))
        self.corners_t = (pv.Point(0,0),pv.Point(639,0),pv.Point(639,479),pv.Point(0,479))
        
        for pt in self.corners_a:
            self.im_a.annotatePoint(pt)

        #self.im_a.show()
        #self.im_b.show()
            
    def test_four_points_a(self):
        p = PerspectiveFromPoints(self.corners_a,self.corners_t,(640,480))
        pts = p.transformPoints(self.corners_a)
        #for pt in pts:
        #    print "Point: %7.2f %7.2f"%(pt.X(), pt.Y())
            
        im = p.transformImage(self.im_a)
        #im.show()

    def test_four_points_b(self):
        p = PerspectiveFromPoints(self.corners_b,self.corners_t,(640,480))
        pts = p.transformPoints(self.corners_b)
        #for pt in pts:
        #    print "Point: %7.2f %7.2f"%(pt.X(), pt.Y())
            
        im = p.transformImage(self.im_b)
        #im.show()
        
    def test_four_points_ab(self):
        p = PerspectiveFromPoints(self.corners_a,self.corners_b,(640,480))
        #pts = p.transformPoints(self.corners_b)
        #for pt in pts:
        #    print "Point: %7.2f %7.2f"%(pt.X(), pt.Y())
            
        im = p.transformImage(self.im_a)
        #im.show()
        #self.im_b.show()
        
        
        
        
        
