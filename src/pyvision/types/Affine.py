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
This module contains the AffineTransform class and a set of factory 
functions used to create AffineTransform instances given different 
sets of parameters.  Most factory functions require information that 
specifies the transformation and a size for the output image.
'''



import unittest
import os.path
import math
import copy
import weakref

try:
    from PIL.Image import AFFINE,NEAREST,BILINEAR,BICUBIC
except:
    from Image import AFFINE,NEAREST,BILINEAR,BICUBIC
    
from numpy import array,dot,sqrt
from numpy.linalg import inv,solve,lstsq
from scipy.ndimage import affine_transform
import random

import pyvision
import pyvision as pv
import numpy as np
try:
    import opencv as cv
except:
    import cv
    
from pyvision.types.img import Image, TYPE_PIL, TYPE_MATRIX_2D, TYPE_OPENCV
from pyvision.types.Point import Point
from pyvision.types.Rect import Rect
from pyvision.vector.RANSAC import RANSAC,LMeDs


def AffineNormalizePoints(points):
    '''
    Create a transform that centers a set of points such that there mean is (0,0)
    and then scale such that there average distance from (0,0) is 1.0
     
    @param points: list of link.Point to normalize
    @returns: an AffineTransform object
    '''
    # compute the center
    mean = Point(0,0)
    count = 0
    for point in points:
        mean += point
        count += 1
    mean = (1.0/count)*mean
    
    # mean center the points
    center = AffineTranslate(-mean.X(),-mean.Y(),(0,0))
    points = center.transformPoints(points)
    
    # Compute the mean distance
    mean_dist = 0.0
    count = 0
    for point in points:
        x,y = point.X(),point.Y()
        dist = sqrt(x*x+y*y)
        mean_dist += dist
        count += 1
    mean_dist = (1.0/count)*mean_dist
    
    # Rescale the points
    scale = AffineScale(1.0/mean_dist,(0,0))
    points = scale.transformPoints(points)
    
    # compute the composite transform
    norm = scale*center

    return norm

    

def AffineTranslate(dx,dy,new_size,filter=BILINEAR):
    '''
    Create a simple translation transform
    
    @param dx: translation in the x direction
    @param dy: translation in the y direction
    @param new_size: new size for the image
    @param filter: PIL filter to use    
    '''
    matrix = array([[1,0,dx],[0,1,dy],[0,0,1]],'d')

    return AffineTransform(matrix,new_size,filter)
    

def AffineScale(scale,new_size,center=None,filter=BILINEAR):
    '''
    Create a simple scale transform.

    @param scale: the amount to scale the image.
    @param new_size: new size for the image.
    @param filter: PIL filter to use.
    '''
    matrix = array([[scale,0,0],[0,scale,0],[0,0,1]],'d')

    scale = AffineTransform(matrix,new_size,filter)
    if center == None:
        return scale
    else:
        return AffineTranslate(center.X(),center.Y(),new_size)*scale*AffineTranslate(-center.X(),-center.Y(),new_size)
    

def AffineNonUniformScale(sx,sy,new_size,filter=BILINEAR):
    '''
    Create a scale transform with different values for the x and y directions.

    @param sx: scale in the x direction.
    @param sy: scale in the y direction.
    @param new_size: new size for the image.
    @param filter: PIL filter to use.
    '''
    matrix = array([[sx,0,0],[0,sy,0],[0,0,1]],'d')

    return AffineTransform(matrix,new_size,filter)
    

def AffineRotate(theta,new_size,center=None,filter=BILINEAR):
    '''
    Create a rotation about the origin.
    
    @param theta: the angle to rotate the image in radians.
    @param new_size: new size for the image.
    @param filter: PIL filter to use.
    '''
    matrix = array([[math.cos(theta),-math.sin(theta),0],[math.sin(theta),math.cos(theta),0],[0,0,1]],'d')

    rotate = AffineTransform(matrix,new_size,filter)
    if center == None:
        return rotate
    else:
        return AffineTranslate(center.X(),center.Y(),new_size)*rotate*AffineTranslate(-center.X(),-center.Y(),new_size)
    
def AffineFromRect(rect,new_size,filter=BILINEAR):
    ''' 
    Create a transform from a source rectangle to a new image.  This basically 
    crops a rectangle out of the image and rescales it to the new size.
    
    @param rect: the source link.Rect.
    @param new_size: new size for the image.
    @param filter: PIL filter to use.
    '''
    w,h = new_size
    
    x_scale = float(w)/rect.w
    y_scale = float(h)/rect.h
    x_trans = -rect.x*x_scale
    y_trans = -rect.y*y_scale
    matrix = array([[x_scale,0,x_trans],[0,y_scale,y_trans],[0,0,1]],'d')

    return AffineTransform(matrix,new_size,filter)
    

def AffineFromTile(center,new_size,filter=BILINEAR):
    '''
    Extract an image tile centered on a point.
    
    @param center: the center link.Point of the tile.
    @param new_size: new size for the image.
    @param filter: PIL filter to use.
    '''
    w,h = new_size
    rect = Rect(center.X()-w/2,center.Y()-h/2,w,h)
    
    x_scale = float(w)/rect.w
    y_scale = float(h)/rect.h
    x_trans = -rect.x*x_scale
    y_trans = -rect.y*y_scale
    matrix = array([[x_scale,0,x_trans],[0,y_scale,y_trans],[0,0,1]],'d')

    return AffineTransform(matrix,new_size,filter)
    

def AffineFromPoints(src1,src2,dst1,dst2,new_size,filter=BILINEAR):
    ''' 
    An affine transform that will rotate, translate, and scale to map one 
    set of points to the other. For example, to align eye coordinates in face images.
     
    Find a transform (a,b,tx,ty) such that it maps the source points to the 
    destination points::
     
        a*x1-b*y1+tx = x2
        b*x1+a*y1+ty = y2
     
    The mapping between the two points creates a set of  four linear equations 
    with four unknowns. This set of equations is solved to find the transform.
    
    @param src1: the first link.Point in the source image.
    @param src2: the second link.Point in the source image.
    @param dst1: the first link.Point in the destination image.
    @param dst2: the second link.Point in the destination image.
    @param new_size: new size for the image.
    @param filter: PIL filter to use.
    '''
    
    # Compute the transformation parameters
    A = [[src1.X(),-src1.Y(),1,0],
         [src1.Y(),src1.X(),0,1],
         [src2.X(),-src2.Y(),1,0],
         [src2.Y(),src2.X(),0,1]]
    b = [dst1.X(),dst1.Y(),dst2.X(),dst2.Y()]
    A = array(A)
    b = array(b)
    result = solve(A,b)
    
    a,b,tx,ty = result    
    # Create the transform matrix
    matrix = array([[a,-b,tx],[b,a,ty],[0,0,1]],'d')
    
    return AffineTransform(matrix,new_size,filter)


def AffineFromPointsLS(src,dst,new_size,filter=BILINEAR, normalize=True):  
    '''
     An affine transform that will rotate, translate, and scale to map one 
     set of points to the other. For example, to align eye coordinates in face images.
     
     Find a transform (a,b,tx,ty) such that it maps the source points to the 
     destination points::
     
         a*x1-b*y1+tx = x2
         b*x1+a*y1+ty = y2
     
     This method minimizes the squared error to find an optimal fit between the 
     points.
    
     @param src: a list of link.Points in the source image.
     @param dst: a list of link.Points in the destination image.
     @param new_size: new size for the image.
     @param filter: PIL filter to use.
    '''  
    if normalize:
        # Normalize Points
        src_norm = AffineNormalizePoints(src)
        src = src_norm.transformPoints(src)
        dst_norm = AffineNormalizePoints(dst)
        dst = dst_norm.transformPoints(dst)
    
    # Compute the transformation parameters
    A = []
    b = []
    for i in range(len(src)):
        A.append([src[i].X(),-src[i].Y(),1,0])
        A.append([src[i].Y(), src[i].X(),0,1])
        b.append(dst[i].X())
        b.append(dst[i].Y())
         
    A = array(A)
    b = array(b)
        
    result,resids,rank,s = lstsq(A,b)
    
    a,b,tx,ty = result    
    # Create the transform matrix
    matrix = array([[a,-b,tx],[b,a,ty],[0,0,1]],'d')
    
    if normalize:
        matrix = dot(dst_norm.inverse,dot(matrix,src_norm.matrix))

    return AffineTransform(matrix,new_size,filter)


def AffineFromPointsRANSAC(src,dst,new_size,filter=BILINEAR, normalize=True,tol=0.15):
    '''
    An affine transform that will rotate, translate, and scale to map one 
    set of points to the other. For example, to align eye coordinates in face images.
     
    Find a transform (a,b,tx,ty) such that it maps the source points to the 
    destination points::
        
        a*x1-b*y1+tx = x2
        b*x1+a*y1+ty = y2
     
    This method minimizes the squared error to find an optimal fit between the 
    points.  Instead of a LS solver the RANSAC solver is used to
    produce a transformation that is robust to outliers.
    
    @param src: a list of link.Points in the source image.
    @param dst: a list of link.Points in the destination image.
    @param new_size: new size for the image.
    @param filter: PIL filter to use.
    '''
    if normalize:
        # Normalize Points
        src_norm = AffineNormalizePoints(src)
        src = src_norm.transformPoints(src)
        dst_norm = AffineNormalizePoints(dst)
        dst = dst_norm.transformPoints(dst)
    
    # Compute the transformation parameters
    A = []
    b = []
    for i in range(len(src)):
        A.append([src[i].X(),-src[i].Y(),1,0])
        A.append([src[i].Y(), src[i].X(),0,1])
        b.append(dst[i].X())
        b.append(dst[i].Y())
         
    A = array(A)
    b = array(b)
        
    result = RANSAC(A,b,tol=tol,group=2)
    
    #print result,resids,rank,s 
    
    a,b,tx,ty = result    
    # Create the transform matrix
    matrix = array([[a,-b,tx],[b,a,ty],[0,0,1]],'d')
    
    if normalize:
        matrix = dot(dst_norm.inverse,dot(matrix,src_norm.matrix))

    return AffineTransform(matrix,new_size,filter)


def AffineFromPointsLMeDs(src,dst,new_size,filter=BILINEAR, normalize=True):
    '''
    An affine transform that will rotate, translate, and scale to map one 
    set of points to the other. For example, to align eye coordinates in face images.
     
    Find a transform (a,b,tx,ty) such that it maps the source points to the 
    destination points::
        
        a*x1-b*y1+tx = x2
        b*x1+a*y1+ty = y2
     
    This method minimizes the squared error to find an optimal fit between the 
    points.  Instead of a LS solver the RANSAC solver is used to
    produce a transformation that is robust to outliers.
    
    @param src: a list of link.Points in the source image.
    @param dst: a list of link.Points in the destination image.
    @param new_size: new size for the image.
    @param filter: PIL filter to use.
    '''
    if normalize:
        # Normalize Points
        src_norm = AffineNormalizePoints(src)
        src = src_norm.transformPoints(src)
        dst_norm = AffineNormalizePoints(dst)
        dst = dst_norm.transformPoints(dst)
    
    # Compute the transformation parameters
    A = []
    b = []
    for i in range(len(src)):
        A.append([src[i].X(),-src[i].Y(),1,0])
        A.append([src[i].Y(), src[i].X(),0,1])
        b.append(dst[i].X())
        b.append(dst[i].Y())
         
    A = array(A)
    b = array(b)
        
    result = LMeDs(A,b)
    
    #print result,resids,rank,s 
    
    a,b,tx,ty = result    
    # Create the transform matrix
    matrix = array([[a,-b,tx],[b,a,ty],[0,0,1]],'d')
    
    if normalize:
        matrix = dot(dst_norm.inverse,dot(matrix,src_norm.matrix))

    return AffineTransform(matrix,new_size,filter)


def AffinePerturb(Dscale, Drotate, Dtranslate, new_size, mirror=False, flip=False, rng = None):
    '''
    Generates an link.AffineTrasform that slightly perturbs the image.  Primarily 
    to generate more training images. 
    
    The perturbations include small scale, rotation, and translations.  The 
    transform can also mirror the image in the left/right direction or flip the
    top and bottom as other ways to generate synthetic training images.

    @param src: a list of link.Points in the source image.
    @param dst: a list of link.Points in the destination image.
    @param new_size: new size for the image.
    @param filter: PIL filter to use.
    '''
    tile_size = new_size
    w,h = tile_size
    if rng == None:
        rng = random
    
    tx = rng.uniform(-Dtranslate,Dtranslate)
    ty = rng.uniform(-Dtranslate,Dtranslate)
    if mirror:
        sx = rng.choice([-1.,1.])
    else:
        sx = 1.0
    if flip:
        sy = rng.choice([-1.,1.])
    else:
        sy = 1.0
    s  = rng.uniform(1-Dscale,1+Dscale)
    r  = rng.uniform(-Drotate,Drotate)
    
    there = AffineTranslate(-w/2,-h/2,tile_size)
    flipflop = AffineNonUniformScale(sx,sy,tile_size)
    scale = AffineScale(s,tile_size)
    rotate = AffineRotate(r,tile_size)
    translate = AffineTranslate(tx,ty,tile_size)
    back = AffineTranslate(w/2,h/2,tile_size)
    affine = back*translate*rotate*scale*flipflop*there
    
    return affine


class AffineTransform:
    '''
    The AffineTransform class is used to transform images and points back and
    and forth between different coordinate systems. 
    '''

    def __init__(self,matrix,new_size,filter=BILINEAR):
        '''
        Constructor for the AffineTransform.  See also the affine transform factories.
        
        @param matrix: a 3-by-3 matrix that defines the transformation.
        @param new_size: the size of any new images created by this affine transform.
        @param filter: the image filtering function used for interpolating between pixels.
        @returns: an AffineTransform object
        '''
        self.matrix = matrix
        self.inverse = inv(matrix)
        self.size = int(new_size[0]),int(new_size[1])
        self.filter = filter
        
    def __call__(self,data):
        '''
        This is a simple interface to transform images or points.  Simply
        call the affine transform like a function and it will try to automatically 
        transform the argument.
        
        @param data: an image, point, or list of points.
        '''
        if isinstance(data,pv.Image):
            return self.transformImage(data)
        elif isinstance(data,pv.Point):
            return self.transformPoint(data)
        else: # assume this is a list of points
            return self.transformPoints(data)
    
    def invert(self,data):
        '''
        This is a simple interface to transform images or points.  Simply
        call invert with the points or list of points and it will automatically
        call the correct function.
        
        @param data: an image, point, or list of points.
        '''
        if isinstance(data,pv.Image):
            return self.invertImage(data)
        elif isinstance(data,pv.Point):
            return self.invertPoint(data)
        else: # assume this is a list of points
            return self.invertPoints(data)
    
    def invertImage(self,im, use_orig=True):
        '''
        Perform the inverse affine transformation on the image.
        '''
        return self.transformImage(im,use_orig=use_orig,inverse=True)

    def transformImage(self,im, use_orig=True, inverse=False):
        ''' 
        Transforms an image into the new coordinate system.
        
        If this image was produced via an affine transform of another image, 
        this method will attempt to trace weak references to the original image 
        and directly compute the new image from that image to improve accuracy.
        To accomplish this a weak reference to the original source image and
        the affine matrix used for the transform are added to any image 
        produced by this method.  This can be disabled using the use_orig 
        parameter.
        
        
        @param im: an Image object
        @param use_original: (True or False) attempts to find and use the original image as the source to avoid an accumulation of errors.
        @returns: the transformed image
        '''
        #TODO: does not support opencv images.  see Perspective.py
        prev_im = im
        
        if inverse:
            inverse = self.matrix
        else:
            inverse = self.inverse
        
        if use_orig:
            # Find the oldest image used to produce this one by following week 
            # references.

            # Check to see if there is an aff_prev list
            if hasattr(prev_im,'aff_prev'):
            
                # If there is... search that list for the oldest image
                found_prev = False
                for i in range(len(prev_im.aff_prev)):
                    ref,cmat = prev_im.aff_prev[i]
                    if not found_prev and ref():
                        im = ref()
                        mat = np.eye(3)
                        found_prev = True
                        
                    if found_prev:
                        mat = np.dot(mat,cmat)
               
                if found_prev:
                    inverse = np.dot(mat,inverse) 
            
        if im.getType() == TYPE_PIL:
            data = inverse[:2,:].flatten()
            #data = (matrix[0,0],matrix[0,1],matrix[0,2],matrix[1,0],matrix[1,1],matrix[1,2])
            pil = im.asPIL().transform(self.size, AFFINE, data, self.filter)
            result = Image(pil)
        elif im.getType() == TYPE_MATRIX_2D:
            mat = im.asMatrix2D()

            mat = affine_transform(mat, self.inverse[:2,:2], offset=self.inverse[:2,2])
            result = Image(mat)
        elif im.getType() == TYPE_OPENCV:
            matrix = pv.NumpyToOpenCV(self.matrix)
            src = im.asOpenCV()
            dst = cv.CreateImage( (self.size[0],self.size[1]), cv.IPL_DEPTH_8U, src.nChannels );
            cv.WarpPerspective( src, dst, matrix, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS,cv.ScalarAll(128))                    
            result = pv.Image(dst)

        else:
            raise NotImplementedError("Unhandled image type for affine transform.")

        
        # Check to see if there is an aff_prev list for this object
        if use_orig and hasattr(prev_im,'aff_prev'):
            # Create one if not
            result.aff_prev = copy.copy(prev_im.aff_prev)
        else:
            result.aff_prev = []
            
        # Append the prev image and new transform
        result.aff_prev.append( (weakref.ref(prev_im), self.inverse) )
        
        return result

    
    def transformPoint(self,pt):
        ''' 
        Transform a point from the old image to the new image.
        
        @param pt: the point
        @returns: the new point
        '''
        vec = dot(self.matrix,pt.asVector2H())
        return Point(x=vec[0,0],y=vec[1,0],w=vec[2,0])
        

    def transformPoints(self,pts):
        ''' 
        Transform a set of points from the old image to the new image.
        
        @param pts: a list of points.
        @returns: a list of transformed points.
        '''
        return [ self.transformPoint(pt) for pt in pts ]
        
    
    def invertPoint(self,pt):
        '''
        Transforms a Point from the new coordinate system to
        the old coordinate system.
        
        @param pt: a single point
        @returns: the transformed point
        '''
        vec = dot(self.inverse,pt.asVector2H())
        return Point(x=vec[0,0],y=vec[1,0],w=vec[2,0])
        
    
    def invertPoints(self,pts):
        '''
        Transforms a list of oints from the new coordinate system to
        the old coordinate system.
        
        @param pts: a list of Points
        @returns: the transformed Points
        '''
        return [ self.invertPoint(pt) for pt in pts ]
        
    
    def asMatrix(self):
        ''' 
        @returns: the transform as a 3 by 3 matrix 
        '''
        return self.matrix
    
    
    def __mul__(self,affine):
        '''
        Used to concatenate transforms.  For example::

            # This code first scales and then translates
            S = AffineScale(2.0)
            T = AffineTranslate(4,5)
            A = T*S
            new_im = A.transformImage(old_im)

        @returns: a single AffineTransform which is the the same as both affine transforms.
        '''
        return AffineTransform(dot(self.matrix,affine.matrix),self.size,self.filter)


# TODO: Add unit tests
class _AffineTest(unittest.TestCase):
    
    def setUp(self):
        fname = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_13.jpg')
        self.test_image = Image(fname)
        #self.test_image.show()
    
    def test_rotation(self):
        transform = AffineRotate(3.14/8,(640,480))
        im = transform.transformImage(self.test_image)
        # im.show()
        
        pt = transform.transformPoint(Point(320,240))
        self.assertAlmostEqual(pt.X(),203.86594448424472)
        self.assertAlmostEqual(pt.Y(),344.14920700118842)

        pt = transform.invertPoint(Point(320,240))
        self.assertAlmostEqual(pt.X(),387.46570317672939)
        self.assertAlmostEqual(pt.Y(),99.349528744542198)
        
    def test_scale(self):
        transform = AffineScale(1.5,(640,480))
        im = transform.transformImage(self.test_image)
        #im.show()
        
        pt = transform.transformPoint(Point(320,240))
        self.assertAlmostEqual(pt.X(),480.)
        self.assertAlmostEqual(pt.Y(),360.)

        pt = transform.invertPoint(Point(320,240))
        self.assertAlmostEqual(pt.X(),213.33333333333331)
        self.assertAlmostEqual(pt.Y(),160.)
        
    def test_translate(self):
        transform = AffineTranslate(10.,15.,(640,480))
        im = transform.transformImage(self.test_image)
        #im.show()
        
        pt = transform.transformPoint(Point(320,240))
        self.assertAlmostEqual(pt.X(),330.)
        self.assertAlmostEqual(pt.Y(),255.)

        pt = transform.invertPoint(Point(320,240))
        self.assertAlmostEqual(pt.X(),310.)
        self.assertAlmostEqual(pt.Y(),225.)
        
    def test_from_rect(self):
                
        transform = AffineFromRect(Rect(100,100,300,300),(100,100))
        im = transform.transformImage(self.test_image)
        #im.show()
        
        pt = transform.transformPoint(Point(320,240))
        self.assertAlmostEqual(pt.X(),73.333333333333329)
        self.assertAlmostEqual(pt.Y(),46.666666666666671)

        pt = transform.invertPoint(Point(50.,50.))
        self.assertAlmostEqual(pt.X(),250.0)
        self.assertAlmostEqual(pt.Y(),250.0)
        
    def test_from_points(self):
        # TODO: Fix this test
        pass
        
    def test_sim_least_sqr(self):
        # TODO: Fix this test
        pass
        
    def test_affine_least_sqr(self):
        # TODO: Fix this test
        pass

    def test_affine_mul(self):
        # TODO: FIx this test
        pass
        
    def test_affine_numpy(self):
        # TODO: FIx this  test
        pass
        
    def test_affine_opencv(self):
        # TODO: FIx this test
        pass
        
    def test_prev_ref1(self):
        fname = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_13.jpg')
        im = Image(fname)
        ref  = weakref.ref(im)

        self.assertEquals(ref(), im)
        
        tmp = im
        del im
        
        self.assertEquals(ref(), tmp)
        
        del tmp
        
        self.assertEquals(ref(), None)
        
 
    def test_prev_ref2(self):
        fname = os.path.join(pyvision.__path__[0],'data','nonface','NONFACE_13.jpg')
        im = Image(fname)
        #im.show()
        w,h = im.size
        
        # Try scaling down and then scaling back up
        tmp1 = AffineScale(0.1,(w/10,h/10)).transformImage(im)
        #tmp1.show()
        
        tmp2 = AffineScale(10.0,(w,h)).transformImage(tmp1,use_orig=False)
        tmp2.annotateLabel(pv.Point(10,10), "This image should be blurry.")
        tmp2.show()
       
        tmp3 = AffineScale(10.0,(w,h)).transformImage(tmp1,use_orig=True)
        tmp3.annotateLabel(pv.Point(10,10), "This image should be sharp.")
        tmp3.show()
        
        del im
        
        tmp4 = AffineScale(10.0,(w,h)).transformImage(tmp1,use_orig=True)
        tmp4.annotateLabel(pv.Point(10,10), "This image should be blurry.")
        tmp4.show()
        
    def test_prev_ref2(self):
        fname = os.path.join(pv.__path__[0],'data','nonface','NONFACE_13.jpg')
        torig = tprev = taccu = im = Image(fname)
        #im.show()
        w,h = im.size
        
        # Scale
        aff = AffineScale(0.5,(w/2,h/2))
        accu = aff
        torig = aff.transformImage(torig)
        tprev = aff.transformImage(tprev,use_orig=False)
        taccu = accu.transformImage(im)
        
        torig.annotateLabel(pv.Point(10,10), "use_orig = True")
        tprev.annotateLabel(pv.Point(10,10), "use_orig = False")
        taccu.annotateLabel(pv.Point(10,10), "accumulated")
        
        #torig.show()
        #tprev.show()
        #taccu.show()
        
        # Translate
        aff = AffineTranslate(20,20,(w/2,h/2))
        accu = aff*accu
        torig = aff.transformImage(torig)
        tprev = aff.transformImage(tprev,use_orig=False)
        taccu = accu.transformImage(im)
        
        torig.annotateLabel(pv.Point(10,10), "use_orig = True")
        tprev.annotateLabel(pv.Point(10,10), "use_orig = False")
        taccu.annotateLabel(pv.Point(10,10), "accumulated")
        
        #torig.show()
        #tprev.show()
        #taccu.show()
        
        
        # Rotate
        aff = AffineRotate(np.pi/4,(w/2,h/2))
        accu = aff*accu
        torig = aff.transformImage(torig)
        tprev = aff.transformImage(tprev,use_orig=False)
        taccu = accu.transformImage(im)
        
        torig.annotateLabel(pv.Point(10,10), "use_orig = True")
        tprev.annotateLabel(pv.Point(10,10), "use_orig = False")
        taccu.annotateLabel(pv.Point(10,10), "accumulated")
        
        #torig.show()
        #tprev.show()
        #taccu.show()
        
        
        
        # Translate
        aff = AffineTranslate(100,-10,(w/2,h/2))
        accu = aff*accu
        torig = aff.transformImage(torig)
        tprev = aff.transformImage(tprev,use_orig=False)
        taccu = accu.transformImage(im)
        
        torig.annotateLabel(pv.Point(10,10), "use_orig = True")
        tprev.annotateLabel(pv.Point(10,10), "use_orig = False")
        taccu.annotateLabel(pv.Point(10,10), "accumulated")
        
        #torig.show()
        #tprev.show()
        #taccu.show()
        
        
        # Scale
        aff = AffineScale(2.0,(w,h))
        accu = aff*accu
        torig = aff.transformImage(torig)
        tprev = aff.transformImage(tprev,use_orig=False)
        taccu = accu.transformImage(im)
        
        torig.annotateLabel(pv.Point(10,10), "use_orig = True")
        tprev.annotateLabel(pv.Point(10,10), "use_orig = False")
        taccu.annotateLabel(pv.Point(10,10), "accumulated")
        
        #torig.show()
        #tprev.show()
        #taccu.show()
        
        
        
        
        
        
        
    