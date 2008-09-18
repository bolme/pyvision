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

import unittest
import os.path
import math

from PIL.Image import AFFINE,NEAREST,BILINEAR,BICUBIC
from numpy import array,dot,sqrt
from numpy.linalg import inv,solve,lstsq
from scipy.ndimage import affine_transform
import random

import pyvision
from pyvision.types.Image import Image, TYPE_PIL, TYPE_MATRIX_2D
from pyvision.types.Point import Point
from pyvision.types.Rect import Rect
from pyvision.vector.RANSAC import RANSAC

##
# This module contains the link.AffineTransform class and a set of factory 
# functions used to create link.AffineTransform instances given different 
# sets of parameters.  Most factory functions require information that 
# specifies the transformation and a size for the output image.
##

##
# Create a transform that centers a set of points such that there mean is (0,0)
# and then scale such that there average distance from (0,0) is 1.0
# 
# @param points list of link.Point to normalize
# @return an link.AffineTransform object
def AffineNormalizePoints(points):
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

    
##
# Create a simple translation transform
#
# @param dx translation in the x direction
# @param dy translation in the y direction
# @param new_size new size for the image
# @keyparam filter PIL filter to use
def AffineTranslate(dx,dy,new_size,filter=BILINEAR):
    matrix = array([[1,0,dx],[0,1,dy],[0,0,1]],'d')

    return AffineTransform(matrix,new_size,filter)
    
##
# Create a simple scale transform.
#
# @param scale the amount to scale the image.
# @param new_size new size for the image.
# @keyparam filter PIL filter to use.
def AffineScale(scale,new_size,filter=BILINEAR):
    matrix = array([[scale,0,0],[0,scale,0],[0,0,1]],'d')

    return AffineTransform(matrix,new_size,filter)
    
##
# Create a scale transform with different values for the x and y directions.
#
# @param sx scale in the x direction.
# @param sy scale in the y direction.
# @param new_size new size for the image.
# @keyparam filter PIL filter to use.
def AffineNonUniformScale(sx,sy,new_size,filter=BILINEAR):
    matrix = array([[sx,0,0],[0,sy,0],[0,0,1]],'d')

    return AffineTransform(matrix,new_size,filter)
    
##
# Create a rotation about the origin.
#
# @param theta the angle to rotate the image in radians.
# @param new_size new size for the image.
# @keyparam filter PIL filter to use.
def AffineRotate(theta,new_size,filter=BILINEAR):
    matrix = array([[math.cos(theta),-math.sin(theta),0],[math.sin(theta),math.cos(theta),0],[0,0,1]],'d')

    return AffineTransform(matrix,new_size,filter)
    
##
# Create a transform from a source rectangle to a new image.  This basically 
# crops a rectangle out of the image and rescales it to the new size.
#
# @param rect the source link.Rect.
# @param new_size new size for the image.
# @keyparam filter PIL filter to use.
def AffineFromRect(rect,new_size,filter=BILINEAR):
    ''' An affine transform that crops out a rectangle and returns it as an image. '''
    w,h = new_size
    
    x_scale = float(w)/rect.w
    y_scale = float(h)/rect.h
    x_trans = -rect.x*x_scale
    y_trans = -rect.y*y_scale
    matrix = array([[x_scale,0,x_trans],[0,y_scale,y_trans],[0,0,1]],'d')

    return AffineTransform(matrix,new_size,filter)
    
##
# Extract an image tile centered on a point.
#
# @param center the center link.Point of the tile.
# @param new_size new size for the image.
# @keyparam filter PIL filter to use.
def AffineFromTile(center,new_size,filter=BILINEAR):
    w,h = new_size
    rect = Rect(center.X()-w/2,center.Y()-h/2,w,h)
    
    x_scale = float(w)/rect.w
    y_scale = float(h)/rect.h
    x_trans = -rect.x*x_scale
    y_trans = -rect.y*y_scale
    matrix = array([[x_scale,0,x_trans],[0,y_scale,y_trans],[0,0,1]],'d')

    return AffineTransform(matrix,new_size,filter)
    
##
# An affine transform that will rotate, translate, and scale to map one 
# set of points to the other. For example, to align eye coordinates in face images.
# <p>
# Find a transform (a,b,tx,ty) such that it maps the source points to the 
# destination points:<br>
# a*x1-b*y1+tx = x2<br>
# b*x1+a*y1+ty = y2
# <p>
# The mapping between the two points creates a set of  four linear equations 
# with four unknowns. This set of equations is solved to find the transform.
#
# @param src1 the first link.Point in the source image.
# @param src2 the second link.Point in the source image.
# @param dst1 the first link.Point in the destination image.
# @param dst2 the second link.Point in the destination image.
# @param new_size new size for the image.
# @keyparam filter PIL filter to use.
def AffineFromPoints(src1,src2,dst1,dst2,new_size,filter=BILINEAR):
    ''' 
    An affine transform that will rotate, translate, and scale to map one 
    set of points to the other. For example, to align eye coordinates in face images.
    
    Find a transform (a,b,tx,ty) such that it maps the source points to the 
    destination points:
    a*x1-b*y1+tx = x2
    b*x1+a*y1+ty = y2
    
    The mapping between the two points creates a set of  four linear equations 
    with four unknowns.
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


##
# An affine transform that will rotate, translate, and scale to map one 
# set of points to the other. For example, to align eye coordinates in face images.
# <p>
# Find a transform (a,b,tx,ty) such that it maps the source points to the 
# destination points:<br>
# a*x1-b*y1+tx = x2<br>
# b*x1+a*y1+ty = y2
# <p>
# This method minimizes the squared error to find an optimal fit between the 
# points.
#
# @param src a list of link.Points in the source image.
# @param dst a list of link.Points in the destination image.
# @param new_size new size for the image.
# @keyparam filter PIL filter to use.
def AffineFromPointsLS(src,dst,new_size,filter=BILINEAR, normalize=True):    
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
    
    #print result,resids,rank,s 
    
    a,b,tx,ty = result    
    # Create the transform matrix
    matrix = array([[a,-b,tx],[b,a,ty],[0,0,1]],'d')
    
    if normalize:
        matrix = dot(dst_norm.inverse,dot(matrix,src_norm.matrix))

    return AffineTransform(matrix,new_size,filter)


##
# An affine transform that will rotate, translate, and scale to map one 
# set of points to the other. For example, to align eye coordinates in face images.
# <p>
# Find a transform (a,b,tx,ty) such that it maps the source points to the 
# destination points:<br>
# a*x1-b*y1+tx = x2<br>
# b*x1+a*y1+ty = y2
# <p>
# This method minimizes the squared error to find an optimal fit between the 
# points.  Instead of a LS solver the <a href="pythondoc-RANSAC.html">RANSAC</a> solver is used to
# produce a transformation that is robust to outliers.
#
# @param src a list of link.Points in the source image.
# @param dst a list of link.Points in the destination image.
# @param new_size new size for the image.
# @keyparam filter PIL filter to use.
def AffineFromPointsRANSAC(src,dst,new_size,filter=BILINEAR, normalize=True,tol=1.0):
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
        
    
    result = RANSAC(A,b,tol=tol)
    
    #print result,resids,rank,s 
    
    a,b,tx,ty = result    
    # Create the transform matrix
    matrix = array([[a,-b,tx],[b,a,ty],[0,0,1]],'d')
    
    if normalize:
        matrix = dot(dst_norm.inverse,dot(matrix,src_norm.matrix))

    return AffineTransform(matrix,new_size,filter)


##
# Generates an link.AffineTrasform that slightly perturbs the image.  Primarily 
# to generate more training images. 
# The perturbations include small scale, rotation, and translations.  The 
# transform can also mirror the image in the left/right direction or flip the
# top and bottom as other ways to generate synthetic training images.
#
# @param src a list of link.Points in the source image.
# @param dst a list of link.Points in the destination image.
# @param new_size new size for the image.
# @keyparam filter PIL filter to use.
def AffinePerturb(Dscale, Drotate, Dtranslate, new_size, mirror=False, flip=False, rng = None):
    '''
    Randomly and slite
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


##
# The AffineTransform class is used to transform images and points back and
# and forth between different coordinate systems.  
# 
# @param matrix a 3-by-3 matrix that defines the transformation.
# @param new_size the size of any new images created by this affine transform.
# @keyparam filter the image filtering function used for interpolating between pixels.
# @return an AffineTransform object
class AffineTransform:

    def __init__(self,matrix,new_size,filter=BILINEAR):
        self.matrix = matrix
        self.inverse = inv(matrix)
        self.size = new_size
        self.filter = filter
    
    ##
    # Transforms an image into the new coordinate system.
    #
    # @param im an pv.Image object
    def transformImage(self,im):
        ''' Transform an image. '''
        matrix = self.inverse
        if im.getType() == TYPE_PIL:
            data = (matrix[0,0],matrix[0,1],matrix[0,2],matrix[1,0],matrix[1,1],matrix[1,2])
            pil = im.asPIL().transform(self.size, AFFINE, data, self.filter)
            return Image(pil)
        if im.getType() == TYPE_MATRIX_2D:
            # TODO: This does not seem to handle translations.
            mat = im.asMatrix2D()
            affine = array([[matrix[0,0],matrix[0,1]],[matrix[1,0],matrix[1,1]]])
            offset = (matrix[0,2],matrix[1,2])
            mat = affine_transform(mat, affine, output_shape=self.size)
            return Image(mat)
        else:
            raise NotImplementedError("Unhandled image type for affine transform.")
        
    ##
    # Transforms a link.Point into the new coordinate system.
    #
    # @param pt a link.Point
    def transformPoint(self,pt):
        ''' Transform a point from the old image to the new image '''
        vec = dot(self.matrix,pt.asVector2H())
        return Point(x=vec[0,0],y=vec[1,0],w=vec[2,0])
        
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
        vec = dot(self.inverse,pt.asVector2H())
        return Point(x=vec[0,0],y=vec[1,0],w=vec[2,0])
        
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
        
    