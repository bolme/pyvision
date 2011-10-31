# PyVision License
#
# Copyright (c) 2006-2009 David S. Bolme
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

'''
__author__ = "$Author$"
__version__ = "$Revision$"

import PIL.ImageDraw
import PIL.Image
import ImageFont

from PIL.Image import BICUBIC, ANTIALIAS

import numpy
import numpy as np
import cv

import unittest
import os.path


import pyvision
import pyvision as pv

TYPE_MATRIX_2D  = "TYPE_MATRIX2D" 
'''Image was created using a 2D "gray-scale" numpy array'''

TYPE_MATRIX_RGB = "TYPE_MATRIX_RGB" 
'''Image was created using a 3D "color" numpy array'''

TYPE_PIL        = "TYPE_PIL" 
'''Image was created using a PIL image instance'''

TYPE_OPENCV     = "TYPE_OPENCV"
'''Image was created using a OpenCV image instance'''

LUMA = [0.299, 0.587, 0.114, 1.0]
'''Values used when converting color to gray-scale.'''


class Image:
    '''
    The primary purpose of the image class is to provide a structure that can
    transform an image back and fourth for different python libraries such as
    U{PIL<http://www.pythonware.com/products/pil>}, 
    U{OpenCV <http://sourceforge.net/projects/opencvlibrary>}, and 
    U{Scipy<http://www.scipy.org">} Images. This class also
    allows some simple operations on the image such as annotation.
    
    B{Note:} When working with images in matrix format, they are transposed such
    that x = col and y = row.  You can therefore still work with coords
    such that im[x,y] = mat[x,y].
    
    Images have the following attributes:
      - width = width of the image
      - height = height of the image
      - size = (width,height)
      - channels = number of channels: 1(gray), 3(RGB)
      - depth = bitdepth: 8(uchar), 32(float), 64(double)
    '''
 
 
    #------------------------------------------------------------------------
    def __init__(self,data,bw_annotate=False):
        '''
        Create an image from a file or a PIL Image, OpenCV Image, or numpy array.
         
        @param data: this can be a numpy array, PIL image, or opencv image.
        @param bw_annotate: generate a black and white image to make color annotations show up better
        @return: an Image object instance
        '''

        self.filename = None
        self.pil = None
        self.matrix2d = None
        self.matrix3d = None
        self.opencv = None
        self.annotated = None
        self.bw_annotate = bw_annotate
        
        if isinstance(data,numpy.ndarray) and len(data.shape) == 2:
            self.type=TYPE_MATRIX_2D
            self.matrix2d = data
            
            self.width,self.height = self.matrix2d.shape
            self.channels = 1
            
            if self.matrix2d.dtype == numpy.float32:
                self.depth=32
            elif self.matrix2d.dtype == numpy.float64:
                self.depth=64
            else:
                raise TypeError("Unsuppoted format for ndarray images: %s"%self.matrix2d.dtype)
            
        elif isinstance(data,numpy.ndarray) and len(data.shape) == 3 and data.shape[0]==3:
            self.type=TYPE_MATRIX_RGB
            self.matrix3d = data
            self.channels=3
            self.width = self.matrix3d.shape[1]
            self.height = self.matrix3d.shape[2]
            if self.matrix3d.dtype == numpy.float32:
                self.depth=32
            elif self.matrix3d.dtype == numpy.float64:
                self.depth=64
            else:
                raise TypeError("Unsuppoted format for ndarray images: %s"%self.matrix2d.dtype)
            
        elif isinstance(data,PIL.Image.Image) or type(data) == str:
            if type(data) == str:
                # Assume this is a filename
                # TODO: Removing the filename causes errors in other unittest.
                #       Those errors should be corrected.
                self.filename = data
                data = PIL.Image.open(data)
            self.type=TYPE_PIL
            self.pil = data
            self.width,self.height = self.pil.size
                        
            if self.pil.mode == 'L':
                self.channels = 1
            elif self.pil.mode == 'RGB':
                self.channels = 3
            elif self.pil.mode == 'RGBA':
                # 
                self.pil = self.pil.convert('RGB')
                self.channels = 3
            else:
                raise TypeError("Unsuppoted format for PIL images: %s"%self.pil.mode)
            
            self.depth = 8
                        
        elif isinstance(data,cv.iplimage):
            self.type=TYPE_OPENCV
            self.opencv=data 
            
            self.width = data.width
            self.height = data.height
            
            assert data.nChannels in (1,3)
            self.channels = data.nChannels 
            
            assert data.depth in (8,)
            self.depth = data.depth   

        else:
            raise TypeError("Could not create from type: %s %s"%(data,type(data)))
        
        self.size = (self.width,self.height)
        self.data = data
        
    def asBW(self):
        '''
        @return: a gray-scale version of this pyvision image
        '''    
        if self.matrix2d == None:
            self._generateMatrix2D()
        return Image(self.matrix2d)
    
    def asMatrix2D(self):
        '''
        @return: the gray-scale image data as a two dimensional numpy array
        '''
        if self.matrix2d == None:
            self._generateMatrix2D()
        return self.matrix2d

    def asMatrix3D(self):
        '''
        @return: color image data as a 3D array with shape (3(rgb),w,h)
        '''
        if self.matrix3d == None:
            self._generateMatrix3D()
        return self.matrix3d

    def asPIL(self):
        '''
        @return: image data as a pil image
        '''
        if self.pil == None:
            self._generatePIL()
        return self.pil

    def asOpenCV(self):
        '''
        @return: the image data in an OpenCV format
        '''
        if self.opencv == None:
            self._generateOpenCV()
        return self.opencv
        
    def asOpenCVBW(self):
        '''
        @return: the image data in an OpenCV one channel format
        '''
        cvim = self.asOpenCV()
        
        if cvim.nChannels == 1:
            return cvim
        
        elif cvim.nChannels == 3:
            cvimbw = cv.CreateImage(cv.GetSize(cvim), cv.IPL_DEPTH_8U, 1);
            cv.CvtColor(cvim, cvimbw, cv.CV_BGR2GRAY);
            return cvimbw
        
        else:
            raise ValueError("Unsupported opencv image format: nChannels=%d"%cvim.nChannels)
        
    def asThermal(self,clip_negative=False):
        '''
        @returns: a thermal colored representation of this image.
        '''
        w,h = self.size
        mat = self.asMatrix2D()
        if clip_negative:
            mat = mat*(mat > 0.0)
            
        mat = mat - mat.min()
        mat = mat / mat.max()

        therm = np.zeros((3,w,h),dtype=np.float)
        
        mask = mat <= 0.1
        therm[2,:,:] += mask*(0.5 + 0.5*mat/0.1)

        mask = (mat > 0.10) & (mat <= 0.4)
        tmp = (mat - 0.10) / 0.30
        therm[2,:,:] += mask*(1.0-tmp)
        therm[1,:,:] += mask*tmp
        therm[0,:,:] += mask*tmp
        
        mask = (mat > 0.4) & (mat <= 0.7)
        tmp = (mat - 0.4) / 0.3
        therm[2,:,:] += mask*0
        therm[1,:,:] += mask*(1-0.5*tmp)
        therm[0,:,:] += mask*1

        mask = (mat > 0.7) 
        tmp = (mat - 0.7) / 0.3
        therm[2,:,:] += mask*0
        therm[1,:,:] += mask*(0.5-0.5*tmp)
        therm[0,:,:] += mask*1

        return pv.Image(therm)
        

    def asAnnotated(self):
        '''
        @return: the PIL image used for annotation.
        '''
        if self.annotated == None:
            if self.bw_annotate:
                # Make a black and white image that can be annotated with color.
                self.annotated = self.asPIL().convert("L").copy().convert("RGB")
            else:
                # Annotate over color if avalible.
                self.annotated = self.asPIL().copy().convert("RGB")
        return self.annotated
            
    def asHSV(self):
        '''
        @return: an OpenCV HSV encoded image
        '''
        cvim = self.asOpenCV()
        dst = cv.CreateImage(cv.GetSize(cvim), cv.IPL_DEPTH_8U, 3)
        cv.CvtColor(cvim, dst, cv.CV_BGR2HSV)
        
        return dst
        
        
    def asLAB(self):
        '''
        @return: an OpenCV LAB encoded image
        '''
        cvim = self.asOpenCV()
        dst = cv.CreateImage(cv.GetSize(cvim), cv.IPL_DEPTH_8U, 3)
        cv.CvtColor(cvim, dst, cv.CV_BGR2Lab)
        
        return dst
        
        
    def annotateRect(self,rect,color='red', fill_color=None):
        '''
        Draws a rectangle on the annotation image
        
        @param rect: a rectangle of type Rect
        @param color: defined as ('#rrggbb' or 'name') 
        @param fill_color: defined as per color, but indicates the color
        used to fill the rectangle. Specify None for no fill.
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        box = [rect.x,rect.y,rect.x+rect.w,rect.y+rect.h]
        draw.rectangle(box,outline=color,fill=fill_color)
        del draw
        
    def annotateThickRect(self,rect,color='red',width=5):
        '''
        Draws a rectangle on the annotation image
        
        @param rect: a rectangle of type Rect
        @param color: defined as ('#rrggbb' or 'name') 
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        x,y,w,h = [rect.x,rect.y,rect.w,rect.h]
        line = [x,y,x+w,y]
        draw.line(line,fill=color,width=width)
        line = [x,y,x,y+h]
        draw.line(line,fill=color,width=width)
        line = [x,y+h,x+w,y+h]
        draw.line(line,fill=color,width=width)
        line = [x+w,y,x+w,y+h]
        draw.line(line,fill=color,width=width)
        del draw

    def annotateEllipse(self,rect,color='red'):
        '''
        Draws an ellipse on the annotation image
        
        @param rect: the bounding box of the elipse of type Rect
        @param color: defined as ('#rrggbb' or 'name') 
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        box = [rect.x,rect.y,rect.x+rect.w,rect.y+rect.h]
        draw.ellipse(box,outline=color)
        del draw
                
    def annotateLine(self,point1,point2,color='red',width=1):
        '''
        Draws a line from point1 to point2 on the annotation image
    
        @param point1: the starting point as type Point
        @param point2: the ending point as type Point
        @param color: defined as ('#rrggbb' or 'name') 
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        line = [point1.X(),point1.Y(),point2.X(),point2.Y()]
        draw.line(line,fill=color,width=width)
        del draw
        
    def annotateLines(self,points,color='red',width=1):
        '''
        Draws a line from point1 to point2 on the annotation image
    
        @param point1: the starting point as type Point
        @param point2: the ending point as type Point
        @param color: defined as ('#rrggbb' or 'name') 
        '''
        n = len(points)-1
        for i in range(n):
            self.annotateLine(points[i],points[i+1],color=color,width=width)
        
    def annotatePolygon(self,points,color='red',width=1):
        '''
        Draws a line from point1 to point2 on the annotation image
    
        @param points: a list of pv points to be plotted
        @param color: defined as ('#rrggbb' or 'name') 
        @param width: the line width
        '''
        n = len(points)
        for i in range(n):
            j = (i+1)%n 
            self.annotateLine(points[i],points[j],color=color,width=width)
        
    def annotatePoint(self,point,color='red'):
        '''
        Marks a point in the annotation image using a small circle
        
        @param point: the point to mark as type Point
        @param color: defined as ('#rrggbb' or 'name') 
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        box = [point.X()-3,point.Y()-3,point.X()+3,point.Y()+3]
        draw.ellipse(box,outline=color)
        del draw

    def annotatePoints(self,points,color='red'):
        '''
        Marks a point in the annotation image using a small circle
        
        @param point: the point to mark as type Point
        @param color: defined as ('#rrggbb' or 'name') 
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        for point in points:
            box = [point.X()-3,point.Y()-3,point.X()+3,point.Y()+3]
            draw.ellipse(box,outline=color)
        del draw

    def annotateCircle(self,point, radius=3, color='red',fill=None):
        '''
        Marks a circle in the annotation image 
        
        @param point: the center of the circle as type Point
        @param radius: the radius of the circle
        @param color: defined as ('#rrggbb' or 'name') 
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        box = [point.X()-radius,point.Y()-radius,point.X()+radius,point.Y()+radius]
        draw.ellipse(box,outline=color,fill=fill)
        del draw
        
    def annotateLabel(self,point,label,color='red',mark=False, font=None, background=None):        
        '''
        Marks a point in the image with text 
        
        @param point: the point to mark as type Point
        @param label: the text to use as a string
        @param color: defined as ('#rrggbb' or 'name') 
        @param mark: of True or ['right', 'left', 'below', or 'above'] then also mark the point with a small circle
        @param font: An optional PIL.ImageFont font object to use. If None, then the default is used.
        @param background: An optional color that will be used to draw a rectangular background underneath the text.
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        if font == None:
            font = ImageFont.load_default()
        
        tw,th = draw.textsize(label, font=font)
            
        if background != None:
            point2 = pv.Point( point.x + tw, point.y+th)
            draw.rectangle([point.asTuple(), point2.asTuple()], fill=background)
            
        if mark in [True, 'right']:
            draw.text([point.X()+5,point.Y()-th/2],label,fill=color, font=font)
            box = [point.X()-3,point.Y()-3,point.X()+3,point.Y()+3]
            draw.ellipse(box,outline=color)
        elif mark in ['left']:
            draw.text([point.X()-tw-5,point.Y()-th/2],label,fill=color, font=font)
            box = [point.X()-3,point.Y()-3,point.X()+3,point.Y()+3]
            draw.ellipse(box,outline=color)
        elif mark in ['below']:
            draw.text([point.X()-tw/2,point.Y()+5],label,fill=color, font=font)
            box = [point.X()-3,point.Y()-3,point.X()+3,point.Y()+3]
            draw.ellipse(box,outline=color)
        elif mark in ['above']:
            draw.text([point.X()-tw/2,point.Y()-th-5],label,fill=color, font=font)
            box = [point.X()-3,point.Y()-3,point.X()+3,point.Y()+3]
            draw.ellipse(box,outline=color)
        else:
            draw.text([point.X(),point.Y()],label,fill=color, font=font)



        del draw

        
    def annotateDot(self,point,color='red'):
        '''
        Like L{annotatePoint} but only draws a point on the given pixel.
        This is useful to avoid clutter if many points are being annotated.
        
        @param point: the point to mark as type Point
        @param color: defined as ('#rrggbb' or 'name') 
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        draw.point([point.X(),point.Y()],fill=color)
        del draw
        
    #------------------------------------------------------------------------
    def valueNormalize(self):
        '''TODO: Deprecated remove this sometime.'''
        print "WARNING: Image.valueNormalize has been depricated."
        return self.normalize()

    ##
    # @return the type of the image
    def getType(self):
        return self.type
    
    #------------------------------------------------------------------------
    def normalize(self):
        import PIL.ImageOps
        pil = self.asPIL().copy()
        pil = PIL.ImageOps.equalize(pil.convert('L'))
        self.pil = pil
        self.matrix2d = None
        mat = self.asMatrix2D()
        mean = mat.mean()
        std = mat.std()
        mat -= mean
        mat /= std
        self.matrix2d=mat
       
    def equalize(self, bw=True):
        import PIL.ImageOps
        pil = self.asPIL().copy()
        if bw:
            pil = PIL.ImageOps.equalize(pil.convert('L'))
        else:
            pil = PIL.ImageOps.equalize(pil)
        return pv.Image(pil)
    #------------------------------------------------------------------------        
    def _generateMatrix2D(self):
        '''
        Create a matrix version of the image.
        '''
        buffer = self.toBufferGray(32)
        self.matrix2d = numpy.frombuffer(buffer,numpy.float32).reshape(self.height,self.width).transpose()
                    

    def _generateMatrix3D(self):
        '''
        Create a matrix version of the image.
        '''
        buffer = self.toBufferRGB(32)
        self.matrix3d = numpy.frombuffer(buffer,numpy.float32).reshape(self.height,self.width,3).transpose()            

    def _generatePIL(self):
        '''
        Create a PIL version of the image
        '''
        if self.channels == 1:
            self.pil = PIL.Image.fromstring("L",self.size,self.toBufferGray(8))
        elif self.channels == 3:
            self.pil = PIL.Image.fromstring("RGB",self.size,self.toBufferRGB(8))
        else:
            raise NotImplementedError("Cannot convert image from type: %s"%self.type)
        
    def _generateOpenCV(self):
        '''
        Create a color opencv representation of the image.
        TODO: The OpenCV databuffer seems to be automatically swapped from RGB to BGR.  This is counter intuitive.
        '''
        
        w,h = self.size
        if self.channels == 1:
            gray = cv.CreateImage((w,h),cv.IPL_DEPTH_8U,1)
            cv.SetData(gray,self.toBufferGray(8))
            self.opencv = gray
        elif self.channels == 3:
            rgb = cv.CreateImage((w,h),cv.IPL_DEPTH_8U,3)
            bgr = cv.CreateImage((w,h),cv.IPL_DEPTH_8U,3)
            cv.SetData(rgb, self.toBufferRGB(8))
            # convert from RGB to BGR
            cv.CvtColor(rgb,bgr,cv.CV_RGB2BGR)
            self.opencv=bgr
        else:
            raise NotImplementedError("Cannot convert image from type: %s"%self.type)
                
        
    def toBufferGray(self,depth):
        '''
            returns the image data as a binary python string.
        '''
        buffer = None
        if self.type == TYPE_PIL:
            pil = self.pil
            if pil.mode != 'L':
                pil = pil.convert('L')
            buffer = pil.tostring()
        elif self.type == TYPE_MATRIX_2D:
            buffer = self.matrix2d.transpose().tostring()
        elif self.type == TYPE_MATRIX_RGB:
            mat = self.matrix3d
            mat = LUMA[0]*mat[0] + LUMA[1]*mat[1] + LUMA[2]*mat[2]
            buffer = mat.transpose().tostring()
        elif self.type == TYPE_OPENCV:
            if self.channels == 1:
                buffer = self.opencv.tostring()
            elif self.channels == 3:
                w,h = self.width,self.height
                gray = cv.CreateImage((w,h),cv.IPL_DEPTH_8U,1)
                cv.CvtColor( self.opencv, gray, cv.CV_BGR2GRAY );
                buffer = gray.tostring()
            else:
                raise TypeError("Operation not supported for image type.")
        else:
            raise TypeError("Operation not supported for image type.")
        
        assert buffer
            
        if depth == self.depth:
            return buffer
        
        else:
            types = {8:numpy.uint8,32:numpy.float32,64:numpy.float64}
            
            # convert the buffer to data
            data = numpy.frombuffer(buffer,types[self.depth])
            
            if depth==8:
                # Make sure the data is in a valid range
                max_value = data.max()
                min_value = data.min()
                data_range = max_value - min_value
                if max_value <= 255 and min_value >= 0 and data_range >= 150:
                    # assume the values are already in a good range for the
                    # 8 bit image
                    pass
                else:
                    # Rescale the values from 0 to 255 
                    if max_value == min_value:
                        max_value = min_value+1
                    data = (255.0/(max_value-min_value))*(data-min_value)
            
            data = data.astype(types[depth])
            return data.tostring()
        

    def toBufferRGB(self,depth):
        '''
            returns the image data as a binary python string.
        '''
        buffer = None
        if self.type == TYPE_PIL:
            pil = self.pil
            if pil.mode != 'RGB':
                pil = pil.convert('RGB')
            buffer = pil.tostring()
        elif self.type == TYPE_MATRIX_2D:
            mat = self.matrix2d.transpose()
            tmp = np.zeros((3,self.height,self.width),numpy.float32)
            tmp[0,:] = mat
            tmp[1,:] = mat
            tmp[2,:] = mat
            buffer = mat.tostring()            
        elif self.type == TYPE_MATRIX_RGB:
            mat = self.matrix3d.transpose()
            buffer = mat.tostring()
        elif self.type == TYPE_OPENCV:
            w,h = self.width,self.height
            if self.channels == 3:
                rgb = cv.CreateImage((w,h),cv.IPL_DEPTH_8U,3)
                cv.CvtColor( self.opencv, rgb, cv.CV_BGR2RGB );
                buffer = rgb.tostring()
            elif self.channels == 1:
                rgb = cv.CreateImage((w,h),cv.IPL_DEPTH_8U,3)
                cv.CvtColor( self.opencv, rgb, cv.CV_GRAY2RGB );
                buffer = rgb.tostring()
            else:
                raise TypeError("Operation not supported for image type.")
        else:
            raise TypeError("Operation not supported for image type.")
        
        assert buffer
            
        if depth == self.depth:
            return buffer
        
        else:
            types = {8:numpy.uint8,32:numpy.float32,64:numpy.float64}
            
            # convert the buffer to data
            data = numpy.frombuffer(buffer,types[self.depth])
            
            if depth==8:
                # Make sure the data is in a valid range
                max_value = data.max()
                min_value = data.min()
                data_range = max_value - min_value
                if max_value <= 255 and min_value >= 0 and data_range >= 50:
                    # assume the values are already in a good range for the
                    # 8 bit image
                    pass
                else:
                    # Rescale the values from 0 to 255 
                    if max_value == min_value:
                        max_value = min_value+1
                    data = (255.0/(max_value-min_value))*(data-min_value)
            
            data = data.astype(types[depth])
            return data.tostring()
        

    def toBufferRGBA(self,depth):
        '''
            returns the image data as a binary python string.
            TODO: Not yet implemented
        '''

    def resize(self, newSize):
        ''' Returns a resized version of the image. This is a convenience function.
        For more control, look at the Affine class for arbitrary transformations.
        @param newSize: tuple (new_width, new_height)
        @returns: a new pyvision image that is the resized version of this image.
        ''' 
        tmp = self.asPIL()
        if newSize[0] < self.size[0] or newSize[1] < self.size[1]:
            #because at least one dimension is being shrinked, we need to use ANTIALIAS filter
            tmp = tmp.resize(newSize, ANTIALIAS)        
        else:
            #use bicubic interpolation
            tmp = tmp.resize(newSize, BICUBIC)

        return pyvision.Image(tmp)
    
    def scale(self, scale):
        ''' Returns a scaled version of the image. This is a convenience function.
        For more control, look at the Affine class for arbitrary transformations.
        @param scale: a float indicating the scale factor
        @returns: a new pyvision image that is the scaled version of this image.
        ''' 
        w,h = self.size
        new_size = (int(round(scale*w)),int(round(scale*h)))
        return self.resize(new_size)
    
    def copy(self):
        '''
        Returns a new pv.Image which is a copy of (only) the current image.
        Other internal data stored by the current pv.Image will NOT be copied.
        This method uses cv.CloneImage so that the underlying image data will be
        disconnected from the original data. (Deep copy)
        '''
        imgdat = self.asOpenCV()
        imgdat2 = cv.CloneImage(imgdat)
        return pv.Image(imgdat2)
    
    def crop(self, rect, size=None, interpolation=None, return_affine=False):
        '''
        Crops an image to the given rectangle. Rectangle parameters are rounded to nearest 
        integer values.  High quality resampling.  The default behavior is to use cv.GetSubRect
        to crop the image.  This returns a slice the OpenCV image so modifying the resulting
        image data will also modify the data in this image.  If a size is provide a new OpenCV
        image is created for that size and cv.Resize is used to copy the image data. If the 
        bounds of the rectangle are outside the image, an affine transform (pv.AffineFromRect)
        is used to produce the croped image to properly handle regions outside the image.
        In this case the downsampling quality may not be as good. 
        
        @param rect: a Rectangle defining the region to be cropped.
        @param size: a new size for the returned image.  If None the result is not resized.
        @param interpolation: None = Autoselect or one of CV_INTER_AREA, CV_INTER_NN, CV_INTER_LINEAR, CV_INTER_BICUBIC
        @param return_affine: If True, also return an affine transform that can be used to transform points.
        @returns: a cropped version of the image or if return affine a tuple of (image,affine)
        @rtype: pv.Image
        '''
        # Notes: pv.Rect(0,0,w,h) should return the entire image. Since pixel values
        # are indexed by zero this means that upper limits are not inclusive: x from [0,w)
        # and y from [0,h)
        x,y,w,h = rect.asTuple()
       
        x = int(np.round(x))
        y = int(np.round(y))
        w = int(np.round(w))
        h = int(np.round(h))
        
        if x < 0 or y < 0 or x+w > self.size[0] or y+h > self.size[1]:
            if size == None:
                size = (w,h)
            
            #print size
            affine = pv.AffineFromRect(pv.Rect(x,y,w,h),size)
            im = affine(self)
            if return_affine:
                return im,affine
            else:
                return im
        
        cvim = self.asOpenCV()
                
        subim = cv.GetSubRect(cvim,(x,y,w,h))
        
        affine = pv.AffineTranslate(-x,-y,(w,h))
        
        if size == None:
            size = (w,h)
        #    if return_affine:
        #        return pv.Image(subim),affine
        #    else:
        #        return pv.Image(subim)
        
        new_image = cv.CreateImage(size,cvim.depth,cvim.nChannels)
        
        if interpolation == None:
            
            if size[0] < w or size[1] < y:
                # Downsampling so use area interpolation
                interpolation = cv.CV_INTER_AREA
            else:
                # Upsampling so use linear
                interpolation = cv.CV_INTER_CUBIC

        cv.Resize(subim,new_image,interpolation)
        
        affine = pv.AffineNonUniformScale(float(size[0])/w,float(size[1])/h,size)*affine
        
        if return_affine: 
            return pv.Image(new_image),affine
        else:
            return pv.Image(new_image)
        
    def save(self,filename,annotations=False):
        '''
        Save the image to a file.  This is performed by converting to PIL and
        then saving to a file based on on the extension.
        '''
        if filename[-4:] == ".raw":
            # TODO: save as a matrix
            raise NotImplementedError("Cannot save as a matrix")
        #elif filename[-4:] == ".mat":
            # TODO: save as a matlab file
        #    raise NotImplementedError("Cannot save in matlab format")
        else:
            if annotations:
                self.asAnnotated().save(filename)
            else:
                self.asPIL().save(filename)
            
    def show(self, window="PyVisionImage", pos=None, delay=1, size=None):
        '''
        Displays the annotated version of the image using OpenCV highgui
        @param window: the name of the highgui window to use, this should
            already have been created using cv.NamedWindow or set newWindow=True
        @param pos: If newWindow, then pos is the (x,y) coordinate for the new window 
        @param delay: A delay in milliseconds to wait for keyboard input (passed to cv.WaitKey).  
            0 delays indefinitely, 1 is good for live updates and animations.  The window
            will disappear after the program exits.  
        @param size: Optional output size for image, None=native size.
        @returns: a key press event,
        '''
        cv.NamedWindow(window)
        
        if pos != None:
            cv.MoveWindow(window, pos[0], pos[1])
            
            
            
        if size != None:
            x = pyvision.Image(self.resize(size).asAnnotated())
        else:
            x = pyvision.Image(self.asAnnotated())    
            
        cv.ShowImage(window, x.asOpenCV() )
        key = cv.WaitKey(delay=delay)
        del x
        return key
##
# Convert a 32bit opencv matrix to a numpy matrix
def OpenCVToNumpy(cvmat):
    '''
    Convert an OpenCV matrix to a numpy matrix.
    
    Based on code from: http://opencv.willowgarage.com/wiki/PythonInterface
    '''
    depth2dtype = {
            cv.CV_8U: 'uint8',
            cv.CV_8S: 'int8',
            cv.CV_16U: 'uint16',
            cv.CV_16S: 'int16',
            cv.CV_32S: 'int32',
            cv.CV_32F: 'float32',
            cv.CV_64F: 'float64',
        }
    assert cvmat.channels == 1
    r = cvmat.rows
    c = cvmat.cols
    
    a = np.fromstring(
             cvmat.tostring(),
             dtype=depth2dtype[cvmat.type],
             count=r*c)
    a.shape = (r,c)
    return a

##
# Convert a numpy matrix to a 32bit opencv matrix
def NumpyToOpenCV(a):
    '''
    Convert a numpy matrix to an OpenCV matrix. 
    
    Based on code from: http://opencv.willowgarage.com/wiki/PythonInterface
    '''
    dtype2depth = {
        'uint8':   cv.CV_8U,
        'int8':    cv.CV_8S,
        'uint16':  cv.CV_16U,
        'int16':   cv.CV_16S,
        'int32':   cv.CV_32S,
        'float32': cv.CV_32F,
        'float64': cv.CV_64F,
    }
  
    assert len(a.shape) == 2
        
    r,c = a.shape
    cv_im = cv.CreateMat(r,c,dtype2depth[str(a.dtype)])
    cv.SetData(cv_im, a.tostring())
    return cv_im


class _TestImage(unittest.TestCase):
    
    def setUp(self):
        # Assume these work correctly
        self.im     = pv.Image(os.path.join(pyvision.__path__[0],"data","nonface","NONFACE_46.jpg"))
        self.pil    = self.im.asPIL()
        self.mat    = self.im.asMatrix2D()
        assert self.mat.shape[0] == 640
        assert self.mat.shape[1] == 480
        self.mat3d  = self.im.asMatrix3D()
        assert self.mat3d.shape[0] == 3
        assert self.mat3d.shape[1] == 640
        assert self.mat3d.shape[2] == 480
        self.opencv = self.im.asOpenCV()
            
    def test_PILToBufferGray(self):
        w,h = self.im.size
        buffer = self.im.toBufferGray(8)
        self.assertEqual(len(buffer),w*h)
        buffer = self.im.toBufferGray(32)
        self.assertEqual(len(buffer),4*w*h)
        buffer = self.im.toBufferGray(64)
        self.assertEqual(len(buffer),8*w*h)

    def test_Matrix3DToBufferGray(self):
        im = Image(self.mat3d)
        w,h = im.size
        buffer = im.toBufferGray(8)
        self.assertEqual(len(buffer),w*h)
        buffer = im.toBufferGray(32)
        self.assertEqual(len(buffer),4*w*h)
        buffer = im.toBufferGray(64)
        self.assertEqual(len(buffer),8*w*h)

    def test_Matrix2DToBufferGray(self):
        im = Image(self.mat)
        w,h = im.size
        buffer = im.toBufferGray(8)
        self.assertEqual(len(buffer),w*h)
        buffer = im.toBufferGray(32)
        self.assertEqual(len(buffer),4*w*h)
        buffer = im.toBufferGray(64)
        self.assertEqual(len(buffer),8*w*h)

    def test_PILToMatrix2D(self):
        im = self.im
        pil = im.asPIL().convert('L')
        pil = pil.resize((180,120))
        im = Image(pil)
        mat = im.asMatrix2D()
        for i in range(im.width):
            for j in range(im.height):
                self.assertAlmostEqual(pil.getpixel((i,j)),mat[i,j])
        
    def test_Matrix2DToPIL(self):
        im = Image(self.mat[:180,:120])
        pil = im.asPIL()
        mat = im.asMatrix2D()
        for i in range(im.width):
            for j in range(im.height):
                self.assertAlmostEqual(pil.getpixel((i,j)),mat[i,j])

    def test_PILToMatrix3D(self):
        pil = self.im.asPIL().resize((180,120))
        im = Image(pil)
        mat = im.asMatrix3D()
        for i in range(im.width):
            for j in range(im.height):
                for c in range(3):
                    self.assertAlmostEqual(pil.getpixel((i,j))[c],mat[c,i,j])

    def test_Matrix3D2PIL(self):
        im = Image(self.mat3d[:,:180,:120])
        pil = self.im.asPIL()
        mat = im.asMatrix3D()
        for i in range(im.width):
            for j in range(im.height):
                for c in range(3):
                    self.assertAlmostEqual(pil.getpixel((i,j))[c],mat[c,i,j])
        
    def test_PILToOpenCV(self):
        pil = self.im.asPIL().resize((180,120))
        im = Image(pil)
        cv = im.asOpenCV()
        #Uncomment this code to compare saved images
        #from opencv import highgui
        #highgui.cvSaveImage('/tmp/cv.png',cv)
        #pil.show()
        #Image('/tmp/cv.png').show()

        for i in range(im.width):
            for j in range(im.height):
                for c in range(3):
                    self.assertAlmostEqual(pil.getpixel((i,j))[c],ord(cv.tostring()[i*3+j*im.width*3+2-c]))
        
    def test_OpenCVToPIL(self):
        pil = self.im.asPIL().resize((180,120))
        im = Image(pil)
        cv = im.asOpenCV()
        pil = Image(cv).asPIL()

        for i in range(im.width):
            for j in range(im.height):
                for c in range(3):
                    self.assertAlmostEqual(pil.getpixel((i,j))[c],ord(cv.tostring()[i*3+j*im.width*3+2-c]))
        
    def test_OpenCVToPILGray(self):
        pil = self.im.asPIL().resize((180,120)).convert('L')
        im = Image(pil)
        cv = im.asOpenCV()
        im = Image(cv)
        pil = im.asPIL()
        
        #Uncomment this code to compare saved images
        #from opencv import highgui
        #highgui.cvSaveImage('/tmp/cv.png',cv)
        #pil.show()
        #Image('/tmp/cv.png').show()
        
        # TODO: There seems to be data loss in the conversion from pil to opencv and back.  Why?
        #for i in range(im.width):
        #    for j in range(im.height):
        #        self.assertAlmostEqual(pil.getpixel((i,j)),ord(cv.imageData[i*3+j*im.width*3]))
        
    def test_BufferToOpenCV(self):
        pil = self.im.asPIL().resize((180,120))
        im = Image(pil)
        cvim = im.asOpenCV()
        buffer = im.toBufferRGB(8)

        for i in range(im.width):
            for j in range(im.height):
                for c in range(3):
                    self.assertAlmostEqual(ord(buffer[i*3+j*im.width*3+c]),ord(cvim.tostring()[i*3+j*im.width*3+2-c]))
     
    def test_asOpenCVBW(self):
        pass #TODO: Create tests for this method.
        
    def test_MatConvertOpenCVToNumpy(self):
        r,c = 10,20
        cvmat = cv.CreateMat(r,c,cv.CV_32F)
        for i in range(r):
            for j in range(c):
                cvmat[i,j] = i*j
        mat = OpenCVToNumpy(cvmat)
        self.assert_(mat.shape == (r,c))
        for i in range(r):
            for j in range(c):
                self.assert_(mat[i,j] == cvmat[i,j])
        
        
    def test_MatConvertNumpyToOpenCV(self):
        r,c = 10,20
        mat = np.zeros((r,c),dtype=np.float32)
        for i in range(r):
            for j in range(c):
                mat[i,j] = i*j
        cvmat = NumpyToOpenCV(mat)
        self.assert_(mat.shape == (r,c))
        for i in range(r):
            for j in range(c):
                self.assert_(mat[i,j] == cvmat[i,j])
                
    def test_ImageCropOutofBounds(self):
        rect = pv.Rect(-3, -2, 35, 70)
        imcrop = self.im.crop(rect)
        cropSize = imcrop.size
        
        self.assertEquals((35,70), cropSize)
        
        rect = pv.Rect(620, 460, 35, 70)
        imcrop = self.im.crop(rect)
        cropSize = imcrop.size
        
        self.assertEquals((35,70), cropSize)
        
    def test_asHSV(self):
        im = pv.Image(os.path.join(pyvision.__path__[0],"data","misc","baboon.jpg"))
        hsv = im.asHSV()
        im = pv.Image(hsv)
        #im.show(delay=0)

        im = pv.Image(os.path.join(pyvision.__path__[0],"data","misc","baboon_bw.jpg"))
        self.assertRaises(Exception, im.asHSV)

        
        
        
        
