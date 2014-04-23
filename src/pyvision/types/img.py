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

# PIL Imports
import PIL.ImageDraw
import PIL.Image
from PIL.Image import BICUBIC, ANTIALIAS
import PIL.ImageFont as ImageFont

# Imaging imports
import numpy
import numpy as np
import cv
import cv2
import pyvision
import pyvision as pv

import cStringIO
import exif
import os

# iPython support for ipython notebook
try:
    import pylab
    import IPython
except:
    pass # do nothing


TYPE_MATRIX_2D  = "TYPE_MATRIX2D" 
'''Image was created using a 2D "gray-scale" numpy array'''

TYPE_MATRIX_RGB = "TYPE_MATRIX_RGB" 
'''Image was created using a 3D "color" numpy array'''

TYPE_PIL        = "TYPE_PIL" 
'''Image was created using a PIL image instance'''

TYPE_OPENCV     = "TYPE_OPENCV"
'''Image was created using a OpenCV image instance'''

TYPE_OPENCV2     = "TYPE_OPENCV2"
'''Image was created using a OpenCV image instance'''

TYPE_OPENCV2BW     = "TYPE_OPENCV2BW"
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
    such that im[x,y] = mat[x,y]. #
    
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
        self.opencv2 = None
        self.opencv2bw = None
        self.annotated = None
        self.bw_annotate = bw_annotate
        
        # Convert floating point ipl images to numpy arrays
        if isinstance(data,cv.iplimage) and data.nChannels == 3 and data.depth == 32:
            w,h = cv.GetSize(data)
            data = np.frombuffer(data.tostring(),dtype=np.float32)
            data.shape = (h,w,3)
            data = data.transpose((2,1,0))
            data = data[::-1,:,:]
            
        # Convert floating point ipl images to numpy arrays
        if isinstance(data,cv.iplimage) and data.nChannels == 1 and data.depth == 32:
            w,h = cv.GetSize(data)
            data = np.frombuffer(data.tostring(),dtype=np.float32)
            data.shape = (h,w)
            data = data.transpose((2,1,0))
            data = data[::-1,:,:]
            
        # Numpy format
        if isinstance(data,numpy.ndarray) and len(data.shape) == 2 and data.dtype != np.uint8:
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
            
        # OpenCV2 gray scale format
        elif isinstance(data,numpy.ndarray) and len(data.shape) == 2 and data.dtype == np.uint8:
            self.type=TYPE_OPENCV2BW
            self.opencv2bw = data
            
            self.height,self.width = self.opencv2bw.shape
            self.channels = 1
            self.depth=8
            
        # Numpy color format    
        elif isinstance(data,numpy.ndarray) and len(data.shape) == 3 and data.shape[0]==3 and data.dtype != np.uint8:
            self.type=TYPE_MATRIX_RGB
            self.matrix3d = data
            self.channels=3
            self.width = self.matrix3d.shape[1]
            self.height = self.matrix3d.shape[2]
            # set the types
            if self.matrix3d.dtype == numpy.float32:
                self.depth=32
            elif self.matrix3d.dtype == numpy.float64:
                self.depth=64
            else:
                raise TypeError("Unsuppoted format for ndarray images: %s"%self.matrix2d.dtype)
            
        # OpenCV2 color format    
        elif isinstance(data,numpy.ndarray) and len(data.shape) == 3 and data.shape[2]==3 and data.dtype == np.uint8:
            self.type=TYPE_OPENCV2
            self.opencv2 = data
            self.channels=3
            self.width = self.opencv2.shape[1]
            self.height = self.opencv2.shape[0]
            self.depth=8
            
        # Load as a pil image    
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
            #elif self.pil.mode == 'RGBA':
                # 
            #    self.pil = self.pil.convert('RGB')
            #    self.channels = 3
            else:
                self.pil.convert('RGB')
                self.channels = 3
                #   raise TypeError("Unsuppoted format for PIL images: %s"%self.pil.mode)
            
            self.depth = 8
        
        # opencv format             
        elif isinstance(data,cv.iplimage):
            self.type=TYPE_OPENCV
            self.opencv=data 
            
            self.width = data.width
            self.height = data.height
                        
            assert data.nChannels in (1,3)
            self.channels = data.nChannels 
            
            assert data.depth == 8
            self.depth = data.depth   

        # unknown type
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
        
    def asOpenCV2(self):
        '''
        @return: the image data in an OpenCV format that is a numpy array of shape (h,w,3) of uint8
        '''
        if self.opencv2 == None:
            self._generateOpenCV2()
        return self.opencv2
        
    def asOpenCV2BW(self):
        '''
        @return: the image data in an OpenCV format that is a numpy array of shape (h,w,1) of uint8
        '''
        if self.opencv2bw == None:
            self._generateOpenCV2BW()
        return self.opencv2bw
        
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
            
        # Rescale 0.0 to 1.0
        mat = mat - mat.min()
        mat = mat / mat.max()

        therm = np.zeros((3,w,h),dtype=np.float)
        
        # Black to blue
        mask = mat <= 0.1
        therm[2,:,:] += mask*(0.5 + 0.5*mat/0.1)
        
        # blue to yellow
        mask = (mat > 0.10) & (mat <= 0.4)
        tmp = (mat - 0.10) / 0.30
        therm[2,:,:] += mask*(1.0-tmp)
        therm[1,:,:] += mask*tmp
        therm[0,:,:] += mask*tmp
        
        # yellow to orange
        mask = (mat > 0.4) & (mat <= 0.7)
        tmp = (mat - 0.4) / 0.3
        therm[2,:,:] += mask*0
        therm[1,:,:] += mask*(1-0.5*tmp)
        therm[0,:,:] += mask*1

        # the orange to red
        mask = (mat > 0.7) 
        tmp = (mat - 0.7) / 0.3
        therm[2,:,:] += mask*0
        therm[1,:,:] += mask*(0.5-0.5*tmp)
        therm[0,:,:] += mask*1

        return pv.Image(therm)
        

    def asAnnotated(self, as_type="PIL"):
        '''
        @param as_type: Specify either "PIL" or "PV". If
        "PIL" (default) then the return type is a PIL image.
        If "PV", then the return type is a pyvision image,
        where the annotations have been 'flattened' onto
        the original source image.
        @return: the PIL image used for annotation.
        '''
        if self.annotated == None:
            if self.bw_annotate:
                # Make a black and white image that can be annotated with color.
                self.annotated = self.asPIL().convert("L").copy().convert("RGB")
            else:
                # Annotate over color if available.
                self.annotated = self.asPIL().copy().convert("RGB")
                
        if as_type.upper() == "PV":
            return pv.Image(self.annotated)
        else:
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
    
    def getExif(self,output='simple'):
        '''
        This function returns the exif headers for an image.  This only works 
        for images that have been read from disk.
        
        @param output: select 'simple' or 'full'. 'full' output contains additional metadata.
        @returns: a dictionary of EXIF data.
        '''
        if self.type == TYPE_PIL and self.filename != None:
            result = {}
            info = self.pil._getexif()
            if info == None:
                return None
            
            # iterate through exif tags
            for key,value in info.iteritems():
                tag = "ukn_%s"%key
                # translate tags to text
                if exif.EXIF_TAGS.has_key(key):
                    tag = exif.EXIF_TAGS[key][0]
                    datatype = exif.EXIF_TAGS[key][1]
                    category = exif.EXIF_TAGS[key][2]
                    description = exif.EXIF_TAGS[key][3]
                # convert to floats    
                if isinstance(value,tuple) and len(value) == 2 and value[1] > 0:
                    value = float(value[0])/float(value[1])
                if output == 'simple': 
                    result[tag] = value
                else:   
                    result[tag] = (value,key,datatype,category,description)
            return result
        else:
            return None    
        
        
    def annotateRect(self,rect,color='red', fill_color=None, alpha=1.0):
        '''
        Draws a rectangle on the annotation image
        
        @param rect: a rectangle of type Rect
        @param color: defined as ('#rrggbb' or 'name') 
        @param fill_color: defined as per color, but indicates the color
        used to fill the rectangle. Specify None for no fill.
        @param alpha: Ignored if no fill. Otherwise, this value controls
        how opaque the fill is. Specify 1.0 (default) for a fully opaque
        fill, or 0.0 for fully transparent. A value of 0.3, for example,
        would show a partially transparent filled rectangle over
        the background image.
        '''
        im = self.asAnnotated()
        box = rect.box()
        offset = (box[0],box[1])
        #this supports filling a rectangle that is semi-transparent
        if fill_color:
            (r,g,b) = PIL.ImageColor.getrgb(fill_color)
            rect_img = PIL.Image.new('RGBA', (int(rect.w),int(rect.h)), (r,g,b,int(alpha*255)))
            im.paste(rect_img,offset,mask=rect_img) #use 'paste' method to support transparency
        
        #just draws the rect outline in the outline color
        draw = PIL.ImageDraw.Draw(im)
        draw.rectangle(box,outline=color,fill=None)
        del draw
        
    def annotateImage(self,im,rect,color='red', fill_color=None):
        '''
        Draws an image
        
        @param im: the image to render
        @param rect: a rectangle of type Rect
        @param color: defined as ('#rrggbb' or 'name') 
        @param fill_color: defined as per color, but indicates the color
        used to fill the rectangle. Specify None for no fill.
        '''
        # Reduce the size of the image
        thumb = im.thumbnail((rect.w,rect.h))
        x = int(rect.x + rect.w/2 - thumb.size[0]/2)
        y = int(rect.y + rect.h/2 - thumb.size[1]/2)
        
        # Get the annotated image buffer
        pil = self.asAnnotated()
        
        # Draw a rect
        draw = PIL.ImageDraw.Draw(pil)
        box = [rect.x,rect.y,rect.x+rect.w,rect.y+rect.h]
        draw.rectangle(box,outline=None,fill=fill_color)
        del draw
        
        # Paste the image
        pil.paste(im.asPIL(),(x,y))
        
        # Draw a rect over the top
        draw = PIL.ImageDraw.Draw(pil)
        box = [rect.x,rect.y,rect.x+rect.w,rect.y+rect.h]
        draw.rectangle(box,outline=color,fill=None)
        del draw
        
    def annotateThickRect(self,rect,color='red',width=5):
        '''
        Draws a rectangle on the annotation image
        
        @param rect: a rectangle of type Rect
        @param color: defined as ('#rrggbb' or 'name') 
        '''
        # get the image buffer
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        x,y,w,h = [rect.x,rect.y,rect.w,rect.h]
        
        # Draw individual lines
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
        
    def annotateMask(self,mask,color='red'):
        '''
        Shades the contents of a mask.
    
        @param mask: a numpy array showing the mask.
        @param color: defined as ('#rrggbb' or 'name') 
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        pil = pv.Image(1.0*mask).asPIL()
        pil = pil.convert('1')
        draw.bitmap((0,0), pil, fill=color)
        del draw
        
    def annotatePolygon(self,points,color='red',width=1,fill=None):
        '''
        Draws a line from point1 to point2 on the annotation image
    
        @param points: a list of pv points to be plotted
        @param color: defined as ('#rrggbb' or 'name') 
        @param width: the line width
        '''
        # Fill the center
        if fill != None:
            im = self.asAnnotated()
            draw = PIL.ImageDraw.Draw(im)
            poly = [(point.X(),point.Y()) for point in points]
            draw.polygon(poly,outline=None,fill=fill)
            del draw
        
        # Draw lines    
        if color != None:
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
        
    def annotateArc(self,point, radius=3, startangle=0, endangle=360, color='red'):
        '''
        Draws a circular arc on the image.
        @param point: the center of the circle as type Point
        @param radius: the radius of the circle
        @param startangle: the starting angle of the arc segment to be drawn, in degrees
        @param endangle: the ending angle in degrees. Arc will be drawn clockwise from
        starting angle to ending angle.
        @param color: defined as ('#rrggbb' or 'name') 
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        box = [int(point.X()-radius),int(point.Y()-radius),
               int(point.X()+radius),int(point.Y()+radius)]
        draw.arc(box, int(startangle), int(endangle), fill=color)
        del draw
                
    def annotateLabel(self,point,label,color='red',mark=False, font=None, background=None):        
        '''
        Marks a point in the image with text 
        
        @param point: the point to mark as type Point
        @param label: the text to use as a string
        @param color: defined as ('#rrggbb' or 'name') 
        @param mark: of True or ['right', 'left', 'below', or 'above','centered'] then also mark the point with a small circle
        @param font: An optional PIL.ImageFont font object to use. Alternatively, specify an integer and the label
        will use Arial font of that size. If None, then the default is used.
        @param background: An optional color that will be used to draw a rectangular background underneath the text.
        '''
        # Get the image buffer
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        
        # Load the font
        if font == None:
            font = ImageFont.load_default()
        elif isinstance(font,int):
            font = ImageFont.truetype(pv.FONT_ARIAL, font)
        
        # Compute the size
        tw,th = draw.textsize(label, font=font)
        
        # Select the position relative to the point
        if mark in [True, 'right']:
            textpt = pv.Point(point.X()+5,point.Y()-th/2)
            box = [point.X()-3,point.Y()-3,point.X()+3,point.Y()+3]
        elif mark in ['left']:
            textpt = pv.Point(point.X()-tw-5,point.Y()-th/2)
            box = [point.X()-3,point.Y()-3,point.X()+3,point.Y()+3]
        elif mark in ['below']: #
            textpt = pv.Point(point.X()-tw/2,point.Y()+5)
            box = [point.X()-3,point.Y()-3,point.X()+3,point.Y()+3]            
        elif mark in ['above']:
            textpt = pv.Point(point.X()-tw/2,point.Y()-th-5)
            box = [point.X()-3,point.Y()-3,point.X()+3,point.Y()+3]
        elif mark in ['centered']:
            textpt = pv.Point(point.X()-tw/2,point.Y()-th/2)
        else:
            textpt = point
            
        # Fill in the background
        if background != None:
            point2 = pv.Point( textpt.x + tw, textpt.y+th)
            draw.rectangle([textpt.asTuple(), point2.asTuple()], fill=background)
        
        # Render the text
        draw.text([textpt.x,textpt.y],label,fill=color, font=font)    
        
        if mark not in [False,None,'centered']:           
            draw.ellipse(box,outline=color)

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
        
        
    def valueNormalize(self):
        '''TODO: Deprecated remove this sometime.'''
        print "WARNING: Image.valueNormalize has been deprecated."
        return self.normalize()


    def getType(self):
        '''Return the type of the image.'''
        return self.type
    

    def normalize(self):
        ''' Equalize and normalize the image. '''
        import PIL.ImageOps
        # Create a copy
        pil = self.asPIL().copy()
        
        # Equalize
        pil = PIL.ImageOps.equalize(pil.convert('L'))
        self.pil = pil
        self.matrix2d = None
        
        # Normalize
        mat = self.asMatrix2D()
        mean = mat.mean()
        std = mat.std()
        mat -= mean
        mat /= std
        self.matrix2d=mat
       
       
    def equalize(self, bw=True):
        ''' Equalize the image '''
        import PIL.ImageOps
        pil = self.asPIL().copy()
        if bw:
            pil = PIL.ImageOps.equalize(pil.convert('L'))
        else:
            pil = PIL.ImageOps.equalize(pil)
        return pv.Image(pil)


    def _generateMatrix2D(self):
        '''
        Create a matrix version of the image.
        '''
        data_buffer = self.toBufferGray(32)
        self.matrix2d = numpy.frombuffer(data_buffer,numpy.float32).reshape(self.height,self.width).transpose()
                    

    def _generateMatrix3D(self):
        '''
        Create a matrix version of the image.
        '''
        data_buffer = self.toBufferRGB(32)
        self.matrix3d = numpy.frombuffer(data_buffer,numpy.float32).reshape(self.height,self.width,3).transpose()            

    def _generatePIL(self):
        '''
        Create a PIL version of the image
        '''
        if self.channels == 1:
            self.pil = PIL.Image.frombytes("L",self.size,self.toBufferGray(8))
        elif self.channels == 3:
            self.pil = PIL.Image.frombytes("RGB",self.size,self.toBufferRGB(8))
        else:
            raise NotImplementedError("Cannot convert image from type: %s"%self.type)
        
    def _generateOpenCV(self):
        '''
        Create a color opencv representation of the image.
        TODO: The OpenCV databuffer seems to be automatically swapped from RGB to BGR.  This is counter intuitive.
        '''
        
        w,h = self.size
        # generate a grayscale opencv image
        if self.channels == 1:
            gray = cv.CreateImage((w,h),cv.IPL_DEPTH_8U,1)
            cv.SetData(gray,self.toBufferGray(8))
            self.opencv = gray
        # Generate a color opencv image
        elif self.channels == 3:
            rgb = cv.CreateImage((w,h),cv.IPL_DEPTH_8U,3)
            bgr = cv.CreateImage((w,h),cv.IPL_DEPTH_8U,3)
            cv.SetData(rgb, self.toBufferRGB(8))
            # convert from RGB to BGR
            cv.CvtColor(rgb,bgr,cv.CV_RGB2BGR)
            self.opencv=bgr
        else:
            raise NotImplementedError("Cannot convert image from type: %s"%self.type)
                
    def _generateOpenCV2(self):
        '''
        Create a matrix version of the image compatible with OpenCV 2 (cv2) in BGR format.
        '''
        data_buffer = self.toBufferRGB(8)
        self.opencv2 = cv2.cvtColor(numpy.frombuffer(data_buffer,numpy.uint8).reshape(self.height,self.width,3),cv2.COLOR_RGB2BGR)            


    def _generateOpenCV2BW(self):
        '''
        Create a matrix version of the image compatible with OpenCV 2 (cv2) in BGR format.
        '''
        data_buffer = self.toBufferGray(8)
        self.opencv2bw = numpy.frombuffer(data_buffer,numpy.uint8).reshape(self.height,self.width)            


        
    def toBufferGray(self,depth):
        '''
        @param depth: Use 8, 32, or 64, to specify the bit depth of the pixels.
        @return: the image data as a binary python string.
        '''
        image_buffer = None
        if self.type == TYPE_PIL:
            # Convert to gray and then get buffer
            pil = self.pil
            if pil.mode != 'L':
                pil = pil.convert('L')
            image_buffer = pil.tobytes()
        elif self.type == TYPE_MATRIX_2D:
            # Just get the buffer
            image_buffer = self.matrix2d.transpose().tostring()
        elif self.type == TYPE_OPENCV2BW:
            # Just get the buffer
            image_buffer = self.opencv2bw.tostring()
        elif self.type == TYPE_OPENCV2:
            # Convert to gray then get buffer
            tmp = cv2.cvtColor(self.opencv2, cv2.cv.CV_BGR2GRAY)
            image_buffer = tmp.tostring()
        elif self.type == TYPE_MATRIX_RGB:
            # Convert to gray
            mat = self.matrix3d
            mat = LUMA[0]*mat[0] + LUMA[1]*mat[1] + LUMA[2]*mat[2]
            image_buffer = mat.transpose().tostring()
        elif self.type == TYPE_OPENCV:
            if self.channels == 1:
                # Just get buffer
                image_buffer = self.opencv.tostring()
            elif self.channels == 3:
                # Convert to gray
                w,h = self.width,self.height
                gray = cv.CreateImage((w,h),cv.IPL_DEPTH_8U,1)
                cv.CvtColor( self.opencv, gray, cv.CV_BGR2GRAY );
                image_buffer = gray.tostring()
            else:
                raise TypeError("Operation not supported for image type.")
        else:
            raise TypeError("Operation not supported for image type.")
        
        # Buffer should now be created
        assert image_buffer
        
        # Make sure the depth is correct
        if depth == self.depth:
            return image_buffer
        
        else:
            types = {8:numpy.uint8,32:numpy.float32,64:numpy.float64}
            
            # convert the image_buffer to data
            data = numpy.frombuffer(image_buffer,types[self.depth])
            
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
        image_buffer = None
        if self.type == TYPE_PIL:
            # Convert to rgb then get buffer
            pil = self.pil
            if pil.mode != 'RGB':
                pil = pil.convert('RGB')
            image_buffer = pil.tostring()
        elif self.type == TYPE_MATRIX_2D:
            # Convert to color
            mat = self.matrix2d.transpose()
            tmp = np.zeros((3,self.height,self.width),numpy.float32)
            tmp[0,:] = mat
            tmp[1,:] = mat
            tmp[2,:] = mat
            image_buffer = mat.tostring()
        elif self.type == TYPE_OPENCV2BW:
            # Convert to color
            tmp = cv2.cvtColor(self.opencv2bw, cv2.cv.CV_GRAY2RGB)
            image_buffer = tmp.tostring()
        elif self.type == TYPE_OPENCV2:
            # Convert BGR to RGB
            tmp = cv2.cvtColor(self.opencv2, cv2.cv.CV_BGR2RGB)
            image_buffer = tmp.tostring() 
        elif self.type == TYPE_MATRIX_RGB:
            # Just get buffer
            mat = self.matrix3d.transpose()
            image_buffer = mat.tostring()
        elif self.type == TYPE_OPENCV:
            # Convert color BGR to RGB
            w,h = self.width,self.height
            if self.channels == 3:
                rgb = cv.CreateImage((w,h),cv.IPL_DEPTH_8U,3)
                cv.CvtColor( self.opencv, rgb, cv.CV_BGR2RGB );
                image_buffer = rgb.tostring()
            elif self.channels == 1:
                rgb = cv.CreateImage((w,h),cv.IPL_DEPTH_8U,3)
                cv.CvtColor( self.opencv, rgb, cv.CV_GRAY2RGB );
                image_buffer = rgb.tostring()
            else:
                # Handle type errors
                raise TypeError("Operation not supported for image type.")
        else:
            # Handle unsupported
            raise TypeError("Operation not supported for image type.")
        
        assert image_buffer
        
        # Correct depth issues
        if depth == self.depth:
            return image_buffer
        
        else:
            types = {8:numpy.uint8,32:numpy.float32,64:numpy.float64}
            
            # convert the image_buffer to data
            data = numpy.frombuffer(image_buffer,types[self.depth])
            
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

    def thumbnail(self, newSize):
        ''' Returns a resized version of the image that fits in new_size but preserves the aspect ratio.

        @param newSize: tuple (new_width, new_height)
        @returns: a new pyvision image that is the resized version of this image.
        ''' 
        w,h = self.size
        s1 = float(newSize[0])/w
        s2 = float(newSize[1])/h
        s = min(s1,s2)
        return self.scale(s)

    
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
        In this case the downsampling quality may not be as good. #
        
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
        
        # Check the bounds for cropping
        if x < 0 or y < 0 or x+w > self.size[0] or y+h > self.size[1]:
            if size == None:
                size = (w,h)
            
            affine = pv.AffineFromRect(pv.Rect(x,y,w,h),size)
            im = affine(self)
            if return_affine:
                return im,affine
            else:
                return im
        
        # Get the image as opencv
        cvim = self.asOpenCV()
                
        # Set up ROI
        subim = cv.GetSubRect(cvim,(x,y,w,h))
        
        affine = pv.AffineTranslate(-x,-y,(w,h))
        
        if size == None:
            size = (w,h)
        
        # Copy to new image
        new_image = cv.CreateImage(size,cvim.depth,cvim.nChannels)        
        if interpolation == None:
            
            if size[0] < w or size[1] < y:
                # Downsampling so use area interpolation
                interpolation = cv.CV_INTER_AREA
            else:
                # Upsampling so use linear
                interpolation = cv.CV_INTER_CUBIC
            
        # Resize to the correct size
        cv.Resize(subim,new_image,interpolation)
        
        affine = pv.AffineNonUniformScale(float(size[0])/w,float(size[1])/h,size)*affine
        
        # Return the result as a pv.Image
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
            
            
    def show(self, window=None, pos=None, delay=0, size=None):
        '''
        Displays the annotated version of the image using OpenCV highgui
        @param window: the name of the highgui window to use, if one already exists by this name,
        or it will create a new highgui window with this name.
        @param pos: if a new window is being created, the (x,y) coordinate for the new window 
        @param delay: A delay in milliseconds to wait for keyboard input (passed to cv.WaitKey). 
            0 delays indefinitely, 30 is good for presenting a series of images like a video.
            For performance reasons, namely when using the same window to display successive 
            frames of video, we don't want to tear-down and re-create the window each time. 
            Thus the window frame will persist beyond the scope of the call to img.show(). The window 
            will disappear after the program exits, or it can be destroyed with a call to cv.DestroyWindow. 
        @param size: Optional output size for image, None=native size.
        @returns: the return value of the cv.WaitKey call.
        '''
        if window==None and pv.runningInNotebook() and 'pylab' in globals().keys():
            # If running in notebook, then try to display the image inline.
            
            if size == None:
                size = self.size
                
                # Constrain the size of the output
                max_dim = max(size[0],size[1])
                
                if max_dim > 800:
                    scale = 800.0/max_dim
                    size = (int(scale*size[0]),int(scale*size[1]))
            
            w,h = size
            
            # TODO: Cant quite figure out how figsize works and how to set it to native pixels
            #pylab.figure()
            IPython.core.pylabtools.figsize(1.25*w/72.0,1.25*h/72.0) #@UndefinedVariable
            pylab.figure()
            pylab.imshow(self.asAnnotated(),origin='upper',aspect='auto')
            
        else:
            # Otherwise, use an opencv window
            if window == None:
                window = "PyVisionImage"

            # Create the window
            cv.NamedWindow(window)
            
            # Set the location
            if pos != None:
                cv.MoveWindow(window, pos[0], pos[1])
            
            # Resize the image.    
            if size != None:
                x = pyvision.Image(self.asAnnotated().resize(size) )
            else:
                x = pyvision.Image(self.asAnnotated())    
                
            # Display the result
            cv.ShowImage(window, x.asOpenCV() )
            key = cv.WaitKey(delay=delay)
            del x
            return key
        
        
    def __repr__(self):
        
        return "pv.Image(w=%d,h=%d,c=%d,type=%s)"%(self.width,self.height,self.channels,self.type)


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
    
    # Check the size and channels
    assert cvmat.channels == 1
    r = cvmat.rows
    c = cvmat.cols
    
    # Convert to numpy
    a = np.fromstring(
             cvmat.tostring(),
             dtype=depth2dtype[cvmat.type],
             count=r*c)
    a.shape = (r,c)
    return a


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
  
    # Check the size
    assert len(a.shape) == 2
    r,c = a.shape
    
    # Convert to opencv
    cv_im = cv.CreateMat(r,c,dtype2depth[str(a.dtype)])
    cv.SetData(cv_im, a.tostring())
    return cv_im


        
        
        
        
