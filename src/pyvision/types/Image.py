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

import PIL.ImageDraw
import PIL.Image
from numpy import array,ndarray
import numpy
import opencv
from  opencv.adaptors import PIL2Ipl,Ipl2PIL,Ipl2NumPy,NumPy2Ipl
import unittest
import os.path


import pyvision
#Image Types - these are subject to changes
TYPE_MATRIX2D = "TYPE_MATRIX2D" # grayscale
TYPE_PIL      = "TYPE_PIL" # PIL image
TYPE_OPENCV   = "TYPE_OPENCV" # OpenCV image


class Image:
    '''
    This is a unified image class that is able to represent images in
    different formats including numpy matrix, PIL, OpenCV.  This also
    allows some simple operations on the image such as anotation.
    '''

    #------------------------------------------------------------------------
    def __init__(self,data,bw_annotate=False):
        self.filename = None
        self.pil = None
        self.matrix2d = None
        self.matrix3d = None
        self.opencv = None
        self.annotated = None
        self.bw_annotate = bw_annotate
        
        if isinstance(data,ndarray) and len(data.shape) == 2:
            self.type=TYPE_MATRIX2D
            self.matrix2d = data
        elif isinstance(data,PIL.Image.Image):
            self.type=TYPE_PIL
            self.pil = data
        elif type(data) == str:
            self.type=TYPE_PIL
            self.pil = PIL.Image.open(data)
            self.filename = data
        elif isinstance(data,opencv.cv.CvMat):
            self.type=TYPE_OPENCV
            self.opencv=data     
            #raise TypeError("OpenCV not supported yet: %s"%type(data))            
        else:
            raise TypeError("Could not create from type: %s"%type(data))
        self.data = data
        
    
    
    #########################################################################
    # The following fuctions are used to obtain different representations 
    # of an image.
    #########################################################################
    
    #------------------------------------------------------------------------
    def asMatrix2D(self):
        '''
        Return a 2D grayscale matrix of this image.
        '''
        if self.matrix2d == None:
            self._generateMatrix2D()
        return self.matrix2d

    #------------------------------------------------------------------------
    def asMatrix3D(self):
        '''
        Return a 2D grayscale matrix of this image.
        '''
        if self.matrix3d == None:
            self._generateMatrix3D()
        return self.matrix3d

    #------------------------------------------------------------------------
    def asPIL(self):
        '''
        Return a 2D grayscale matrix of this image.
        '''
        if self.pil == None:
            self._generatePIL()
        return self.pil

    #------------------------------------------------------------------------
    def asOpenCV(self):
        if self.opencv == None:
            self._generateOpenCV()
        return self.opencv
        
    #------------------------------------------------------------------------
    def asPyramid(self):
        '''
        Return a 2D grayscale pyramids of matrices.
        '''
        # TODO: asPyramid
        raise NotImplementedError()


    #------------------------------------------------------------------------
    def asAnnotated(self):
        '''
        Returns the annotated version of this image.
        '''
        if self.annotated == None:
            if self.bw_annotate:
                # Make a black and white image that can be annotated with color.
                self.annotated = self.asPIL().convert("L").copy().convert("RGB")
            else:
                # Annotate over color if avalible.
                self.annotated = self.asPIL().copy().convert("RGB")
        return self.annotated
            
    #------------------------------------------------------------------------
    def annotateRect(self,rect,color='red'):
        '''
        Render a rectangle on the annotated version of the image.
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        box = [rect.x,rect.y,rect.x+rect.w,rect.y+rect.h]
        draw.rectangle(box,outline=color)
        del draw

    #------------------------------------------------------------------------
    def annotateEllipse(self,rect,color='red'):
        '''
        Render an ellipse on the annotated image.
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        box = [rect.x,rect.y,rect.x+rect.w,rect.y+rect.h]
        draw.ellipse(box,outline=color)
        del draw
                
    #------------------------------------------------------------------------
    def annotateLine(self,point1,point2,color='red'):
        '''
        Render an ellipse on the annotated image.
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        line = [point1.X(),point1.Y(),point2.X(),point2.Y()]
        draw.line(line,fill=color,width=1)
        del draw
        
    #------------------------------------------------------------------------
    def annotatePoint(self,point,color='red'):
        '''
        Mark a point on the annotated image.
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        #line = [point.X()+3,point.Y(),point.X()+6,point.Y()]
        #draw.line(line,fill=color,width=1)
        #line = [point.X()-3,point.Y(),point.X()-6,point.Y()]
        #draw.line(line,fill=color,width=1)
        #line = [point.X(),point.Y()+3,point.X(),point.Y()+6]
        #draw.line(line,fill=color,width=1)
        #line = [point.X(),point.Y()-3,point.X(),point.Y()-6]
        #draw.line(line,fill=color,width=1)
        box = [point.X()-3,point.Y()-3,point.X()+3,point.Y()+3]
        draw.ellipse(box,outline=color)
        del draw

    #------------------------------------------------------------------------
    def annotateCircle(self,point, radius=3, color='red'):
        '''
        Mark a point on the annotated image.
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        #line = [point.X()+3,point.Y(),point.X()+6,point.Y()]
        #draw.line(line,fill=color,width=1)
        #line = [point.X()-3,point.Y(),point.X()-6,point.Y()]
        #draw.line(line,fill=color,width=1)
        #line = [point.X(),point.Y()+3,point.X(),point.Y()+6]
        #draw.line(line,fill=color,width=1)
        #line = [point.X(),point.Y()-3,point.X(),point.Y()-6]
        #draw.line(line,fill=color,width=1)
        box = [point.X()-radius,point.Y()-radius,point.X()+radius,point.Y()+radius]
        draw.ellipse(box,outline=color)
        del draw
        
    #------------------------------------------------------------------------
    def annotateLabel(self,point,label,color='red'):
        '''
        Render text on the annotated image.
        '''
        # TODO: annotateLabel
        raise NotImplementedError()

        
    #------------------------------------------------------------------------
    def annotateDot(self,point,color='red'):
        '''
        Render an ellipse on the annotated image.
        '''
        im = self.asAnnotated()
        draw = PIL.ImageDraw.Draw(im)
        draw.point([point.X(),point.Y()],fill=color)
        del draw
        
    #------------------------------------------------------------------------
    def valueNormalize(self):
        print "WARNING: Image.valueNormalize has been depricated."
        return self.normalize()

    #------------------------------------------------------------------------
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
       

    #------------------------------------------------------------------------        
    def _generateMatrix2D(self):
        '''
        Create a matrix version of the image.
        '''
        im = self.asPIL()
        if self.type == TYPE_PIL:
            im = im.convert('L')
            w,h = im.size
            data = list(im.getdata())
            self.matrix2d = numpy.array(data,numpy.float32).reshape((h,w)).transpose()
        elif self.type == TYPE_OPENCV:
            if self.opencv.nChannels == 3:
                gray = opencv.cvCreateImage( opencv.cvGetSize(self.opencv), 8, 1 );
                opencv.cvCvtColor( self.opencv, gray, opencv.CV_BGR2GRAY );    
                self.matrix2d = numpy.array(Ipl2NumPy(gray),'f')
            if self.opencv.nChannels == 1:
                self.matrix2d = numpy.array(Ipl2NumPy(self.opencv),'f')
        else:
            raise NotImplementedError("Cannot convert image from type: %s"%self.type)
            

    def _generateMatrix3D(self):
        '''
        Create a matrix version of the image.
        '''
        im = self.asPIL()
        if self.type == TYPE_PIL:
            im = im.convert('RGBA')
            w,h = im.size
            data = list(im.getdata())
            self.matrix3d = numpy.array(data,numpy.float32).reshape((h,w,4)).transpose()
        else:
            raise NotImplementedError("Cannot convert image from type: %s"%self.type)
            

    def _generatePIL(self):
        '''
        Create a PIL version of the image
        '''
        if self.type == TYPE_MATRIX2D:
            mat = self.matrix2d.transpose()
            h,w = mat.shape
            max_value = mat.max()
            min_value = mat.min()
            if max_value == min_value:
                max_value = min_value+1
            mat = (255.0/(max_value-min_value))*(mat-min_value)
            data = array(mat.flatten(),'i')
            im = PIL.Image.new('L',(w,h))
            im.putdata(data)
            self.pil = im
        elif self.type == TYPE_OPENCV:
            self.pil = Ipl2PIL(self.opencv)
        else:
            raise NotImplementedError("Cannot convert image from type: %s"%self.type)
        
    def _generateOpenCV(self):
        '''
        Create an opencv represenation of the image
        '''
        im = self.asPIL()
        self.opencv = PIL2Ipl(im)
        
            
    def save(self,filename):
        '''
        Save the image to a file.
        '''
        if filename[-4:] == ".raw":
            # TODO: save as a matrix
            raise NotImplementedError("Cannot save as a matrix")
        elif filename[-4:] == ".mat":
            # TODO: save as a matlab file
            raise NotImplementedError("Cannot save in matlab format")
        else:
            self.asPIL().save(filename)
            
    def show(self):
        '''
        Shows the annotated versoin of the image.
        '''
        self.asAnnotated().show()
            
class _TestImage(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_image3d(self):
        img = Image(os.path.join(pyvision.__path__[0],"data","nonface","NONFACE_46.jpg"))
        mat3d = img.asMatrix3D()
        self.assert_( mat3d.shape == (4,640,480) )
        
        #Image(mat3d[0,:,:]).show()
        #Image(mat3d[1,:,:]).show()
        #Image(mat3d[2,:,:]).show()
        #Image(mat3d[3,:,:]).show()

