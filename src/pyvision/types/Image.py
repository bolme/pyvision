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
import numpy
try:
    import opencv
except:
    print "Warning: Could not import opencv."
import unittest
import os.path


import pyvision

#Image Types - these are subject to changes
TYPE_MATRIX_2D  = "TYPE_MATRIX2D" # grayscale
TYPE_MATRIX_RGB = "TYPE_MATRIX_RGB" # RGB or RGBA
TYPE_PIL        = "TYPE_PIL" # PIL image
TYPE_OPENCV     = "TYPE_OPENCV" # OpenCV image

LUMA = [0.299, 0.587, 0.114, 1.0]

class Image:
    '''
    This is a unified image class that is able to represent images in
    different formats including numpy matrix, PIL, OpenCV.  This also
    allows some simple operations on the image such as anotation.
    
    When working with images in matrix format, they are transposed such
    that x = col and y = row.  You can therefore still work with coords
    such that im[x,y] = mat[x,y].
    
    Images have the following attributes:
    width = width of the image
    height = height of the image
    size = (width,height)
    channels = number of channels: 1(gray), 3(RGB)
    depth = bitdepth: 8(uchar), 32(float), 64(double)
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
            else:
                raise TypeError("Unsuppoted format for PIL images: %s"%self.pil.mode)
            
            self.depth = 8
                        
        elif isinstance(data,opencv.cv.CvMat):
            self.type=TYPE_OPENCV
            self.opencv=data 
            
            self.width = data.width
            self.height = data.height
            
            assert data.nChannels in (1,3)
            self.channels = data.nChannels 
            
            assert data.depth in (8,)
            self.depth = data.depth   

        else:
            raise TypeError("Could not create from type: %s"%type(data))
        
        self.size = (self.width,self.height)
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
        '''TODO: Deprecated remove this sometime.'''
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
        Create a color opencv represenation of the image.
        TODO: The OpenCV databuffer seems to be automatically swapped from RGB
              to BGR.  This is counter inuitive.
        '''
        
        w,h = self.size
        if self.channels == 1:
            gray = opencv.cvCreateImage(opencv.cvSize(w,h),opencv.IPL_DEPTH_8U,1)
            gray.imageData = self.toBufferGray(8)
            self.opencv = gray
        elif self.channels == 3:
            rgb = opencv.cvCreateImage(opencv.cvSize(w,h),opencv.IPL_DEPTH_8U,3)
            #bgr = opencv.cvCreateImage(opencv.cvSize(w,h),opencv.IPL_DEPTH_8U,3)
            rgb.imageData = self.toBufferRGB(8)
            # convert from RGB to BGR
            #opencv.cvCvtColor(rgb,bgr,opencv.CV_RGB2BGR)
            self.opencv=rgb
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
                buffer = self.opencv.imageData
            elif self.channels == 3:
                w,h = self.width,self.height
                gray = opencv.cvCreateImage(opencv.cvSize(w,h),opencv.IPL_DEPTH_8U,1)
                opencv.cvCvtColor( self.opencv, gray, CV_BGR2GRAY );
                buffer = gray.imageData
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
            tmp = zeros((3,self.height,self.width),numpy.float32)
            tmp[0,:] = mat
            tmp[1,:] = mat
            tmp[2,:] = mat
            buffer = mat.tostring()            
        elif self.type == TYPE_MATRIX_RGB:
            mat = self.matrix3d
            mat = LUMA[0]*mat[0] + LUMA[1]*mat[1] + LUMA[2]*mat[2]
            buffer = mat.transpose().tostring()
        elif self.type == TYPE_OPENCV:
            w,h = self.width,self.height
            if self.channels == 3:
                rgb = opencv.cvCreateImage(opencv.cvSize(w,h),opencv.IPL_DEPTH_8U,3)
                opencv.cvCvtColor( self.opencv, rgb, opencv.CV_BGR2RGB );
                buffer = rgb.imageData
            elif self.channels == 1:
                rgb = opencv.cvCreateImage(opencv.cvSize(w,h),opencv.IPL_DEPTH_8U,3)
                opencv.cvCvtColor( self.opencv, rgb, opencv.CV_GRAY2RGB );
                buffer = rgb.imageData
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
        '''
        
    def save(self,filename):
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
            self.asPIL().save(filename)
            
    def show(self):
        '''
        Shows the annotated versoin of the image.
        '''
        self.asAnnotated().show()
            
class _TestImage(unittest.TestCase):
    
    def setUp(self):
        # Assume these work correctly
        self.im     = Image(os.path.join(pyvision.__path__[0],"data","nonface","NONFACE_46.jpg"))
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
                    self.assertAlmostEqual(pil.getpixel((i,j))[c],ord(cv.imageData[i*3+j*im.width*3+2-c]))
        
    def test_OpenCVToPIL(self):
        pil = self.im.asPIL().resize((180,120))
        im = Image(pil)
        cv = im.asOpenCV()
        pil = Image(cv).asPIL()

        for i in range(im.width):
            for j in range(im.height):
                for c in range(3):
                    self.assertAlmostEqual(pil.getpixel((i,j))[c],ord(cv.imageData[i*3+j*im.width*3+2-c]))
        
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
        cv = im.asOpenCV()
        buffer = im.toBufferRGB(8)

        for i in range(im.width):
            for j in range(im.height):
                for c in range(3):
                    self.assertAlmostEqual(ord(buffer[i*3+j*im.width*3+c]),ord(cv.imageData[i*3+j*im.width*3+2-c]))
        
        
        

