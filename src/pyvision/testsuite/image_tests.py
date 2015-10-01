'''
Copyright David S. Bolme

Created on Nov 5, 2010

@author: bolme
'''
import unittest

import pyvision as pv
import numpy as np
import os.path
import cv2

DATA_DIR = os.path.join(pv.__path__[0],'data','test')
SYNC_VIDEO = 'video_sync.mov'
SYNC_FRAMES = ['video_sync_0001.jpg', 'video_sync_0002.jpg', 'video_sync_0003.jpg', 'video_sync_0004.jpg', 'video_sync_0005.jpg',]

BUGS_VIDEO = os.path.join(pv.__path__[0],'data','test','BugsSample.m4v')
TAZ_VIDEO = os.path.join(pv.__path__[0],'data','test','TazSample.m4v')
TOYCAR_VIDEO = os.path.join(pv.__path__[0],'data','test','toy_car.m4v')

class TestImage(unittest.TestCase):
    
    def setUp(self):
        # Assume these work correctly
        self.im     = pv.Image(os.path.join(pv.__path__[0],"data","nonface","NONFACE_46.jpg"))
        self.pil    = self.im.asPIL()
        self.mat    = self.im.asMatrix2D()
        assert self.mat.shape[0] == 640
        assert self.mat.shape[1] == 480
        self.mat3d  = self.im.asMatrix3D()
        assert self.mat3d.shape[0] == 3
        assert self.mat3d.shape[1] == 640
        assert self.mat3d.shape[2] == 480
        self.opencv = self.im.asOpenCV2()
        self.opencv = self.im.asOpenCV2BW()
            
    def test_PILToBufferGray(self):
        w,h = self.im.size
        data_buffer = self.im.toBufferGray(8)
        self.assertEqual(len(data_buffer),w*h)
        data_buffer = self.im.toBufferGray(32)
        self.assertEqual(len(data_buffer),4*w*h)
        data_buffer = self.im.toBufferGray(64)
        self.assertEqual(len(data_buffer),8*w*h)

    def test_Matrix3DToBufferGray(self):
        im = pv.Image(self.mat3d)
        w,h = im.size
        data_buffer = im.toBufferGray(8)
        self.assertEqual(len(data_buffer),w*h)
        data_buffer = im.toBufferGray(32)
        self.assertEqual(len(data_buffer),4*w*h)
        data_buffer = im.toBufferGray(64)
        self.assertEqual(len(data_buffer),8*w*h)

    def test_Matrix2DToBufferGray(self):
        im = pv.Image(self.mat)
        w,h = im.size
        data_buffer = im.toBufferGray(8)
        self.assertEqual(len(data_buffer),w*h)
        data_buffer = im.toBufferGray(32)
        self.assertEqual(len(data_buffer),4*w*h)
        data_buffer = im.toBufferGray(64)
        self.assertEqual(len(data_buffer),8*w*h)

    def test_PILToMatrix2D(self):
        im = self.im
        pil = im.asPIL().convert('L')
        pil = pil.resize((180,120))
        im = pv.Image(pil)
        mat = im.asMatrix2D()
        for i in range(im.width):
            for j in range(im.height):
                self.assertAlmostEqual(pil.getpixel((i,j)),mat[i,j])
        
    def test_Matrix2DToPIL(self):
        im = pv.Image(self.mat[:180,:120])
        pil = im.asPIL()
        mat = im.asMatrix2D()
        for i in range(im.width):
            for j in range(im.height):
                self.assertAlmostEqual(pil.getpixel((i,j)),mat[i,j])

    def test_PILToMatrix3D(self):
        pil = self.im.asPIL().resize((180,120))
        im = pv.Image(pil)
        mat = im.asMatrix3D()
        for i in range(im.width):
            for j in range(im.height):
                for c in range(3):
                    self.assertAlmostEqual(pil.getpixel((i,j))[c],mat[c,i,j])

    def test_Matrix3D2PIL(self):
        im = pv.Image(self.mat3d[:,:180,:120])
        pil = self.im.asPIL()
        mat = im.asMatrix3D()
        for i in range(im.width):
            for j in range(im.height):
                for c in range(3):
                    self.assertAlmostEqual(pil.getpixel((i,j))[c],mat[c,i,j])
        
    def test_PILToOpenCV(self):
        pil = self.im.asPIL().resize((180,120))
        im = pv.Image(pil)
        cv = im.asOpenCV2()
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
        im = pv.Image(pil)
        cv = im.asOpenCV2()
        pil = pv.Image(cv).asPIL()

        for i in range(im.width):
            for j in range(im.height):
                for c in range(3):
                    self.assertAlmostEqual(pil.getpixel((i,j))[c],ord(cv.tostring()[i*3+j*im.width*3+2-c]))
        
    def test_OpenCVToPILGray(self):
        pil = self.im.asPIL().resize((180,120)).convert('L')
        im = pv.Image(pil)
        cv = im.asOpenCV2()
        im = pv.Image(cv)
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
        im = pv.Image(pil)
        cvim = im.asOpenCV2()
        data_buffer = im.toBufferRGB(8)

        for i in range(im.width):
            for j in range(im.height):
                for c in range(3):
                    self.assertAlmostEqual(ord(data_buffer[i*3+j*im.width*3+c]),ord(cvim.tostring()[i*3+j*im.width*3+2-c]))
     
    def test_asOpenCVBW(self):
        pass #TODO: Create tests for this method.
                
        
    def test_ConvertIPLImage32FToPvImage(self):
        im = pv.Image(pv.LENA)
        im = im.resize((512,400))
        cv_im = im.asOpenCV2()
        mat = im.asMatrix3D()
        
        for x in range(50):
            for y in range(50):
                for c in range(3):
                    self.assertAlmostEqual(cv_im[y,x,2-c],mat[c,x,y])
                    
                        
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
        im = pv.Image(pv.BABOON)
        hsv = im.asHSV()
        im = pv.Image(hsv)
        #im.show(delay=0)

        im = pv.Image(os.path.join(pv.__path__[0],"data","misc","baboon_bw.jpg"))




class TestVideo(unittest.TestCase):
    '''Tests for the video class.'''


    def testSync(self):
        """Video Sync Test"""
        # Tests a kludge that makes sure the first frame of video is read properly.
        
        # Uncomment next line to show image diagnostics
        ilog = None # pv.ImageLog()
        video_path = os.path.join(DATA_DIR,SYNC_VIDEO)
        video = pv.Video(video_path)
        
        frame_num = 0
        for frame_name in SYNC_FRAMES:
            frame_path = os.path.join(DATA_DIR,frame_name)
            ffmpeg_frame = pv.Image(frame_path)
            opencv_frame = video.next()
            #print ffmpeg_frame.asMatrix3D().shape
            #print opencv_frame.asMatrix3D().shape
            diff = ffmpeg_frame.asMatrix3D() - opencv_frame.asMatrix3D()
            diff_max = max(abs(diff.max()),abs(diff.min()))
            self.assert_(diff_max < 30.0) # Test on MacOS never exceeds 25
            diff = pv.Image(diff)
            if ilog != None:
                #print frame_name,diff_max
                ilog(ffmpeg_frame,"ffmpeg_%04d"%frame_num)
                ilog(opencv_frame,"opencv_%04d"%frame_num)
                ilog(diff,"diff_%04d"%frame_num)
            frame_num += 1

        # Make sure that this is the last frame of the video
        self.assertRaises(StopIteration, video.next)
            
        if ilog != None:
            ilog.show()
        
    def testVideoFrameCount(self):
        """Frame Count Test"""
        video_path = os.path.join(DATA_DIR,SYNC_VIDEO)
        video = pv.Video(video_path)
        
        count = 0
        for _ in video:
            #_.show(delay=0)
            count += 1
        
        self.assertEquals(count,5)
        
    


       
        
def test():
    unittest.main()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    test()
