'''
Copyright David S. Bolme

Created on Nov 19, 2010

@author: bolme
'''
import unittest
import os.path
import pyvision as pv

BUGS_VIDEO = os.path.join(pv.__path__[0],'data','test','BugsSample.m4v')
TAZ_VIDEO = os.path.join(pv.__path__[0],'data','test','TazSample.m4v')
TOYCAR_VIDEO = os.path.join(pv.__path__[0],'data','test','toy_car.m4v')

class MotionDetectTest(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testMotionDetectAMF(self):
        ilog = None # pv.ImageLog()
        
        md = pv.MotionDetector(method=pv.BG_SUBTRACT_AMF,minArea=200)
        
        video = pv.Video(BUGS_VIDEO)
        
        i = 0
        for frame in video:            
            count = md.detect(frame)

            rects = md.getStandardizedRects()
            boxes = md.getBoundingRects()
            
            polygons = md.getPolygons(return_all=True)
            
            if ilog != None:
                print "Processing Frame:",i
                
                key_frame = md.getKeyFrame()
                
                md.annotateFrame(key_frame)
                
                if key_frame != None:
                    ilog(key_frame,format='jpg')
                            
            i += 1
            if i > 20: break
        
        if ilog != None:
            ilog.show()


    def testMotionDetectMF(self):
        ilog =  None # pv.ImageLog()
        
        md = pv.MotionDetector(method=pv.BG_SUBTRACT_MF,minArea=200)
        
        video = pv.Video(BUGS_VIDEO)
        
        i = 0
        for frame in video:            
            count = md.detect(frame)

            rects = md.getStandardizedRects()
            boxes = md.getBoundingRects()
            
            polygons = md.getPolygons(return_all=True)
            
            if ilog != None:
                print "Processing Frame:",i
                
                key_frame = md.getKeyFrame()
                
                md.annotateFrame(key_frame)
                
                if key_frame != None:
                    ilog(key_frame,format='jpg')
                            
            i += 1
            if i > 20: break
        
        if ilog != None:
            ilog.show()



    def testMotionDetectFD(self):
        ilog = None # pv.ImageLog()
        
        md = pv.MotionDetector(method=pv.BG_SUBTRACT_FD,minArea=200)
        
        video = pv.Video(BUGS_VIDEO)
        
        i = 0
        for frame in video:            
            count = md.detect(frame)

            rects = md.getStandardizedRects()
            boxes = md.getBoundingRects()
            
            polygons = md.getPolygons(return_all=True)
            
            if ilog != None:
                print "Processing Frame:",i
                
                key_frame = md.getKeyFrame()
                
                md.annotateFrame(key_frame)
                
                if key_frame != None:
                    ilog(key_frame,format='jpg')
                            
            i += 1
            if i > 20: break
        
        if ilog != None:
            ilog.show()

    def testMotionDetectMCFD(self):
        ilog =  pv.ImageLog()
        
        flow = pv.OpticalFlow()
        md = pv.MotionDetector(method=pv.BG_SUBTRACT_MCFD,minArea=200,rect_type=pv.STANDARDIZED_RECTS)
        video = pv.Video(TOYCAR_VIDEO)
        
        i = 0
        for frame in video:    

            flow.update(frame)
            md.detect(frame)
            
            if ilog != None:
                print "Processing Frame:",i
                flow.annotateFrame(frame)

                key_frame = md.getKeyFrame()
                
                md.annotateFrame(key_frame)
                
                if key_frame != None:
                    ilog(key_frame,format='jpg')
                            
            i += 1
            #if i > 20: break
        
        if ilog != None:
            ilog.show()
            



class OpticalFlowTest(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testOpticalFlow(self):
        ilog = None # pv.ImageLog()
        
        flow = pv.OpticalFlow()
        
        video = pv.Video(TAZ_VIDEO)
        
        i = 0
        for frame in video:          
            flow.update(frame)
            
            flow.annotateFrame(frame)
            if ilog != None:
                print "Processing Frame:",i  
                ilog(frame,format='jpg')

            i += 1
            if i > 10: break
        
        if ilog != None:
            ilog.show()
            
    def testHomographies(self):
        ilog =  None # pv.ImageLog()
        
        flow = pv.OpticalFlow()
        
        video = pv.Video(TAZ_VIDEO)
        
        i = 0
        prev_frame = None
        for frame in video:  
                 
            flow.update(frame)
            
            flow.annotateFrame(frame)
            if ilog != None:
                print "Processing Frame:",i  
                if hasattr(frame,'to_prev'):
                    prev = frame.to_prev(frame)
                    ilog(prev,'back',format='jpg')
                    
                if prev_frame != None:
                    next = prev_frame.to_next(prev_frame)
                    ilog(next,'forward',format='jpg')
                
                ilog(frame,"current",format='jpg')

            i += 1
            if i > 10: break
            
            prev_frame = frame
        
        if ilog != None:
            ilog.show()
                        
            

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main(testRunner = unittest.TextTestRunner(verbosity=2))