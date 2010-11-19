'''
Copyright David S. Bolme

Created on Nov 19, 2010

@author: bolme
'''
import unittest
import os.path

import pyvision as pv

BUGS_VIDEO = os.path.join(pv.__path__[0],'data','test','BugsSample.m4v')

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testName(self):
        '''Test motion detection on the bugs video.'''
        ilog = pv.ImageLog()
        
        md = pv.MotionDetector(minArea=200)
        
        video = pv.Video(BUGS_VIDEO)
        
        for frame in video:
            print "frame",frame.size
            
            count = md.detect(frame)
            #print count
            
            #if count >= 0:
            key_frame = md.getKeyFrame()
            rects = md.getStandardizedRects()
            boxes = md.getBoundingRects()
            
            polygons = md.getPolygons(return_all=True)
            
            if key_frame != None:
                for poly in polygons:
                    key_frame.annotatePolygon(poly,color='#00FF00',width=1)
                    
                for rect in boxes:
                    key_frame.annotatePolygon(rect.asPolygon(),width=1,color='yellow')
                    
                #for rect in rects:
                #    key_frame.annotatePolygon(rect.asPolygon(),width=2)
                #    key_frame.annotatePoint(rect.center())
                    
                key_frame.show("daves")
                md.getAnnotatedImage(showContours=True).show("steves")
            
            #print polygons
                
            
            if ilog != None:
                if key_frame != None:
                    ilog(key_frame)
                    #anImage = md.getAnnotatedImage(showContours=True)
                    #anImage.show('frame')
                    #ilog(anImage,format='jpg')
        
        if ilog != None:
            ilog.show()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()