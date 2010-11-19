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
            polygons = md.asPolygons()
            
            if key_frame != None:
                for poly in polygons:
                    key_frame.annotatePolygon(poly,width=2)
                    box,area,center = pv.polygonStats(poly)
                    key_frame.annotatePoint(center)
                key_frame.show()
            
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