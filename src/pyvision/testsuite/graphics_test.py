'''
Created on Nov 18, 2017

@author: bolme
'''
import unittest
import pyvision as pv
import time

class ImageTest(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testDisplay(self):
        im = pv.Image(pv.LENA)
        key = im.show(delay=5000)
        print 'Pressed Key:',key

        im = pv.Image(pv.BABOON)
        key = im.show(delay=5000)
        print 'Pressed Key:',key
        

    def dtestMemory(self):
        for i in range(100):
            im = pv.Image(pv.LENA)
            key = im.show(delay=250)
    
            im = pv.Image(pv.BABOON)
            key = im.show(delay=250)
        

    def testSpeed(self):
        bab = pv.Image(pv.BABOON)
        lena = pv.Image(pv.LENA)
        start = time.time()
        for i in range(20):
            key = lena.show(delay=1)
            key = bab.show(delay=1)
        finish = time.time()
        print "FPS:",40.0/(finish-start)


class VideoTest(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testName(self):
        pass


class CaptureTest(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testName(self):
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()