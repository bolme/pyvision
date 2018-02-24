'''
Copyright David S. Bolme

Created on May 29, 2011

@author: bolme
'''
import unittest

import pyvision as pv
import pyvision.features.v1like as v1

class V1LikeTest(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testV1LikeShape(self):
        im = pv.Image(pv.BABOON)
        
        im = im.resize((128,128))
        
        v1like = v1.V1Processing(v1.V1LIKE_PARAMS_A,v1.V1LIKE_FEATURES_A)
        tmp = v1like.extractFromImage(im)
        
        print(tmp.shape)
        self.assertEqual(tmp.shape, (30,30,96))
        
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()