'''
Created on Jul 1, 2009

This module contains tests for the face recognition algorithms.

@author: bolme
'''

import unittest

import pyvision as pv
import numpy as np
pv.disableCommercialUseWarnings()

from pyvision.analysis.FaceAnalysis.FaceDatabase     import ScrapShotsDatabase
from pyvision.analysis.FaceAnalysis.EyeDetectionTest import EyeDetectionTest
from pyvision.face.CascadeDetector                   import CascadeDetector
from pyvision.face.FilterEyeLocator                  import FilterEyeLocator

from pyvision.analysis.roc                           import ROC

class TestFilterEyeLocator(unittest.TestCase):
    
    def test_ASEFEyeLocalization(self):
        '''FilterEyeLocator: Scrapshots Both10 rate == 0.4800...............'''
        ilog = None
        if 'ilog' in globals().keys():
            ilog = globals()['ilog']
            
        # Load a face database
        ssdb = ScrapShotsDatabase()
                
        # Create a face detector 
        face_detector = CascadeDetector()

        # Create an eye locator
        eye_locator = FilterEyeLocator()
        
        # Create an eye detection test
        edt = EyeDetectionTest(name='asef_scraps')

        #print "Testing..."
        for face_id in ssdb.keys()[:25]:
            face = ssdb[face_id]
            im = face.image
            
            dist = face.left_eye.l2(face.right_eye)
            dist = np.ceil(0.1*dist)
            im.annotateCircle(face.left_eye,radius=dist,color='white')
            im.annotateCircle(face.right_eye,radius=dist,color='white')

            # Detect the faces
            faces = face_detector.detect(im)
                            
            # Detect the eyes
            pred_eyes = eye_locator(im,faces)
            for rect,leye,reye in pred_eyes:
                im.annotateRect(rect)
                im.annotateCircle(leye,radius=1,color='red')
                im.annotateCircle(reye,radius=1,color='red')
                
            
            truth_eyes = [[face.left_eye,face.right_eye]]
            
            pred_eyes = [ [leye,reye] for rect,leye,reye in pred_eyes]
            
            # Add to eye detection test
            edt.addSample(truth_eyes, pred_eyes, im=im, annotate=True)
            if ilog != None:
                ilog.log(im,label='test_ASEFEyeLocalization')
                
        edt.createSummary()
        
        # Very poor accuracy on the scrapshots database
        self.assertAlmostEqual( edt.face_rate ,   1.0000, places = 3 )
        self.assertAlmostEqual( edt.both25_rate , 0.8800, places = 3 )
        self.assertAlmostEqual( edt.both10_rate , 0.5200, places = 3 )
        self.assertAlmostEqual( edt.both05_rate , 0.2800, places = 3 )
        



def test():
    '''Run the face test suite.'''
    pv.disableCommercialUseWarnings()
    
    fel_suite = unittest.TestLoader().loadTestsFromTestCase(TestFilterEyeLocator)
    
    
    
    test_suites = [
                   fel_suite,
                   ]
    
    pyvision_suite = unittest.TestSuite(test_suites)
    
    unittest.TextTestRunner(verbosity=2).run(pyvision_suite)

    
if __name__ == '__main__':
    # By default run the test suite
    #ilog = pv.ImageLog()
    test()
    #ilog.show()
    
    
    