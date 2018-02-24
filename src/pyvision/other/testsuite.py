'''
Created on Jul 1, 2009

This module contains tests for the face recognition algorithms.

@author: bolme
'''

import unittest

import pyvision as pv
import numpy as np
#from optic_flow import *
#from distance import *
#import cv

import os.path

class _TestNormalize(unittest.TestCase):
    
    def setUp(self):
        # Eye coordinates generated automatically
        #leye = pv.Point(250.336538,174.074519) 
        #reye = pv.Point(343.828125,180.042067)
        
        fname = os.path.join(pv.__path__[0],'data','misc','lena.jpg')
        im = pv.Image(fname,bw_annotate=True)

        #affine = pv.AffineFromPoints(leye,reye,pv.Point(48.0,64.0),pv.Point(144.0,64.0),(192,192))

        self.tile = im
    
    def test_1_meanStd(self):
        '''meanStd Normalization: norm.mean() = 0.0 and norm.std() = 1.0....'''
        ilog = None
        if 'ilog' in globals().keys():
            ilog = globals()['ilog']
            
        norm = pv.meanStd(self.tile)
        
        if ilog != None:
            ilog.log(norm,label="meanStd_Normalization")
            
        mat = norm.asMatrix2D()
        self.assertAlmostEqual(mat.mean(),0.0,places=3)
        self.assertAlmostEqual(mat.std(),1.0,places=3)

    def test_2_meanUnit(self):
        '''meanUnit Normalization: norm.mean() = 0.0  and ||norm|| = 1.0....'''
        ilog = None
        if 'ilog' in globals().keys():
            ilog = globals()['ilog']
            
        norm = pv.meanUnit(self.tile)
        
        if ilog != None:
            ilog.log(norm,label="meanUnit_Normalization")
            
        mat = norm.asMatrix2D()
        self.assertAlmostEqual(mat.mean(),0.0)
        length = np.sqrt((mat**2).sum())
        self.assertAlmostEqual(length,1.0,places=4)

    def test_3_unit(self):
        '''unit Normalization: ||norm|| = 1.0 and dot(norm,im)/||im|| = 1.0.'''
        ilog = None
        if 'ilog' in globals().keys():
            ilog = globals()['ilog']
            
        norm = pv.unit(self.tile)
        
        if ilog != None:
            ilog.log(norm,label="unit_Normalization")
            
        mat = norm.asMatrix2D()
        length = np.sqrt((mat**2).sum())
        self.assertAlmostEqual(length,1.0,places=3)

        mat = norm.asMatrix2D()
        mat = mat.flatten()
        im = self.tile.asMatrix2D().flatten()
        proj = np.dot(mat,im)
        length = np.sqrt((im**2).sum())
        self.assertAlmostEqual(proj/length,1.0,places=3)

    def test_4_bandPass(self):
        '''bandPassFilter Normalization: ...................................'''
        ilog = None
        if 'ilog' in globals().keys():
            ilog = globals()['ilog']
            
        norm = pv.bandPassFilter(self.tile,10.0,4.0)
        
        if ilog != None:
            ilog.log(norm,label="bandPass_Normalization")
            
        mat = norm.asMatrix2D()
        self.assertAlmostEqual(mat.mean(),0.0,places=4)
        self.assertAlmostEqual(mat.std(),12.090113839874826,delta=0.01)

    def test_5_lowPass(self):
        '''lowPassFilter Normalization: ....................................'''
        ilog = None
        if 'ilog' in globals().keys():
            ilog = globals()['ilog']
            
        norm = pv.lowPassFilter(self.tile,10.0)
        
        if ilog != None:
            ilog.log(norm,label="lowPass_Normalization")
            
        mat = norm.asMatrix2D()
        self.assertAlmostEqual(mat.mean(),123.69997406005859,delta=0.01)
        self.assertAlmostEqual(mat.std(),36.886999835117216,delta=0.01)

    def test_6_highPass(self):
        '''highPassFilter Normalization: ...................................'''
        ilog = None
        if 'ilog' in globals().keys():
            ilog = globals()['ilog']
            
        norm = pv.highPassFilter(self.tile,10.0)
        
        if ilog != None:
            ilog.log(norm,label="highPass_Normalization")
            
        mat = norm.asMatrix2D()
        self.assertAlmostEqual(mat.mean(),0.0,delta=0.001)
        self.assertAlmostEqual(mat.std(),22.936873341661158,delta=0.01)

    def test_7_veryHighPass(self):
        '''highPassFilter Normalization: sigma = 1.5........................'''
        ilog = None
        if 'ilog' in globals().keys():
            ilog = globals()['ilog']
            
        # This setting corsponds to the default gaussian in selfQuotient
        norm = pv.highPassFilter(self.tile,1.5)
        
        if ilog != None:
            ilog.log(norm,label="veryHighPass_Normalization")
            
        mat = norm.asMatrix2D()
        self.assertAlmostEqual(mat.mean(),0.0,delta=4)
        self.assertAlmostEqual(mat.std(),8.0027218003238687,delta=0.01)

    def test_8_selfQuotient(self):
        '''selfQuotient Normalization: .....................................'''
        ilog = None
        if 'ilog' in globals().keys():
            ilog = globals()['ilog']
            
        norm = pv.selfQuotientImage(self.tile)
        
        if ilog != None:
            ilog.log(norm,label="selfQuotient_Normalization")
            
        mat = norm.asMatrix2D()
        self.assertAlmostEqual(mat.mean(),0.98861616849899292,delta=0.001)
        self.assertAlmostEqual(mat.std(),0.1647989569275968,delta=0.001)


         
                    
class _TestDistance(unittest.TestCase):
    
    def setUp(self):
        '''Initialize the tests'''

    
    def test_1_bool2Ubyte(self):
        '''distance::boolToUbyte ...........................................'''
        a = np.random.randint(2,size=16) > 0
        b = pv.boolToUbyte(a)
        c = pv.ubyteToBool(b)
        d = pv.boolToUbyte(c)

        self.assert_((a == c).sum() == 16)
        self.assert_((b == d).sum() == 2)
        
        a = np.random.randint(2,size=5000) > 0
        b = pv.boolToUbyte(a)
        c = pv.ubyteToBool(b)
        d = pv.boolToUbyte(c)

        self.assert_((a == c).sum() == 5000)
        self.assert_((b == d).sum() == 625)
        
        
        

    def test_2_hamming(self):
        '''distance::hamming 1..............................................'''
        a = np.random.randint(2,size=16) > 0
        b = np.random.randint(2,size=16) > 0
        
        bin_hamming = pv.hamming(a,b)

        a = pv.boolToUbyte(a)
        b = pv.boolToUbyte(b)
        
        byte_hamming = pv.hamming(a,b)
        
        self.assertEquals(bin_hamming,byte_hamming)
        

    def test_3_hamming(self):
        '''distance::hamming 2..............................................'''
        a = np.random.randint(2,size=1769472) > 0
        b = np.random.randint(2,size=1769472) > 0
        
        bin_hamming = pv.hamming(a,b)

        a = pv.boolToUbyte(a)
        b = pv.boolToUbyte(b)
        
        byte_hamming = pv.hamming(a,b)
        
        self.assertEquals(bin_hamming,byte_hamming)
        
        
            
         
                    
def test():
    '''Run the face test suite.'''
    pv.disableCommercialUseWarnings()
    
    normalize_suite = unittest.TestLoader().loadTestsFromTestCase(_TestNormalize)
    surf_suite = unittest.TestLoader().loadTestsFromTestCase(_TestSURF)
    dist_suite = unittest.TestLoader().loadTestsFromTestCase(_TestDistance)
    
    
    
    test_suites = [
                   normalize_suite,
                   surf_suite,
                   dist_suite
                   ]
    
    
    pyvision_suite = unittest.TestSuite(test_suites)
    
    unittest.TextTestRunner(verbosity=2).run(pyvision_suite)

    
if __name__ == '__main__':
    # By default run the test suite
    unittest.main(testRunner = unittest.TextTestRunner(verbosity=2))
    
    
    