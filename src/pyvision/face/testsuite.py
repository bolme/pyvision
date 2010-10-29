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
#from pyvision.face.Eigenfaces                        import Eigenfaces

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
        self.assertAlmostEqual( edt.face_rate ,   0.9200, places = 3 )
        self.assertAlmostEqual( edt.both25_rate , 0.8000, places = 3 )
        self.assertAlmostEqual( edt.both10_rate , 0.4800, places = 3 )
        self.assertAlmostEqual( edt.both05_rate , 0.4000, places = 3 )
        

class TestEigenfaces(unittest.TestCase):
    def setUp(self):
        self.scraps_ids = ['00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '00009', '00010', '00011', '00012', '00013', '00014', '00015', '00016', '00017', '00018', '00019', '00020', '00021', '00022', '00023', '00024', '00025', '00026', '00027', '00028', '00029', '00030']
        self.scraps_training = self.scraps_ids[:15]
        self.scraps_testing = self.scraps_ids[15:]

    def trainScraps(self,whiten=True,ilog=None):
        # Load a face database
        ssdb = ScrapShotsDatabase()
        keys = list(ssdb.keys())
        keys.sort()
        
        # Create an eigenfaces instance
        ef = Eigenfaces(whiten=whiten)

        for face_id in keys:
            #print "Processing:",face_id
            face = ssdb[face_id]
            if face.person_id not in self.scraps_training:
                continue

            ef.addTraining(face.image,face.face_rect,face.left_eye,face.right_eye,ilog=ilog)
        
        ef.train(ilog=ilog)
        return ef
        
    def test_1_EigenFaceTrain(self):
        '''Eigenfaces: Training Checks......................................'''
        ilog = None
        if 'ilog' in globals().keys():
            ilog = globals()['ilog']
        
        ef = self.trainScraps(ilog=ilog)
        
        self.assertEqual(ef.eigenvalues.shape[0],53)
        self.assertEqual(ef.eigenbasis.shape[0],53)
        self.assertEqual(ef.eigenbasis.shape[1],20480)
        self.assertAlmostEqual(ef.total_energy, 15326.398, places=0)
        self.assertAlmostEqual(ef.final_energy, 4432.7671, places=0)
        
    def test_2_EigenFaceWhiten(self):
        '''Eigenfaces: Stddev of whitened training vectors == 1.0...........'''
        ilog = None
        if 'ilog' in globals().keys():
            ilog = globals()['ilog']
        
        ef = self.trainScraps()
        
        # TODO: Double check.  Remove these later
        
        ssdb = ScrapShotsDatabase()
        keys = list(ssdb.keys())
        keys.sort()
        
        matrix = []
        for face_id in keys:
            #print "Processing:",face_id
            face = ssdb[face_id]
            if face.person_id not in self.scraps_training:
                continue

            fr = ef.getFaceRecord(face.image,face.face_rect,face.left_eye,face.right_eye)
            matrix.append(fr.eigenfaces_feature)
        
        #print matrix
        matrix = np.array(matrix)
        
        vals = matrix.std(axis=0)
        self.assertEquals(len(vals),53)
        self.assert_(np.abs(vals-1.0).max()<0.0001)

        
    def test_3_EigenFaceCovariance(self):
        '''Eigenfaces: Stddev of non whitened training vectors..............'''
        ilog = None
        if 'ilog' in globals().keys():
            ilog = globals()['ilog']
        
        ef = self.trainScraps(whiten=False)
        
        ssdb = ScrapShotsDatabase()
        keys = list(ssdb.keys())
        keys.sort()
        
        matrix = []
        for face_id in keys:
            face = ssdb[face_id]
            if face.person_id not in self.scraps_training:
                continue

            fr = ef.getFaceRecord(face.image,face.face_rect,face.left_eye,face.right_eye)
            matrix.append(fr.eigenfaces_feature)
        
        matrix = np.array(matrix)
        
        vals = matrix.std(axis=0).flatten()
        self.assertEquals(len(vals),53)
        self.assert_(np.abs(vals-1.0).max()>0.0001)
        for i in range(len(vals)):
            # Make sure that the eigenvalues are an estimate of the variance
            self.assertAlmostEquals(vals[i],np.sqrt(ef.eigenvalues[i]),places=4)
        

    def test_4_EigenfacesScrapShotsIdent(self):
        '''Eigenfaces: Scrapshot verification rate == 0.45 at FAR == 1/10...'''
        ilog = None
        if 'ilog' in globals().keys():
            ilog = globals()['ilog']
        
        ef = self.trainScraps()
        
        ssdb = ScrapShotsDatabase()
        keys = list(ssdb.keys())
        keys.sort()
        
        face_ids = []
        face_records = []
        
        for face_id in keys:
            face = ssdb[face_id]
            if face.person_id not in self.scraps_testing:
                continue

            fr = ef.getFaceRecord(face.image,face.face_rect,face.left_eye,face.right_eye)
            face_records.append(fr)
            face_ids.append(face_id)
            
        sim_matrix = ef.similarityMatrix(face_records,face_records)
        
        match_scores = []
        nonmatch_scores = []
        for i in range(1,len(face_ids)):
            for j in range(i+1,len(face_ids)):
                i_id = face_ids[i]
                j_id = face_ids[j]
                if i_id[:5] == j_id[:5]:
                    match_scores.append(sim_matrix[i,j])
                else:
                    nonmatch_scores.append(sim_matrix[i,j])
        
        roc = ROC(match_scores,nonmatch_scores,is_distance=False)
        roc_point = roc.getFAR(1.0/10.0)
        
        self.assertAlmostEqual(roc_point.frr,0.548571428571,places=2)

    def test_5_EigenfacesScrapShotsIdentNonWhitened(self):
        '''Eigenfaces: Scrapshot nonwhitened  rate == 0.46 at FAR == 1/10...'''
        ilog = None
        if 'ilog' in globals().keys():
            ilog = globals()['ilog']
        
        ef = self.trainScraps(whiten=False)
        
        ssdb = ScrapShotsDatabase()
        keys = list(ssdb.keys())
        keys.sort()
        
        face_ids = []
        face_records = []
        
        for face_id in keys:
            face = ssdb[face_id]
            if face.person_id not in self.scraps_testing:
                continue

            fr = ef.getFaceRecord(face.image,face.face_rect,face.left_eye,face.right_eye)
            face_records.append(fr)
            face_ids.append(face_id)
            
        sim_matrix = ef.similarityMatrix(face_records,face_records)
        
        match_scores = []
        nonmatch_scores = []
        for i in range(1,len(face_ids)):
            for j in range(i+1,len(face_ids)):
                i_id = face_ids[i]
                j_id = face_ids[j]
                if i_id[:5] == j_id[:5]:
                    match_scores.append(sim_matrix[i,j])
                else:
                    nonmatch_scores.append(sim_matrix[i,j])
        
        roc = ROC(match_scores,nonmatch_scores,is_distance=False)
        roc_point = roc.getFAR(1.0/10.0)
        
        self.assertAlmostEqual(roc_point.frr,0.53714285714285714,places=2)

#    def test_4_EigenfacesPickle(self):
#        '''Eigenfaces: Eigenfaces Pickle....................................'''
#
#    def test_5_EigenfacesPickleSaveLoad(self):
#        '''Eigenfaces: Eigenfaces Save and Load.............................'''
#
#    def test_5_EigenfacesReproject(self):
#        '''Eigenfaces: Eigenfaces Save and Load.............................'''


def test():
    '''Run the face test suite.'''
    pv.disableCommercialUseWarnings()
    
    fel_suite = unittest.TestLoader().loadTestsFromTestCase(TestFilterEyeLocator)
    eigenfaces_suite = unittest.TestLoader().loadTestsFromTestCase(TestEigenfaces)
    
    
    
    test_suites = [
                   fel_suite,
                   eigenfaces_suite,
                   ]
    
    pyvision_suite = unittest.TestSuite(test_suites)
    
    unittest.TextTestRunner(verbosity=2).run(pyvision_suite)

    
if __name__ == '__main__':
    # By default run the test suite
    #ilog = pv.ImageLog()
    test()
    #ilog.show()
    
    
    