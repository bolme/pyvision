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

import os.path
#from PIL.Image import ANTIALIAS
import scipy
import unittest
#from scipy.signal import fft2

from pyvision.face.FaceRecognizer import FaceRecognizer
from pyvision.analysis.FaceAnalysis import FaceRecognitionTest
import pyvision.vector.PCA
from pyvision.other.normalize import *
from pyvision.types.img import Image
from pyvision.types.Point import Point
from pyvision.analysis.face import EyesFile
from pyvision.analysis.roc import *
from pyvision.types.Affine import *

PCA_L1      = 1
PCA_L2      = 2
PCA_COS     = 3

PCA_NO_NORM   = 1
PCA_MEAN_STD_NORM  = 2
PCA_MEAN_UNIT_NORM = 3
PCA_UNIT_NORM = 4

class PCA(FaceRecognizer):
    ''' This is a basic implementation of PCA'''
    
    def __init__(self, face_size=(128,128), left_eye=Point(32,52), right_eye=Point(96,52), normalize=PCA_MEAN_STD_NORM, measure=PCA_COS, whiten=True, drop_front=2, basis_vectors=100):
        '''Crate a PCA classifier'''
        FaceRecognizer.__init__(self)
        
        self.face_size = face_size
        self.pca      = pyvision.vector.PCA.PCA()
        self.norm     = normalize
        self.trained  = False
        
        self.whiten = whiten
        self.drop_front = drop_front
        self.basis_vectors = basis_vectors
        self.measure = measure
        self.left_eye = left_eye
        self.right_eye = right_eye

    def cropFace(self,im,eyes):
        left,right = eyes
        affine = AffineFromPoints(left,right,self.left_eye,self.right_eye,self.face_size)
        im = affine.transformImage(im)
        return im
    
    #def addTraining(self,img,rect=None,eyes=None):
    #    ''' '''
    #    assert not self.trained
    #    
    #    img = self.cropFace(img,eyes) 
    #    vec = self.computeVector(img)
    #    self.pca.addFeature(vec)

    def computeFaceRecord(self,img,rect=None,eyes=None):
        '''Given an image and face detection box, compute a face identification record'''
        assert self.trained
        
        img = self.cropFace(img,eyes) 
        vec = self.computeVector(img)
        fir = self.pca.project(vec,whiten=True)
        if self.measure == PCA_COS:
            scale = scipy.sqrt((fir*fir).sum())
            fir = (1.0/scale)*fir
        return fir
    
    def computeVector(self,img):
        '''Creates a vector from a face'''
        #face = img.asPIL().crop(rect.box()).resize(self.face_size,ANTIALIAS)
        vec  = img.asMatrix2D().flatten()
        
        if self.norm == PCA_MEAN_STD_NORM:
            vec = meanStd(vec)
        if self.norm == PCA_MEAN_UNIT_NORM:
            vec = meanUnit(vec)
        if self.norm == PCA_UNIT_NORM:
            vec = unit(vec)
            
        return vec
    
    def train(self):
        '''Train the PCA classifier'''
        assert self.trained == False
        
        for img,rect,eyes,id in self.training_data:
            img = self.cropFace(img,eyes) 
            vec = self.computeVector(img)
            self.pca.addFeature(vec)
        
        self.pca.train( drop_front=self.drop_front, number=self.basis_vectors)
        
        self.trained = True
        
       
    def similarity(self,fir1,fir2):
        '''Compute the similarity of two faces'''
        assert self.trained == True

        if self.measure == PCA_L1:
            return (scipy.abs(fir1-fir2)).sum()

        if self.measure == PCA_L2:
            return scipy.sqrt(((fir1-fir2)*(fir1-fir2)).sum())

        if self.measure == PCA_COS:
            return (fir1*fir2).sum()
        
        raise NotImplementedError("Unknown distance measure: %d"%self.measure)
    
    
    def getBasis(self):
        basis = self.pca.getBasis()
        images = []
        
        print basis.shape
        r,c = basis.shape
        for i in range(r):
            im = basis[i,:]
            im = im.reshape(self.face_size)
            im = Image(im)
            images.append(im)
        print len(images)
        return images
        
    

SCRAPS_FACE_DATA = os.path.join(pyvision.__path__[0],"data","csuScrapShots")

PCA_SIZE = (64,64)

class _TestFacePCA(unittest.TestCase):
    
    def setUp(self):
        self.images = []
        self.names = []
        
        self.eyes = EyesFile(os.path.join(SCRAPS_FACE_DATA,"coords.txt"))
        for filename in self.eyes.files():
            img = Image(os.path.join(SCRAPS_FACE_DATA, filename + ".pgm"))
            self.images.append(img)
            self.names.append(filename)
        
        self.assert_( len(self.images) == 173 )

    
    def test_pca_scraps(self):
        face_test = FaceRecognitionTest.FaceRecognitionTest(name='PCA_CSUScraps',score_type=FaceRecognitionTest.SCORE_TYPE_HIGH)
        pca = PCA(drop_front=2,basis_vectors=55)
        
        for im_name in self.eyes.files():
            im = Image(os.path.join(SCRAPS_FACE_DATA, im_name + ".pgm"))
            rect = self.eyes.getFaces(im_name)
            eyes = self.eyes.getEyes(im_name)
            pca.addTraining(im,rect=rect[0],eyes=eyes[0])

        pca.train()
                
        face_records = {}
        for im_name in self.eyes.files():
            im = Image(os.path.join(SCRAPS_FACE_DATA, im_name + ".pgm"))
            rect = self.eyes.getFaces(im_name)
            eyes = self.eyes.getEyes(im_name)
            fr = pca.computeFaceRecord(im,rect=rect[0],eyes=eyes[0])
            face_records[im_name] = fr
        
        for i_name in face_records.keys():
            scores = []
            for j_name in face_records.keys():
                similarity = pca.similarity(face_records[i_name],face_records[j_name])
                scores.append((j_name,similarity))
            face_test.addSample(i_name,scores)
            
        #print face_test.rank1_bounds
        self.assertAlmostEqual(face_test.rank1_rate,0.43930635838150289)
        self.assertAlmostEqual(face_test.rank1_bounds[0],0.3640772723094895)
        self.assertAlmostEqual(face_test.rank1_bounds[1],0.51665118592791259)

        roc = face_test.getROCAnalysis()        

        # Test based of fpr=0.01
        roc_point = roc.getFAR(far=0.01)
        #TODO: does not work... 
        #self.assertAlmostEqual(1.0-roc_point.frr,0.16481069042316257)

        # Test the equal error rate
        #fp,tp,th = roc.findEqualError()
        #self.assertAlmostEqual(tp,0.68819599109131402)
        


            
