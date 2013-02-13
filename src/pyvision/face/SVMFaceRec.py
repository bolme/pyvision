import pyvision as pv
from pyvision.vector.SVM import SVM,TYPE_C_SVC
#from pyvision.vector.VectorClassifier import NORM_NONE
#from pyvision.vector.PCA import PCA
from scipy import ndimage
#import random


class SVMFaceRec:
    
    def __init__(self):
        self.tile_size = (128,160)
        self.leye = pv.Point(26.0,40.0)
        self.reye = pv.Point(102.0,40.0)
        self.norm_sigma = 8.0
        
        self.svm = None
        
        self.n_faces = 0
        self.n_labels = 0
        
        self.training_data = {}
        
            
    def preprocess(self,im,leye,reye,ilog=None):
        im = pv.Image(im.asPIL())
        affine = pv.AffineFromPoints(leye,reye,self.leye,self.reye,self.tile_size)
        tile = affine.transformImage(im)
        
        mat = tile.asMatrix2D()
        
        # High pass filter the image
        mat = mat - ndimage.gaussian_filter(mat,self.norm_sigma)
        
        # Value normalize the image.
        mat = mat - mat.mean()
        mat = mat / mat.std()
                
        tile = pv.Image(mat)
        
        return tile
        
    def addTraining(self,im,leye,reye,sub_id,ilog=None):
        self.svm = None
        
        tile = self.preprocess(im,leye,reye,ilog)
        if not self.training_data.has_key(sub_id):
            self.training_data[sub_id] = []
        
        self.training_data[sub_id].append(tile)

        self.n_labels = len(self.training_data)
        self.n_faces += 1
        
        
    def train(self, C = None, Gamma = None, ilog=None, callback=None):
        # Create the SVM
        self.svm = SVM(type=TYPE_C_SVC)
        
        # Add training data
        for sub_id, tiles in self.training_data.iteritems():
            for tile in tiles:
                self.svm.addTraining(sub_id,tile)
                
        # Train the SVM
        if C != None and Gamma != None:
            self.svm.train(C_range = C, G_range = Gamma, verbose=True,callback=callback)
        else:
            #Automatic
            self.svm.train(verbose=True,callback=callback)
            

        
    def predict(self,im,leye,reye):
        assert self.svm != None
        
        tile = self.preprocess(im,leye,reye)
        return self.svm.predict(tile)

    def score(self,im,leye,reye):
        tile = self.preprocess(im,leye,reye)
        return self.svm.predictSVMValues(tile)
    
    def reset(self):
        self.svm = None
        
        self.n_faces = 0
        self.n_labels = 0
        
        self.training_data = {}
        
    def isTrained(self):
        return self.svm != None
        
        
        
        
        
        
        
        
        
        
        