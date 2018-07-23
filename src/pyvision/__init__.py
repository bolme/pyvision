# PyVision License
#
# Copyright (c) 2006-2011 David S. Bolme
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

'''
The top level of this package contains some basic types used throughout 
PyVision.  Subpackages some of the more advanced functionality of the 
PyVision library.  These include:

    * Image Processing    
    * Detection           
    * Machine Learning    
    * Optimization/Search 
    * Face Recognition    
    * Analysis            

Typically, all these types are used in a program.  A good convention is to 
import the pyvision library as "pv" and then prefix all function names with "pv." 
This will avoid possible namespace conflicts. For example::

    import pyvision as pv
    im = pv.Image(filename) 
    im.annotateLabel(pv.Point(10,10),"Hello, World!")
    im.show()
'''

'''
-----------------------------------------------------------------------------
                            ALGORITHM DIRECTORY
Algorithm Name        Problem             Module
-----------------------------------------------------------------------------
Support Vector Mach.  classify/regression pyvision.vector.SVM
PCA

Cascade Classifier    face/object detect  pyvision.face.CascadeClassifier
PCA (Face)            face recognition    pyvision.face.PCA

Genetic Algorithm     optimization        pyvision.optimize.GeneticAlgorithm
'''

import unittest
import sys
import pickle as pkl

__version__ = "1.3.0"
__info__ = "https://github.com/bolme/pyvision/wiki"
__license__= '''
PyVision License

Copyright (c) 2006-2010 David S. Bolme
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
 
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither name of copyright holders nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Warning: Some parts of PyVision may link to libraries using more 
restrictive licenses and some algorithms in PyVision by be covered 
under patents.  In these cases PyVision should display a warning
for commercial use.  If you believe this a warning should be added
for any algorithm or interface please contact me at
bolme@cs.colostate.edu
'''

__all__ = ['analysis','edge','face','optimize','other','point','types','vector']

WARN_COMMERCIAL_USE = True

def disableCommercialUseWarnings():
    '''
    Most of PyVision is released under the BSD license and
    can therefore be used free of charge in commercial 
    projects. In some limited cases PyVision uses algorithms
    that are covered by patents or source code released under
    copy left open source licenses such as GPL which may make
    software produced using those components unsuitable for 
    commercial distribution. 
    
    When a PyVision module contains or links to  non-commercial 
    code a warning message will be printed to stdout.  If you
    would like to disable these warning simply call this function
    before importing the offending module.  The users PyVision are 
    responsible for determining if their use of those components 
    respects all applicable licenses and patents.
    
    If you believe this a warning should be added for any algorithm 
    or interface please contact me at bolme@cs.colostate.edu 
    '''
    global WARN_COMMERCIAL_USE
    WARN_COMMERCIAL_USE = False

#Import basic pyvision types

#================================== Imports =====================================


HAS_CV2 = False

try:
    import cv2
    HAS_CV2 = True
except:
    pass

from pyvision.types.img import Image

from pyvision.types.Video import Video, Webcam, VideoFromImages, VideoFromFileList, VideoFromImageStack, VideoFromDirectory

from pyvision.types.Point import Point,readPointsFile

from pyvision.types.Rect import Rect,BoundingRect,CenteredRect

from pyvision.types.Affine import  AffineNormalizePoints, AffineTranslate, AffineScale, AffineNonUniformScale, AffineRotate, AffineFromRect, AffineFromTile, AffineFromPoints, AffineFromPointsLS, AffineFromPointsRANSAC, AffineFromPointsLMeDs, AffinePerturb, AffineTransform

from pyvision.types.Perspective import  PerspectiveTransform, PerspectiveFromPoints, logPolar

from pyvision.analysis.ImageLog import ImageLog

from pyvision.analysis.Montage import ImageMontage, VideoMontage

try:
    from pyvision.analysis.plot import Plot
except:
    print("Warning: could not load plotting library.")
    
from pyvision.analysis.Table import Table

from pyvision.analysis.Timer import Timer

from pyvision.analysis.roc import ROC,ROC_LOG_SAMPLED,ROC_MATCH_SAMPLED

from pyvision.types.ImageBuffer import ImageBuffer

from pyvision.surveillance.BackgroundSubtraction import AbstractBGModel, FrameDifferencer, MedianFilter, ApproximateMedianFilter, MotionCompensatedFrameDifferencer, BG_SUBTRACT_FD, BG_SUBTRACT_MCFD, BG_SUBTRACT_MF, BG_SUBTRACT_AMF

from pyvision.surveillance.MotionDetector import MotionDetector,BOUNDING_RECTS,STANDARDIZED_RECTS

from pyvision.surveillance.optical_flow import OpticalFlow

from pyvision.other.normalize import clipRange, meanStd, meanUnit, unit, selfQuotientImage, lowPassFilter, highPassFilter, bandPassFilter, gaussianFilter

from pyvision.other.distance import boolToUbyte, ubyteToBool, hamming, hamming_table

from pyvision.ml.knn import RobustPNorm,PNorm,correlation,chisquared,KNearestNeighbors,FLANNTree,FLANN_IMPORTED

try:
    from pyvision.util.fast_util import LocalMaximumDetector
except:
    print("Warning: could not import fast_util.")
    
from pyvision.util.windows import cosineWindow, hammingWindow, hannWindow

from pyvision.analysis.stats import pbinom, qbinom, cibinom, mcnemar_test, SummaryStats, cor, cov, cov2cor

from pyvision.point.GaborJets import GaborFilters, GaborImage, GaborJet, FilterBank, GaborWavelet

from pyvision.other.color import Histogram, hsBackProjectHist, rgbBackProjectHist, RGBHist, HSHist, HIST_HS, HIST_RGB

from pyvision.ml.pca import prcomp, pca

from pyvision.ml.lda import lda

from pyvision.ml.regression import LogisticRegression

from pyvision.other.texture import lbp,LBP_CLASSIC,LBP_RAD1,LBP_RAD2,LBP_RAD3,LBP_RAD4,LBP_RAD8

from pyvision.analysis.bee import parseSigSet,saveSigset,computeMaskMatrix,BEE_CODE_MAP,BEE_DONTCARE,BEE_MATCH,BEE_NONMATCH,BEEDistanceMatrix

try:
    from pyvision.data import AIRPLANE,BABOON,FRUITS,LENA,LOGO,TAZ_IMAGE,TAZ_VIDEO,FONT_ARIAL,BUGS_VIDEO,CAR_VIDEO, IRIS_DATA, IRIS_LABELS
except:
    print("Warning: could not import data.")

from .analysis.progress_bar import ProgressBar

from .analysis.gui_tools import capturePointsFromMouse

from pyvision.face import REDUCED_LEYE, REDUCED_REYE, REDUCED_SIZE

# Removed for opencv 3 from pyvision.types.kinect import Kinect

from pyvision.surveillance.kalman import KalmanFilter

from pyvision.beta.vtm import VideoTaskManager, VideoTask, _VideoDataItem

from pyvision.analysis.html_report import HtmlReport
# Import the beta components
from . import beta

import time

IMAGE_EXTENSIONS = ['.JPG','.JPEG','.GIF','.TIF','.TIFF','.PNG','.BMP','.PGM','.PPM',]
VIDEO_EXTENSIONS = ['.MOV','.M4V','.FLV','.AVI','.MPEG','.MJPEG','.MP4','.MPG','.WMV',]

#================================== Misc Functions =====================================

def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

def searchNames(text,item):
    '''Search dir(object) for patterns matching text'''
    for name in dir(item):
        if text.upper() in name.upper():
            print(name)
    
def getTypeName(item):
    '''
    Return a short name describing the type.
    '''
    try:
        if isinstance(item,list):
            type_list = set([getTypeName(each) for each in item])
            type_name = 'list:%d('%len(item)
            for each in type_list:
                type_name += each+","
            type_name = type_name[:-1]
            type_name += ")"
        elif isinstance(item,tuple):
            type_list = set([getTypeName(each) for each in item])
            type_name = 'tuple:%d('%len(item)
            for each in type_list:
                type_name += each+","
            type_name = type_name[:-1]
            type_name += ")"
        elif isinstance(item,dict):
            type_list = set([getTypeName(each) for _,each in item.items()])
            type_name = 'dict:%d('%len(item)
            for each in type_list:
                type_name += each+","
            type_name = type_name[:-1]
            type_name += ")"
        elif isinstance(item,str):
            type_name = 'str'
        elif isinstance(item,int):
            type_name = 'int'
        elif isinstance(item,float):
            type_name = 'float'
        elif 'instance' in repr(type(item)):
            type_name = 'instance'
        else:
            type_name = str(type(item))
            if type_name.startswith('<class'):
                type_name = type_name[8:-2]
                type_name = type_name.split('.')[-1]
            if type_name.startswith('<type'):
                type_name = type_name[7:-2]
                type_name = type_name.split('.')[-1]

            if type_name == 'ndarray':
                type_name += ":%s:%s"%(item.shape,item.dtype)
                type_name = "".join(type_name.split())
    except:
        type_name = 'unknown'
        
    return type_name
    
    
def inspectObject(item,name='<top>',max_depth=5,verbose=False,print_depth=0,info=None):
    '''
    Produce a pv.Table describing this object and its members.
    '''
    if max_depth < 0:
        return
    
    if not verbose and name[:2] == '__':
        return
    if 'function' in str(type(item)):
        return
    if 'method' in str(type(item)):
        return
    
    if info == None:
        info = pv.Table()
        info.setColumnFormat('name','%s')
        print(dir(info))
    i = info.nRows()

    info[i,'name'] = ('    '*print_depth) + name
    try:
        
        # Add info about the type
        type_name = str(getTypeName(item))
        if len(type_name) > 30:
            type_name = type_name[:27]+'...'
        info[i,'type'] = type_name
    
        value = " ".join(repr(item).split())
        if len(value) > 30:
            value = value[:27]+'...'
        info[i,'value'] = value
        
        try:
            item_size = len(pkl.dumps(item, protocol=2))
        except:
            item_size = 'error'
            
        info[i,'pickle size'] = item_size
        
        if type_name in ['int','float','str']:
            return info
        if type_name.startswith('ndarray'):
            return info
        
        for each in dir(item):
            inspectObject(getattr(item,each),name=each,max_depth=max_depth-1,verbose=verbose,print_depth=print_depth+1,info=info)
            
    
        if isinstance(item,list) or isinstance(item,tuple):
            processed_types = set()
            for each in item:
                if type(each) in processed_types:
                    continue
                processed_types.add(type(each))
                inspectObject(each,name='<sample item>',max_depth=max_depth-1,verbose=verbose,print_depth=print_depth+1,info=info)
                
        if isinstance(item,dict):
            processed_types = set()
            for _,each in item.items():
                if type(each) in processed_types:
                    continue
                processed_types.add(type(each))
                inspectObject(each,name='<sample item>',max_depth=max_depth-1,verbose=verbose,print_depth=print_depth+1,info=info)
    except:
        info[i,'error'] = "Could not process this object."    
    return info

def numpy2r(mat):
    '''
    Convert a matrix into a string that can be executed in R.
    '''
    num_rows = mat.shape[0]
    data = mat.T.flatten()
    data = ["%f"%each for each in data]
    data = "c(" + ",".join(data) + ")"
    return "matrix(%s,%d)"%(data,num_rows)

def isImage(filename):
    '''
    Determines if the filename corresponds to a known image extension.
    '''
    for ext in IMAGE_EXTENSIONS:
        if filename.upper().endswith(ext):
            return True
    return False

def isVideo(filename):
    '''
    Determines if the filename corresponds to a known image extension.
    '''
    for ext in VIDEO_EXTENSIONS:
        if filename.upper().endswith(ext):
            return True
    return False


def runningInNotebook():
    '''
    @return: True if python interpreter is running in an iPython HTML Notebook.
    (This may not work on all platforms.)
    '''
    # Check that the type of ipython instance is consistent with the notebook.
    try:
        import IPython
        gui = IPython.core.pylabtools.find_gui_and_backend() #@UndefinedVariable
        if 'inline' not in gui:
            #print 'missing inline'
            return False
    except:
        #print "error importing"
        return False
    
    #print "success in notebook"
    return True
    


# TODO: Features to be included in the initial release.
#     analysis: 
#         FaceRec(FERET,BioID) 
#         FaceDetection(perdue,FERET,BioID) 
#         EyeFinding (FERET,BioID)
#         AutoFace(FERET,BioID)
#         Support: ImageLog, Stats, ROC, RegressionEval, ClassifierEval, HomographyEval
#     edge:
#         Algorithms: canny, sobel, prewitt, hough transform
#     face:
#         Algorithms: CascadeDetect, SVM Eye Localization, Normalization, PCA Recognition, LDA Recognition, SVM Recognition 
#         Cascade Training
#     gui: 
#         PointSelector (eye_coords)
#         CovariateTool
#     other:
#         ChangeDetection
#         Normalization
#     point:
#         DetectorCorner
#         DetectorDog
#         PhaseCorrelation
#         SVMPointLocator
#     transform:
#         Affine(least squares)
#         Perspective(least squares/homography)
#         AutoHomography
#     types:
#         Image, Point, Rect
#     vector:
#         PCA
#         LDA
#         SVM
#         ID3
#         LinearRegression
#         2DPoly
#         VectorClassifier


# TODO: Features ideas for future releases
#     OpenCV Webcam Support
#     Animation Support
#     Interactive 3D Visualization
#     Live Demo Support
#     EBGM
#     Object Recognition
#     Neural Networks
#     Fundamental Matrix and 3D Reconstruction
#     LBP - Texture analysis and face recognition.

class _VersionTest(unittest.TestCase):
    ''' Check the installed versions of the dependencies '''
    
    def test_python_version(self):
        major,minor,sub = sys.version.split(' ')[0].split('.')[:3]
        rmajor,rminor,rsub = 2,3,0 # 2008/03/20
        major,minor,sub = int(major),int(minor),int(sub)
        print("%d.%d.%d >= %d.%d.%d "%(major,minor,sub,rmajor,rminor,rsub))
        sys.stderr.flush()
        self.assertTrue(major > rmajor 
                     or major == rmajor and minor >= rminor 
                     or major == rmajor and minor == rminor and sub >= sub)

    def test_pil_version(self):
        import PIL.Image
        major,minor,sub = PIL.Image.VERSION.split('.')[:3]
        rmajor,rminor,rsub = 1,1,5 # 2008/03/20
        major,minor,sub = int(major),int(minor),int(sub)
        print("%d.%d.%d >= %d.%d.%d "%(major,minor,sub,rmajor,rminor,rsub))
        sys.stderr.flush()
        self.assertTrue(major > rmajor 
                     or major == rmajor and minor >= rminor 
                     or major == rmajor and minor == rminor and sub >= sub)

    def test_opencv_version(self):
        if HAS_CV2:
            import cv2
            major,minor,sub = cv2.__version__.split('.')[:3]
            rmajor,rminor,rsub = 2,4,2 # 2008/03/20
            major,minor,sub = int(major),int(minor),int(sub)
            print("%d.%d.%d >= %d.%d.%d "%(major,minor,sub,rmajor,rminor,rsub))
            sys.stderr.flush()
            self.assertTrue(major > rmajor 
                         or major == rmajor and minor >= rminor 
                         or major == rmajor and minor == rminor and sub >= sub)
        else:
            print("NA ")
            sys.stderr.flush()
            

    def test_scipy__version(self):
        import scipy
        major,minor,sub = scipy.__version__.split('.')[:3]
        rmajor,rminor,rsub = 0,7,0 # 2008/03/20
        major,minor,sub = int(major),int(minor),int(sub)
        print("%d.%d.%d >= %d.%d.%d "%(major,minor,sub,rmajor,rminor,rsub))
        sys.stderr.flush()
        self.assertTrue(major > rmajor 
                     or major == rmajor and minor >= rminor 
                     or major == rmajor and minor == rminor and sub >= sub)
        
    def test_numpy__version(self):
        import numpy
        major,minor,sub = numpy.__version__.split('.')[:3]
        rmajor,rminor,rsub = 1,0,4 # 2008/03/20
        major,minor,sub = int(major),int(minor),int(sub)
        print("%d.%d.%d >= %d.%d.%d "%(major,minor,sub,rmajor,rminor,rsub))
        sys.stderr.flush()
        self.assertTrue(major > rmajor 
                     or major == rmajor and minor >= rminor 
                     or major == rmajor and minor == rminor and sub >= sub)
                

        
def test():
    disableCommercialUseWarnings()
    
    test_suites = []
    
    version_suite = unittest.TestLoader().loadTestsFromTestCase(_VersionTest)
    test_suites.append(version_suite)
    
    from pyvision.types.Affine import _AffineTest
    affine_suite = unittest.TestLoader().loadTestsFromTestCase(_AffineTest)
    test_suites.append(affine_suite)
    
    from pyvision.testsuite.image_tests import TestImage, TestVideo
    image_suite = unittest.TestLoader().loadTestsFromTestCase(TestImage)
    test_suites.append(image_suite)
    video_suite = unittest.TestLoader().loadTestsFromTestCase(TestVideo)
    test_suites.append(video_suite)
    
    #from pyvision.vector.VectorClassifier import _TestVectorClassifier
    #vc_suite = unittest.TestLoader().loadTestsFromTestCase(_TestVectorClassifier)
    #test_suites.append(vc_suite)
    
    #from pyvision.vector.SVM import _TestSVM
    #svm_suite = unittest.TestLoader().loadTestsFromTestCase(_TestSVM)

    #from pyvision.vector.Polynomial import _PolyTest
    #poly_suite = unittest.TestLoader().loadTestsFromTestCase(_PolyTest)
    #test_suites.append(poly_suite)
    
    from pyvision.point.DetectorCorner import _CornerTest
    corner_suite = unittest.TestLoader().loadTestsFromTestCase(_CornerTest)
    test_suites.append(corner_suite)
    
    from pyvision.point.PhaseCorrelation import _TestPhaseCorrelation
    pc_suite = unittest.TestLoader().loadTestsFromTestCase(_TestPhaseCorrelation)
    test_suites.append(pc_suite)
    
    from pyvision.optimize.testsuite import GeneticAlgorithmTest
    ga_suite = unittest.TestLoader().loadTestsFromTestCase(GeneticAlgorithmTest)
    test_suites.append(ga_suite)
    
    try:
        from pyvision.face.CascadeDetector import _TestCascadeDetector
        cd_suite = unittest.TestLoader().loadTestsFromTestCase(_TestCascadeDetector)
        test_suites.append(cd_suite)
    except:
        print("WARNING: Could not load _TestCascadeDetector.")
        
    #from pyvision.face.PCA import _TestFacePCA
    #fpca_suite = unittest.TestLoader().loadTestsFromTestCase(_TestFacePCA)
    #test_suites.append(fpca_suite)
    
    from pyvision.face.FilterEyeLocator import _TestFilterEyeLocator
    asefed_suite = unittest.TestLoader().loadTestsFromTestCase(_TestFilterEyeLocator)
    test_suites.append(asefed_suite)
    
    # Replaced by ASEF work
    # from pyvision.face.SVMEyeDetector import _TestSVMEyeDetector 
    # svmed_suite = unittest.TestLoader().loadTestsFromTestCase(_TestSVMEyeDetector)
    if HAS_CV2:
        from pyvision.edge.canny import _TestCanny
        canny_suite = unittest.TestLoader().loadTestsFromTestCase(_TestCanny)
        test_suites.append(canny_suite)

        from pyvision.point.DetectorHarris import _HarrisTest
        harris_suite = unittest.TestLoader().loadTestsFromTestCase(_HarrisTest)
        test_suites.append(harris_suite)        

        from pyvision.point.DetectorDOG import _DetectorDOGTestCase
        dog_suite = unittest.TestLoader().loadTestsFromTestCase(_DetectorDOGTestCase)
        test_suites.append(dog_suite)
        
    
    from pyvision.analysis.stats import _TestStats
    stats_suite = unittest.TestLoader().loadTestsFromTestCase(_TestStats)
    test_suites.append(stats_suite)
    
    from pyvision.analysis.Table import _TestTable
    table_suite = unittest.TestLoader().loadTestsFromTestCase(_TestTable)
    test_suites.append(table_suite)
    
    from pyvision.analysis.classifier.ConfusionMatrix import _TestConfusionMatrix
    cm_suite = unittest.TestLoader().loadTestsFromTestCase(_TestConfusionMatrix)
    test_suites.append(cm_suite)
    
    from pyvision.other.testsuite import _TestDistance
    dist_suite = unittest.TestLoader().loadTestsFromTestCase(_TestDistance)
    test_suites.append(dist_suite)
    
    from pyvision.other.testsuite import _TestNormalize
    norm_suite = unittest.TestLoader().loadTestsFromTestCase(_TestNormalize)
    test_suites.append(norm_suite)
    
    try:
        from pyvision.other.testsuite import _TestSURF
        surf_suite = unittest.TestLoader().loadTestsFromTestCase(_TestSURF)
        test_suites.append(surf_suite)
    except:
        print("WARNING: Could not load _TestSURF.")
            
    pyvision_suite = unittest.TestSuite(test_suites)
    
    unittest.TextTestRunner(verbosity=2).run(pyvision_suite)
        
        
if __name__ == '__main__':
    test()
