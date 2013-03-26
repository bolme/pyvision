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

__version__ = "0.9.0 $Rev: 446 $"
__info__ = "$Id: __init__.py 446 2012-10-25 03:24:15Z bolme $"
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


from pyvision.types.img import Image,OpenCVToNumpy,NumpyToOpenCV

from pyvision.types.Video import Video, Webcam, VideoFromImages, VideoFromFileList, VideoFromImageStack, FFMPEGVideo, VideoFromDirectory

from pyvision.types.Point import Point,readPointsFile

from pyvision.types.Rect import Rect,BoundingRect,CenteredRect

from pyvision.types.Affine import  AffineNormalizePoints, AffineTranslate, AffineScale, AffineNonUniformScale, AffineRotate, AffineFromRect, AffineFromTile, AffineFromPoints, AffineFromPointsLS, AffineFromPointsRANSAC, AffineFromPointsLMeDs, AffinePerturb, AffineTransform

from pyvision.types.Perspective import  PerspectiveTransform, PerspectiveFromPoints

from pyvision.analysis.ImageLog import ImageLog

from pyvision.analysis.Montage import ImageMontage, VideoMontage

from pyvision.analysis.plot import Plot

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

from pyvision.util.fast_util import LocalMaximumDetector

from pyvision.util.windows import cosineWindow, hammingWindow, hannWindow

from pyvision.analysis.stats import pbinom, qbinom, cibinom, mcnemar_test, SummaryStats, cor, cov, cov2cor

from pyvision.point.GaborJets import GaborFilters, GaborImage, GaborJet, FilterBank, GaborWavelet

from pyvision.other.color import Histogram, hsBackProjectHist, rgbBackProjectHist, RGBHist, HSHist, HIST_HS, HIST_RGB

from pyvision.ml.pca import prcomp, pca

from pyvision.ml.lda import lda

from pyvision.ml.regression import LogisticRegression

from pyvision.other.surf import surf

from pyvision.other.texture import lbp,LBP_CLASSIC,LBP_RAD1,LBP_RAD2,LBP_RAD3,LBP_RAD4,LBP_RAD8

from pyvision.analysis.bee import parseSigSet,saveSigset,computeMaskMatrix,BEE_CODE_MAP,BEE_DONTCARE,BEE_MATCH,BEE_NONMATCH,BEEDistanceMatrix

from pyvision.data import AIRPLANE,BABOON,FRUITS,LENA,LOGO,TAZ_IMAGE,TAZ_VIDEO,FONT_ARIAL,BUGS_VIDEO,CAR_VIDEO, IRIS_DATA, IRIS_LABELS

from pyvision.surveillance.VideoStreamProcessor import *

from analysis.progress_bar import ProgressBar

from analysis.gui_tools import capturePointsFromMouse

from pyvision.face import REDUCED_LEYE, REDUCED_REYE, REDUCED_SIZE

from pyvision.types.kinect import Kinect

from pyvision.surveillance.kalman import KalmanFilter

from pyvision.ml.opencv_ml import svc_linear, svc_rbf, svr_linear, svr_rbf, random_forest, boost, gbtrees


# Beta components place holders
VideoTask = None

# Import the beta components
import beta


#================================== Misc Functions =====================================


def searchNames(text,item):
    '''Search dir(object) for patterns matching text'''
    for name in dir(item):
        if text.upper() in name.upper():
            print name
    
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
        print >> sys.stderr, "%d.%d.%d >= %d.%d.%d "%(major,minor,sub,rmajor,rminor,rsub),
        sys.stderr.flush()
        self.assert_(major > rmajor 
                     or major == rmajor and minor >= rminor 
                     or major == rmajor and minor == rminor and sub >= sub)

    def test_pil_version(self):
        import PIL.Image
        major,minor,sub = PIL.Image.VERSION.split('.')[:3]
        rmajor,rminor,rsub = 1,1,5 # 2008/03/20
        major,minor,sub = int(major),int(minor),int(sub)
        print >> sys.stderr, "%d.%d.%d >= %d.%d.%d "%(major,minor,sub,rmajor,rminor,rsub),
        sys.stderr.flush()
        self.assert_(major > rmajor 
                     or major == rmajor and minor >= rminor 
                     or major == rmajor and minor == rminor and sub >= sub)

    def test_opencv_version(self):
        import cv2
        major,minor,sub = cv2.__version__.split('.')[:3]
        rmajor,rminor,rsub = 2,4,2 # 2008/03/20
        major,minor,sub = int(major),int(minor),int(sub)
        print >> sys.stderr, "%d.%d.%d >= %d.%d.%d "%(major,minor,sub,rmajor,rminor,rsub),
        sys.stderr.flush()
        self.assert_(major > rmajor 
                     or major == rmajor and minor >= rminor 
                     or major == rmajor and minor == rminor and sub >= sub)

    def test_scipy__version(self):
        import scipy
        major,minor,sub = scipy.__version__.split('.')[:3]
        rmajor,rminor,rsub = 0,7,0 # 2008/03/20
        major,minor,sub = int(major),int(minor),int(sub)
        print >> sys.stderr, "%d.%d.%d >= %d.%d.%d "%(major,minor,sub,rmajor,rminor,rsub),
        sys.stderr.flush()
        self.assert_(major > rmajor 
                     or major == rmajor and minor >= rminor 
                     or major == rmajor and minor == rminor and sub >= sub)
        
    def test_numpy__version(self):
        import numpy
        major,minor,sub = numpy.__version__.split('.')[:3]
        rmajor,rminor,rsub = 1,0,4 # 2008/03/20
        major,minor,sub = int(major),int(minor),int(sub)
        print >> sys.stderr, "%d.%d.%d >= %d.%d.%d "%(major,minor,sub,rmajor,rminor,rsub),
        sys.stderr.flush()
        self.assert_(major > rmajor 
                     or major == rmajor and minor >= rminor 
                     or major == rmajor and minor == rminor and sub >= sub)
        
    def test_libsvm_version(self):
        import svm
        #major,minor,sub = svm.__version__.split('.')[:3]
        rmajor,rminor,_ = 2,86,0 # 2008/03/20
        #major,minor,sub = int(major),int(minor),int(sub)
        #print "%d.%d.%d >= %d.%d.%d "%(major,minor,sub,rmajor,rminor,rsub),
        print >> sys.stderr, "No way to get version numbers >= %d.%d "%(rmajor,rminor),
        sys.stderr.flush()
        #self.assert_(major > rmajor 
        #             or major == rmajor and minor >= rminor 
        #             or major == rmajor and minor == rminor and sub >= sub)
        self.assert_(True)        
        
def test():
    disableCommercialUseWarnings()
    
    version_suite = unittest.TestLoader().loadTestsFromTestCase(_VersionTest)

    from pyvision.types.Affine import _AffineTest
    affine_suite = unittest.TestLoader().loadTestsFromTestCase(_AffineTest)
    
    from pyvision.types.testsuite import _TestImage
    image_suite = unittest.TestLoader().loadTestsFromTestCase(_TestImage)

    from pyvision.vector.VectorClassifier import _TestVectorClassifier
    vc_suite = unittest.TestLoader().loadTestsFromTestCase(_TestVectorClassifier)
    
    #from pyvision.vector.SVM import _TestSVM
    #svm_suite = unittest.TestLoader().loadTestsFromTestCase(_TestSVM)

    from pyvision.vector.Polynomial import _PolyTest
    poly_suite = unittest.TestLoader().loadTestsFromTestCase(_PolyTest)
    
    from pyvision.point.DetectorCorner import _CornerTest
    corner_suite = unittest.TestLoader().loadTestsFromTestCase(_CornerTest)

    from pyvision.point.DetectorDOG import _DetectorDOGTestCase
    dog_suite = unittest.TestLoader().loadTestsFromTestCase(_DetectorDOGTestCase)
    
    from pyvision.point.DetectorHarris import _HarrisTest
    harris_suite = unittest.TestLoader().loadTestsFromTestCase(_HarrisTest)
    
    from pyvision.point.PhaseCorrelation import _TestPhaseCorrelation
    pc_suite = unittest.TestLoader().loadTestsFromTestCase(_TestPhaseCorrelation)
    
    from pyvision.optimize.testsuite import GeneticAlgorithmTest
    ga_suite = unittest.TestLoader().loadTestsFromTestCase(GeneticAlgorithmTest)
    
    from pyvision.face.CascadeDetector import _TestCascadeDetector
    cd_suite = unittest.TestLoader().loadTestsFromTestCase(_TestCascadeDetector)
    
    from pyvision.face.PCA import _TestFacePCA
    fpca_suite = unittest.TestLoader().loadTestsFromTestCase(_TestFacePCA)
    
    from pyvision.face.FilterEyeLocator import _TestFilterEyeLocator
    asefed_suite = unittest.TestLoader().loadTestsFromTestCase(_TestFilterEyeLocator)

    # Replaced by ASEF work
    # from pyvision.face.SVMEyeDetector import _TestSVMEyeDetector 
    # svmed_suite = unittest.TestLoader().loadTestsFromTestCase(_TestSVMEyeDetector)
    
    from pyvision.edge.canny import _TestCanny
    canny_suite = unittest.TestLoader().loadTestsFromTestCase(_TestCanny)
    
    from pyvision.analysis.stats import _TestStats
    stats_suite = unittest.TestLoader().loadTestsFromTestCase(_TestStats)
    
    from pyvision.analysis.Table import _TestTable
    table_suite = unittest.TestLoader().loadTestsFromTestCase(_TestTable)
    
    from pyvision.analysis.classifier.ConfusionMatrix import _TestConfusionMatrix
    cm_suite = unittest.TestLoader().loadTestsFromTestCase(_TestConfusionMatrix)
    
    from pyvision.other.testsuite import _TestDistance
    dist_suite = unittest.TestLoader().loadTestsFromTestCase(_TestDistance)

    from pyvision.other.testsuite import _TestNormalize
    norm_suite = unittest.TestLoader().loadTestsFromTestCase(_TestNormalize)
    
    from pyvision.other.testsuite import _TestSURF
    surf_suite = unittest.TestLoader().loadTestsFromTestCase(_TestSURF)
    
    
    test_suites = [
                   version_suite,
                   affine_suite,
                   image_suite,
                   vc_suite,
                   #svm_suite, #TODO: uncomment
                   poly_suite,
                   corner_suite,
                   dog_suite,
                   harris_suite,
                   pc_suite,
                   ga_suite,
                   cd_suite,
                   fpca_suite,
                   asefed_suite,
                   #svmed_suite,
                   canny_suite,
                   stats_suite,
                   table_suite,
                   cm_suite,
                   dist_suite,
                   norm_suite,
                   surf_suite,
                   ]
    
    pyvision_suite = unittest.TestSuite(test_suites)
    
    unittest.TextTestRunner(verbosity=2).run(pyvision_suite)
        
        
if __name__ == '__main__':
    test()
