'''
Created on Apr 16, 2009

@author: bolme
'''
from pyvision.point.DetectorROI import DetectorROI
#import pyvision as pv
#from scipy import weave
import cv2

def ExtractSURF(im,min_hessian=300):
    '''
    Uses OpenCV to extract SURF keypoints.  Currently does not compute SURF features.
    
    TODO: An option should be added to also compute and return the SURF descriptors.
    TODO: This should be extended with options for octaves and levels.
    TODO: I believe there are no memory leaks but this should be checked. cvSURFParams?
    '''
    cvim= im.asOpenCVBW()
    #mat = int(cvim.this)
    min_hessian = float(min_hessian)
    #TODO: OpenCV python interface now includes cv.ExtractSURF(cvim, mask, storage, params)
    #This is my (Steve's) attempt at this, but I am concerned we're not returning the
    # some of the information once this gets back to the caller...perhaps the parent
    # class is filtering out the addtnl data that SURF points provide?
    
    #TODO: Now that we have the descriptors, we need to return them to user if desired.
    (keypts, _) = cv.ExtractSURF(cvim, None, cv.CreateMemStorage(), (0, min_hessian, 3, 1))
    
    keypoints = list()
    for ((x, y), laplacian, size, direction, hessian) in keypts:
        keypoints.append((hessian,x,y,size,direction,laplacian) )
    
    return keypoints


class DetectorSURF(DetectorROI):
    def __init__(self, min_hessian=400.0, **kwargs):
        '''
        '''
        self.min_hessian = min_hessian
        DetectorROI.__init__(self,**kwargs)
        
    
    def _detect(self,im):
        keypoints = ExtractSURF(im,min_hessian=self.min_hessian)
        keypoints.sort(lambda x,y: -cmp(x[0],y[0]))         
        return keypoints
