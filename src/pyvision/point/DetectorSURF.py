'''
Created on Apr 16, 2009

@author: bolme
'''
from pyvision.point.DetectorROI import DetectorROI
#import pyvision as pv
#from scipy import weave
import cv

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
    (keypts, descriptors) = cv.ExtractSURF(cvim, None, cv.CreateMemStorage(), (0, min_hessian, 3, 1))
    
    keypoints = list()
    for ((x, y), laplacian, size, dir, hessian) in keypts:
        keypoints.append((hessian,x,y,size,dir,laplacian) )
    
    return keypoints
#   keypoints = weave.inline(
#        '''   
#        CvMat* im = (CvMat*)mat;
#        
#        CvMemStorage* storage = cvCreateMemStorage();
#        CvSeq* keypoints = cvCreateSeq(0,sizeof(CvSeq),sizeof(CvSURFPoint),storage);
#        cvExtractSURF(im,NULL,&keypoints,NULL,storage,cvSURFParams(min_hessian));
#        
#        
#        int n = keypoints->total;
#        PyObject* list = PyList_New(n);
#        CvSURFPoint pt;
#        for(int i = 0 ; i < n; i++){
#            cvSeqPop(keypoints,&pt);
#            
#            PyObject* tuple = PyTuple_New(5);
#            PyTuple_SetItem(tuple, 1, PyFloat_FromDouble(pt.pt.x));
#            PyTuple_SetItem(tuple, 2, PyFloat_FromDouble(pt.pt.y));
#            PyTuple_SetItem(tuple, 3, PyInt_FromLong(pt.size));
#            PyTuple_SetItem(tuple, 4, PyFloat_FromDouble(pt.dir));
#            PyTuple_SetItem(tuple, 0, PyFloat_FromDouble(pt.hessian));
#            
#            PyList_SetItem(list,i,tuple);
#            //printf("%5d %10.5f %10.5f %5d %10.5f %10.5f\\n", i, pt.pt.x, pt.pt.y, pt.size, pt.dir, pt.hessian);
#        
#        
#        cvClearMemStorage(storage);
#        cvReleaseMemStorage(&storage);
#        
#        return_val = list;
#        ''',
#        arg_names=['mat','min_hessian'],
#        include_dirs=['/usr/local/include'],
#        headers=['<opencv/cv.h>'],
#        library_dirs=['/usr/local/lib'],
#        libraries=['cv']
#    )
    
    #return keypoints


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
