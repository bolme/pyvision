'''
Created on Feb 2, 2011

@author: bolme
'''

import pyvision as pv
import cv

def surf(im,mask=None,extended=False,hessianThreshold=500, nOctaves=3, nOctaveLayers=4):
    '''
    @param im: image to extract features from.
    @type im:  pv.Image
    @param mask: a mask that controls where features are extracted from.
    @type mask:  OpenCV 8bit image
    @return: (keypoints,descriptors)
    '''
    cvim = im.asOpenCVBW()
    keypoints,descriptors = cv.ExtractSURF(cvim,mask,cv.CreateMemStorage(),(int(extended),hessianThreshold,nOctaves,nOctaveLayers))
    return keypoints,descriptors