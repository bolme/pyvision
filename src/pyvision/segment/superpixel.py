'''
Created on May 24, 2013

@author: bolme
'''
import cv2
import scipy.cluster.vq as vq
import scipy as sp
import scipy.ndimage as ndi
import pyvision as pv
import numpy as np


def slic(im,k,blur=7,color_weight=0.30):
    '''
    This is a k-means based super pixel algorithm inspired by slic.
    
    http://ivrg.epfl.ch/supplementary_material/RK_SLICSuperpixels/index.html
    '''
    cvmat = im.asOpenCV2()
    cvmat = cv2.blur(cvmat, (blur,blur))
    mat = color_weight*cvmat
    r,c,_ = mat.shape
    w,h = im.size
    rows = np.arange(r).reshape(r,1)*np.ones((1,c))
    cols = np.arange(c).reshape(1,c)*np.ones((r,1))
    mat = np.array([mat[:,:,0],mat[:,:,1],mat[:,:,2],rows,cols])
    mat = mat.swapaxes(0,2)
    mat = mat.reshape(r*c,5)
    centroids,labels = vq.kmeans2(mat, k, minit = 'points')
    labels.shape = (c,r)

    mask = 9*labels != ndi.correlate(labels, [[1,1,1],[1,1,1],[1,1,1]])
    
    return labels,mask

if __name__ == '__main__':
    im = pv.Image(pv.FRUITS )
    #im = im.resize((100,100))
    labels = slic(im,300)
    pv.AffineTranslate(1,0,im.size)
    pv.Image(np.float32(labels)).show()
    im.show()