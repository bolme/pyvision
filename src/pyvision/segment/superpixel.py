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
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.data import lena


def _assignInitialPoints(cvmat,S):
    h,w,c = cvmat.shape
    
    # Compute the max grid assignment
    nx = w/S
    ny = h/S

    # Compute the super pixel x,y grid
    xgrid = np.arange(nx).reshape(1,nx)*np.ones(ny,dtype=np.int).reshape(ny,1)
    ygrid = np.arange(ny).reshape(ny,1)*np.ones(nx,dtype=np.int).reshape(1,nx)
    
    # compute an x,y lookup to a label look up
    label_map = nx*ygrid + xgrid    
    
    # Compute the x groups in pixel space
    tmp = np.arange(nx)
    tmp = np.resize(tmp,(w,))
    xgroups = tmp[tmp.argsort()]

    # Compute the y groups in pixel space
    tmp = np.arange(ny)
    tmp = np.resize(tmp,(h,))
    ygroups = tmp[tmp.argsort()]

    labels = np.zeros((h,w),dtype=np.int)
    
    for x in xrange(w):
        for y in xrange(h):
            labels[y,x] = label_map[ygroups[y],xgroups[x]]

    return label_map,xgroups,ygroups,labels
    
def _computeCentriods(mat,labels,label_map):
    c,h,w = mat.shape
    mat = mat.reshape(c,h*w)
    centroids = np.zeros((label_map.max()+1,c))
    counts = np.zeros((label_map.max()+1,1))
    for i in xrange(label_map.max()+1):
        pass
        mask = (labels == i)
        #print mask.flatten()
        centroids[i,:] = mat[:,mask.flatten()].mean(axis=1)
        #pv.Image(1.0*mask).show()
    return centroids
    for x in xrange(w):
        for y in xrange(h):
            lab = labels[y,x]
            centroids[lab] += mat[:,y,x]
            counts[lab] += 1.0
    centroids = centroids / counts
    return centroids


def _computeLabels(mat,label_map,centroids,xgroups,ygroups,S,m):
    c,h,w = mat.shape
    labels = np.zeros((h,w),dtype=np.int)
    dists = np.zeros((h,w),dtype=np.float32)
    dists[:,:] = np.inf
    for lab in xrange(len(centroids)):
        centroid = centroids[lab]
        x,y = centroid[:2]
        centroid = centroid.reshape(c,1,1)
        minx = max(0,int(x-S))
        maxx = min(w,int(x+S))
        miny = max(0,int(y-S))
        maxy = min(h,int(y+S))
        tile = mat[:,miny:maxy,minx:maxx] 
        ds2 = ((tile[:2,:,:] - centroid[:2,:,:])**2).sum(axis=0)
        dc2 = ((tile[2:,:,:] - centroid[2:,:,:])**2).sum(axis=0)
        
        D = np.sqrt(dc2 + (m*m)*(ds2/(S*S)))
        dist_tile = dists[miny:maxy,minx:maxx]
        labels_tile = labels[miny:maxy,minx:maxx]
        
        mask = (dist_tile > D)
        dist_tile[mask] = D[mask]
        labels_tile[mask] = lab
    return labels

    
def slic2(im,S,m=20,L_scale=0.5,mean_scale=1.0,std_scale=3.0):
    '''
    This is a k-means based super pixel algorithm inspired by slic.
    
    http://ivrg.epfl.ch/supplementary_material/RK_SLICSuperpixels/index.html
    '''
    cvmat = im.asOpenCV2()
    
    # Compute a label map and assign initial labels
    label_map,xgroups,ygroups,labels = _assignInitialPoints(cvmat,S)
    print label_map
    
    # Compute color features
    mat = cv2.cvtColor(cvmat,cv2.cv.CV_BGR2Lab)
    h,w,c = cvmat.shape
    
    # Compute location features
    x = np.arange(w).reshape(1,w)*np.ones(h,dtype=np.int).reshape(h,1)
    y = np.arange(h).reshape(h,1)*np.ones(w,dtype=np.int).reshape(1,w)
    
    # Scale the features
    mat = mat
    
    # Compute local statistics
    mean_L = mat[:,:,0].copy()
    mean_L = ndi.gaussian_filter(mean_L,0.5*S)
    std_L = (mat[:,:,0].copy() - mean_L)**2
    std_L = np.sqrt(ndi.gaussian_filter(std_L,0.5*S))

    mean_a = mat[:,:,1].copy()
    mean_a = ndi.gaussian_filter(mean_a,0.5*S)
    std_a = (mat[:,:,1].copy() - mean_a)**2
    std_a = np.sqrt(ndi.gaussian_filter(std_a,0.5*S))

    mean_b = mat[:,:,2].copy()
    mean_b = ndi.gaussian_filter(mean_b,0.5*S)
    std_b = (mat[:,:,2].copy() - mean_b)**2
    std_b = np.sqrt(ndi.gaussian_filter(std_b,0.5*S))

    # Create a feature vector matrix
    features = np.array([x,y, mat[:,:,0],mat[:,:,1],mat[:,:,2],])
    #features = np.array([x,y, L_scale*mat[:,:,0],mat[:,:,1],mat[:,:,2],mean_scale*mean_L,std_scale*std_L,mean_scale*mean_a,std_scale*std_a,mean_scale*mean_b,std_scale*std_b])
    
    for i in range(10):
        # Compute centriods
        timer = pv.Timer()
        centroids = _computeCentriods(features,labels,label_map)
        timer.mark('centroids')
        labels = _computeLabels(features,label_map,centroids,xgroups,ygroups,S,m)
        timer.mark('labels')
    mask = 9*labels != ndi.correlate(labels, [[1,1,1],[1,1,1],[1,1,1]])
    return labels.T,centroids,mask.T

if __name__ == '__main__':
    ilog = pv.ImageLog()
    
    #im = pv.Image(np.array(lena(),dtype=np.float32))
    #print lena()
    #im.show()
    #assert 0
    #im = im.resize((256,256))
    #labels,centroids,mask = slic(im,10,50.0)
    #segments = felzenszwalb(im.asOpenCV2(),scale=100, sigma=0.5, min_size=50)
    for im in images:
        #im = im.resize((256,256))
        segments = slic(im.asOpenCV2(),ratio=80, n_segments=16, sigma=2)
        mask = 9*segments != ndi.correlate(segments, [[1,1,1],[1,1,1],[1,1,1]])
        im.annotateMask(mask.T)
        im.show()
        
        
        
        
        
        
        