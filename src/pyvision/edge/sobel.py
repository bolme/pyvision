'''
Created on Oct 18, 2010
@author: Stephen O'Hara
'''
import cv
import pyvision as pv

def sobel(im,xorder=1,yorder=0,aperture_size=3,sigma=None):
    '''
    void cv.Sobel(src, dst, xorder, yorder, apertureSize = 3) 
    @param im: Input image
    @param xorder: The order of the x derivative (see cv.Sobel openCV docs) 
    @param yorder: The order of the y derivative (see cv.Sobel openCV docs)
    @param aperture_size: How large a convolution window to use
    @param sigma: Optional smoothing parameter to be applied prior to detecting edges
    '''
    gray = im.asOpenCVBW()
    edges = cv.CreateImage(cv.GetSize(gray), 8, 1)
    
    if sigma!=None:
        cv.Smooth(gray,gray,cv.CV_GAUSSIAN,int(sigma)*4+1,int(sigma)*4+1,sigma,sigma)

    #sobel requires a destination image with larger bit depth... 
    #...so we have to convert it back to 8 bit for the pv Image...
    dst32f = cv.CreateImage(cv.GetSize(gray),cv.IPL_DEPTH_32F,1) 
    
    cv.Sobel(gray, dst32f, xorder, yorder, aperture_size)
    cv.Convert(dst32f, edges)    
    edges = pv.Image(edges)
        
    return edges
    

        
