'''
Copyright 2013 Oak Ridge National Laboratory
Created on Jun 3, 2013

@author: David S. Bolme
'''

import pyvision as pv

def skintone(im,low_thresh=0.02511,high_thresh=0.1177):
    '''
    Returns a mask showing pixels classified as skin tone.
    
    Based on: 
    Cheddad et.al. "A new colour space for skin tone detection" ICIP. 2009.
    '''
    mat = im.asMatrix3D()
    r,g,b = mat/255.0
    I = 0.298936021*r + 0.5870430*g + 0.14020904255*b
    Ip = (g>b)*g + (b>=g)*b
    #pv.Image(I).show()
    #pv.Image(Ip).show()
    
    e = I - Ip
    #pv.Image(e).show()
    mask = (e > low_thresh) & (e < high_thresh) & (g > 0.05)  & (g < 0.98)
    #mask = (e > low_thresh) 
    #print low_thresh,high_thresh,mask.sum(),e.min(),e.max()
    #mask = (e < high_thresh)
    
    #mask = g > 0.5
    
    return mask 

if __name__ == '__main__':
    video = pv.VideoFromDirectory("/data/retrieval/test_images/people/")
    for im in video: 
        #for lt in [0.0,0.02,0.04,0.08,0.10,0.12]:
        mask = skintone(im,low_thresh=.03,high_thresh=.10)
        tmp = im.copy()
        tmp.annotateMask(mask)
        tmp.show()