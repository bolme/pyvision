# PyVision License
#
# Copyright (c) 2006-2008 David S. Bolme
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


#TODO: update to numpy
#import numarray
#from numarray import mlab, zeros, ones
#from numarray.nd_image import maximum_filter
from math import *
import unittest

DEFAULT_THRESHOLD = 23
DEFAULT_FILTER = (5,5)

DEFAULT_C_THRESHOLD = 0.75
DEFAULT_C_FILTER = 40


def computeMedian(image_list):
    '''This algorithm computes the median image of a set of images'''
    images = numarray.array(image_list)
    print images.shape
    median = mlab.median(images)
    print median.shape
    return median
    
    
def computeMean(image_list):
    '''This algorithm computes the median image of a set of images'''
    images = numarray.array(image_list)
    print images.shape
    mean = mlab.mean(images)
    print mean.shape
    return mean
    
    
def computeChanges(novel, background, threshold = DEFAULT_THRESHOLD, filter_size = DEFAULT_FILTER):
    ''' Givin a novel image and a background image, select any part that is different'''
    changes = abs(novel - background) > threshold
    changes = maximum_filter(changes,size=filter_size)
    return changes

def correlationChanges(novel, background, threshold = DEFAULT_C_THRESHOLD, stdev_thresh = 5.0, filter_size = DEFAULT_C_FILTER):
    ''' Givin a novel image and a background image, select any part that is different'''
    changes = zeros(background.shape,'i')
    w,h = background.shape
    for i in range(0,w,filter_size):
        for j in range(0,h,filter_size):
            n = novel     [i:i+filter_size,j:j+filter_size]
            b = background[i:i+filter_size,j:j+filter_size]
            
            #mean subtract
            n = n - n.mean()
            b = b - b.mean()

            #print n.stddev(), b.stddev()
            if n.stddev() < stdev_thresh and b.stddev() < stdev_thresh:
                continue
            
            
            #unit length
            n = (1.0/sqrt((n*n).sum()+0.0001))*n
            b = (1.0/sqrt((b*b).sum()+0.0001))*b
            
            
            # correlation
            c = (n*b).sum()
            
            if c < threshold:
                changes[i:i+filter_size,j:j+filter_size] = ones(n.shape,'i')

    return changes

#def computeChanges(novel,background):
#    ''' Givin a novel image and a background image, select any part that is different'''
#    changes = abs(novel - background) > 20
#    return changes

# TODO:
class _ChangeDetectionTest(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_simple_changes(self):
        self.assert_(False)
        
    def test_correlation_changes(self):
        self.assert_(False)
        
        
if __name__ == '__main__':
    import sys
    import PIL.Image
    import os
    import convert
    
    image_dir = sys.argv[1]
    image_list = []
    for each in os.listdir(image_dir):
        if each.upper()[-4:] != '.JPG':
            continue
        image = PIL.Image.open(image_dir + '/' + each)
        mat = convert.image_to_matrix(image)
        image_list.append(mat)
    median = computeMedian(image_list)
    im = convert.matrix_to_image(median)
    im.show()
    mean = computeMean(image_list)
    im = convert.matrix_to_image(mean)
    im.show()
    for each in image_list:
        im = convert.matrix_to_image(each)
        im.show()
        changes = computeChanges(each,median)
        im = convert.matrix_to_image(changes)
        im.show()