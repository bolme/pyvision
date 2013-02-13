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

import pyvision as pv
import numpy as np

class DetectorROI:
    '''
    This class defines an interface to a Region Of Interest (ROI) detector.
    '''
    def __init__(self,n=250,selector='bins',bin_size=50):
        '''
        n        - is the approximate number of points requested.
        bin_size - the width and height of each bin in pixels.
        selector - ('all', 'bins', or 'best') stratagy for point selection.
        
        When corner_selector is set to bins, the image is subdivided in to bins of
        size <bin_size>X<bin_size> pixels and an equal number of points will be taken
        from each of those bins.  This insures that points are found in all parts of the
        image not just where the corners are strongest.
        '''

        self.n = n
        self.selector = selector
        self.bin_size = bin_size
        pass
    
    def detect(self,image,**kwargs):
        '''
        Returns a list of region of interest. Each element in the list is a 
        tuple of (score,centerpoint,radius). Radius of "None" is used for point 
        detectors.  Higher scores are better and scores of "None" indicate no 
        score is avalible.
        '''
        # TODO: Call subclass
        A = None
        if isinstance(image,pv.Image):
            A = image.asMatrix2D()
        elif isinstance(image,np.array) and len(image.shape)==2:
            A = image
        else:
            raise TypeError("ERROR Unknown Type (%s) - Only arrays and pyvision images supported."%type(image))
    
        L = self._detect(image,**kwargs)
        
        L.sort()
        L.reverse()
        
        if self.selector == 'best':
            L=L[:self.n]
        elif self.selector == 'bins':
            nbins = A.shape[0]/self.bin_size*A.shape[1]/self.bin_size
            npts = self.n / nbins + 1
    
            corners = []
            for xmin in range(0,A.shape[0],self.bin_size):
                xmax = xmin + self.bin_size
                for ymin in range(0,A.shape[1],self.bin_size):
                    bin_data = []
                    ymax = ymin + self.bin_size
                    for each in L:
                        #print each
                        if xmin <= each[1] and each[1] < xmax and ymin <= each[2] and each[2] < ymax:
                            bin_data.append(each)
                            if len(bin_data) >= npts:
                                break
                    corners += bin_data
            L = corners
        else: # TODO: assume all
            pass
            
        roi = []                   
        for each in L:
            roi.append([each[0],pv.Point(each[1],each[2]),each[3]])
            
        #L = concatenate((L.transpose,ones((1,L.shape[0]))))
        return roi

    def _detect(self):
        raise NotImplementedError("This method should be overridden in a sub class.")
    
    
    
    