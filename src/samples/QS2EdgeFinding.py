# PyVision License
#
# Copyright (c) 2006-2010 David S. Bolme
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
from pyvision.edge.canny import canny  # An interface to the OpenCV Canny.
from scipy.ndimage import gaussian_filter # scipy gaussian filter function
from numpy import arange;

#This code is from part 2 of the PyVision Quick Start Guide.

if __name__ == '__main__':
    # Create the image log
    ilog = pv.ImageLog("../..")

    # Load an image from a file.
    im = pv.Image(pv.__path__[0]+"/data/nonface/NONFACE_16.jpg")
    
    # Rescale the image
    im = pv.AffineScale(0.5,(320,240)).transformImage(im)
    ilog.log(im,label="Source")    
    
    # Try a range of sigmas
    for sigma in arange(1.0,5.1,0.5):
        
        # Perform a Gaussian Blur
        mat = im.asMatrix2D()
        mat = gaussian_filter(mat,sigma)
        blur = pv.Image(mat)
        blur.annotateLabel(pv.Point(10,10),"Sigma: " + str(sigma))
        ilog.log(blur,label="Blur")    

        #Try a range of thresholds
        for thresh in arange(50,150,10):
    
            # Run the canny function with different defaults.
            edge = canny(blur,threshold1=thresh/2,threshold2=thresh)
            
            # Annotate the edge image
            edge.annotateLabel(pv.Point(10,10),"Sigma: " + str(sigma))
            edge.annotateLabel(pv.Point(10,20),"Thresh: " + str(thresh))
    
            # Save the results to a log.
            ilog.log(edge,label="Canny")
    
    # Display the results.
    ilog.show()
    
    
    