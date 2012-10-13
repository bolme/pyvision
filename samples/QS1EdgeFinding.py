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

'''
This code is from part 1 of the PyVision Quick Start Guide.
'''
if __name__ == '__main__':
    # (1) Load an image from a file.
    im = pv.Image(pv.__path__[0]+"/data/nonface/NONFACE_16.jpg")
    
    # (2) Rescale the image
    im = pv.AffineScale(0.5,(320,240)).transformImage(im)
    
    # (3) Run the canny function to locate the edges.
    edge_im1 = canny(im)
    
    # (4) Run the canny function with different defaults.
    edge_im2 = canny(im,threshold1=100,threshold2=250)
    
    # (5) Save the results to a log.
    ilog = pv.ImageLog("../..")
    ilog.log(im,label="Source")    
    ilog.log(edge_im1,label="Canny1")
    ilog.log(edge_im2,label="Canny2")
    
    # (6) Display the results.
    ilog.show()
    
    
    