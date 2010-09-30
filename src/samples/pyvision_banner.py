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

import os.path
from PIL.Image import composite,LINEAR
import pyvision as pv
#from pyvision.edge.canny import canny
from pyvision.point.DetectorSURF import DetectorSURF
import cv

if __name__ == '__main__':
    ilog = pv.ImageLog()
    source_name = os.path.join(pv.__path__[0],'data','misc','p5240019.jpg')
    im = pv.Image(source_name)
    im = pv.AffineScale(0.25,(320,240)).transformImage(im)
    #im.show()
    ilog.log(im)    
    
    im.show(window='Input', pos=(0,0))
    
    mat = im.asMatrix2D()
    high = mat > 180
    low = mat < 50
    mask = high#+low
    
    #edges = canny(im,100,200)   #canny crashes in opencv2.1 on linux64...
        
    #sobel edge detector
    #sobel requires a destination image with larger bit depth... 
    #...so we have to convert it back to 8 bit for the pv Image...
    cvim_bw = im.asOpenCVBW()
    dst = cv.CreateImage((320,240), 8, 1)
    dst32f = cv.CreateImage((320,240),cv.IPL_DEPTH_32F,1) 
    cv.Sobel(cvim_bw, dst32f, 1, 0, 3)
    cv.Convert(dst32f, dst)    
    edges = pv.Image(dst)
    
    #edges.show()
    edges.show(window='Edges', pos=(360,0))
    
    ilog.log(edges)    
    ilog.log(pv.Image(1.0*mask))
    
    e = edges.asPIL().convert('RGB')
    m = pv.Image(1.0*mask).asPIL()
    i = im.asPIL()
    logo = pv.Image(composite(i,e,m))
    ilog.log(logo)
    
    logo.show(window='Composite', pos=(0,250) )
    
    sm = pv.Image(im.asPIL().resize((320,240),LINEAR))
    detector = DetectorSURF()    
    points = detector.detect(sm)
    for score,pt,radius in points:
        logo.annotateCircle(pt*4,radius*4)
    
    #logo.show()
    logo.show(window='Annotated',pos=(360,250))
    
    cv.WaitKey(0)
    
    ilog.log(logo)
    ilog.show()  #in linux, need a previewer that supports input list of images...

    
    