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
from Image import composite,LINEAR
import pyvision as pv
from pyvision.edge.sobel import sobel
#from pyvision.edge.canny import canny
from pyvision.point.DetectorSURF import DetectorSURF
import cv

if __name__ == '__main__':
    ilog = pv.ImageLog()
    source_name = os.path.join(pv.__path__[0],'data','misc','p5240019.jpg')

    #Load source image and resize to smaller scale
    im = pv.Image(source_name)
    print "Size before affine scale: %s"%str(im.size)
    im = pv.AffineScale(0.25,(320,240)).transformImage(im)
    print "Size after scaling: %s"%str(im.size)
    ilog.log(im, 'Input')    
    #im.show(window='Input', pos=(0,0))
            
    #Generate edge image using sobel edge detector
    edges = sobel(im, 1, 0 , 3, 0)
    ilog.log(edges, 'Edges')  
    #edges.show(window='Edges', pos=(360,0))
    
    #Generate threshold mask, shows numpy integration
    mat = im.asMatrix2D()
    high = mat > 180
    low = mat < 50
    mask = high#+low
    ilog.log(pv.Image(1.0*mask), 'Mask')
    
    #Composite operation using PIL
    e = edges.asPIL().convert('RGB')
    m = pv.Image(1.0*mask).asPIL()
    i = im.asPIL()
    logo = pv.Image(composite(i,e,m))
    ilog.log(logo, 'Composite')    
    #logo.show(window='Composite', pos=(0,300) )
    
    #Keypoint detection using OpenCV's SURF detector
    logo_surf = logo.copy()
    sm = pv.Image(im.asPIL().resize((320,240),LINEAR))
    detector = DetectorSURF()    
    points = detector.detect(sm)
    for score,pt,radius in points:
        logo_surf.annotateCircle(pt*4,radius*4)
    ilog.log(logo_surf, 'Annotated')
    #logo_surf.show(window='Annotated',pos=(360,300))
    
    #Demonstrate use of ImageMontage class to show a few small images in a single window
    print "Have the image montage focused in UI and hit spacebar to continue..."
    imontage = pv.ImageMontage([im,edges,logo,logo_surf], layout=(2,2), tileSize=im.size, gutter=3, byrow=True, labels=None)
    imontage.show(window="Image Montage", delay=0)
    
    #Show the images stored to the image log object
    print "Showing image log. These images are stored in a tmp directory."
    ilog.show()

    
    