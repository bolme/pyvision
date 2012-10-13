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
import os.path

    
def Example1(ilog):
    
    # This illustrates how images keep weak refs through affine 
    # transformation to avoid an accumulation of errors through multiple 
    # transformations.
    fname = os.path.join(pv.__path__[0],'data','nonface','NONFACE_13.jpg')
    im = pv.Image(fname)
    im = pv.Image(im.asPIL().resize((320,240)))
    im.annotateLabel(pv.Point(10,10), "ScaleTest: Original image.")
    ilog.log(im)
    w,h = im.size
    
    # Create a small version of the image
    aff_small = pv.AffineScale(0.2,(w/5,h/5))
    tmp1 = aff_small.transformImage(im)
    tmp1.annotateLabel(pv.Point(10,10), "Small")        
    ilog.log(tmp1)
    
    # Scale the image back to its original size without using the original
    aff_big = pv.AffineScale(5.0,(w,h))
    tmp2 = aff_big.transformImage(tmp1,use_orig=False)
    tmp2.annotateLabel(pv.Point(10,10), "ScaleTest: use_orig=False")
    tmp2.annotateLabel(pv.Point(20,20), "This image should be blurry.")
    ilog.log(tmp2)
   
    # Use the affine class to produce a transform that maps the original
    # directly to the final image and therefore keeps most of the detail.
    tmp3 = (aff_big*aff_small).transformImage(im)
    tmp3.annotateLabel(pv.Point(10,10), "ScaleTest: aff_big*aff_small")
    tmp3.annotateLabel(pv.Point(20,20), "This image should be sharp.")
    ilog.log(tmp3)
   
    # Scale the image back to its original size using the original
    tmp4 = aff_big.transformImage(tmp1,use_orig=True)
    tmp4.annotateLabel(pv.Point(10,10), "ScaleTest: use_orig=True")
    tmp4.annotateLabel(pv.Point(20,20), "This image should be sharp.")
    ilog.log(tmp4)
    
    # Now remove the reverence to the im instance.  The weak references within
    # tmp1 do not hold onto the original data so now there is no choice but to
    # use the scaled down image.
    del im    
    tmp5 = aff_big.transformImage(tmp1,use_orig=True)
    tmp5.annotateLabel(pv.Point(10,10), "ScaleTest: use_orig=True")
    tmp5.annotateLabel(pv.Point(20,20), "This image should be blurry")
    tmp5.annotateLabel(pv.Point(20,30), "because the original has be")
    tmp5.annotateLabel(pv.Point(20,40), "removed from memory.")
    ilog.log(tmp5)
    
    # Weak references are used to prevent python from hanging onto larger images
    # when the user deletes there references or they go out of scope.  To 
    # use the original images further down the image pipeline make sure that
    # they are expressly kept around.  This feature is designed as a convenience
    # so that the user does not have to keep track of many affine transforms and
    # images, and can accumulate transforms as if each one goes all the way back
    # to the original image if it is still available.  It also does not interfere
    # memory management if the user decides to drop references to large images.
    

if __name__ == '__main__':
    ilog = pv.ImageLog()
    
    Example1(ilog)
        
    ilog.show()