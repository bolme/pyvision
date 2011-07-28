# PyVision License
#
# Copyright (c) 2011 David S. Bolme
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
'''
Created on Jan 15, 2011

@author: bolme
'''

import pyvision as pv
import numpy as np
import time

LBP_CLASSIC = [
            [-1.0,-1.0],
            [ 0.0,-1.0],
            [ 1.0,-1.0],
            [ 1.0, 0.0],
            [ 1.0, 1.0],
            [ 0.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 0.0],
            ]

LBP_RAD1 = np.array([[0.0, 1.0], [0.70710678118654746, 0.70710678118654757], [1.0, 6.123233995736766e-17], [0.70710678118654757, -0.70710678118654746], [1.2246467991473532e-16, -1.0], [-0.70710678118654746, -0.70710678118654768], [-1.0, -1.8369701987210297e-16], [-0.70710678118654768, 0.70710678118654746]])
LBP_RAD2 = 2.0*LBP_RAD1
LBP_RAD3 = 3.0*LBP_RAD1
LBP_RAD4 = 4.0*LBP_RAD1
LBP_RAD8 = 8.0*LBP_RAD1



def lbp(im,pattern=LBP_CLASSIC):
    im = pv.Image(im.asOpenCVBW()) #TODO: Use opencv for speed
    
    mat = im.asMatrix2D()
    lbp = np.zeros(mat.shape,dtype=np.uint8)

    w,h = mat.shape
    
    bit = 1
    for dx,dy in pattern:
        affine = pv.AffineTranslate(-dx,-dy,(w,h))
        mat2 = affine.transformImage(im).asMatrix2D()
                
        lbp += bit*(mat < mat2)
        
        bit = bit * 2
        
    return lbp
         
         
