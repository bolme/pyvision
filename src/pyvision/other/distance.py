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


from math import sqrt
import numpy as np


# Create a table that can look up the hamming distance for each byte
hamming_table = np.zeros(256,dtype=np.int32)
for i in range(256):
    bits = (i&1>0) + (i&2>0) + (i&4>0) + (i&8>0) + (i&16>0) + (i&32>0) + (i&64>0) + (i&128>0)
    hamming_table[i] = bits  
    

def boolToUbyte(x):
    "Convert a boolean vector to a ubyte vector which is much more space efficient."
    assert isinstance(x,np.ndarray)
    assert x.dtype in [np.bool]
    assert len(x.shape) == 1
    assert x.shape[0] % 8 == 0
    
    out =   1*x[7::8] + \
            2*x[6::8] + \
            4*x[5::8] + \
            8*x[4::8] + \
           16*x[3::8] + \
           32*x[2::8] + \
           64*x[1::8] + \
          128*x[0::8]
    
    out = out.astype(np.ubyte)
    return out

def ubyteToBool(x):
    "Convert a byte vector to a bool vector."
    assert isinstance(x,np.ndarray)
    assert x.dtype in [np.ubyte]
    assert len(x.shape) == 1
    
    out = np.zeros(x.shape[0]*8,dtype=np.bool)
    
    out[7::8] = 1&x > 0
    out[6::8] = 2&x > 0
    out[5::8] = 4&x > 0
    out[4::8] = 8&x > 0
    out[3::8] = 16&x > 0
    out[2::8] = 32&x > 0
    out[1::8] = 64&x > 0
    out[0::8] = 128&x > 0
    
    return out
          
    
    

def hamming(a,b):
    if a.dtype == np.bool and b.dtype == bool:
        return (a ^ b).sum()
    elif a.dtype == np.ubyte and b.dtype == np.ubyte:
        return hamming_table[a^b].sum()
    else:
        raise NotImplementedError("Unsupported array types %s and %s",a.dtype,b.dtype)

def l1(a,b):
    ''' Compute the l1 distance measure '''
    return abs(a - b).sum()

def l2(a,b):
    ''' compute the l2 distance '''
    d = (a - b)
    return sqrt( (d*d).sum() )
    
def correlation(a,b):
    ''' Compute the correlation of two vectors '''
    #mean subtract
    a = a - a.mean()
    b = b - b.mean()

    #unit length - avoid dev by zero
    a = (1.0/sqrt((a*a).sum()+0.000001))*a
    b = (1.0/sqrt((b*b).sum()+0.000001))*b
                
    # correlation
    c = (a*b).sum()
    
    return c
