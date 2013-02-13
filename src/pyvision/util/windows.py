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

import numpy as np

def hammingWindow(size):
    '''
    Windowing function from:
    http://en.wikipedia.org/wiki/Window_function
    '''
    w,h = size
    X = np.arange(w).reshape(w,1)
    Y = np.arange(h).reshape(1,h)
    X = X*np.ones((1,h),'d')
    Y = Y*np.ones((w,1),'d')
    
    window = (5.3836-0.46164*np.cos(2*np.pi*X/(w-1.0)))*(5.3836-0.46164*np.cos(2*np.pi*Y/(h-1.0)))
    return window


def hannWindow(size):
    '''
    Windowing function from:
    http://en.wikipedia.org/wiki/Window_function
    '''
    w,h = size
    X = np.arange(w).reshape(w,1)
    Y = np.arange(h).reshape(1,h)
    X = X*np.ones((1,h),'d')
    Y = Y*np.ones((w,1),'d')
    
    window = (0.5*(1-np.cos(2*np.pi*X/(w-1.0))))*(0.5*(1-np.cos(2*np.pi*Y/(h-1.0))))
    return window

def cosineWindow(size):
    '''
    Windowing function from:
    http://en.wikipedia.org/wiki/Window_function
    '''
    w,h = size
    X = np.arange(w).reshape(w,1)
    Y = np.arange(h).reshape(1,h)
    X = X*np.ones((1,h),'d')
    Y = Y*np.ones((w,1),'d')
    
    window = (np.sin(np.pi*X/(w-1.0)))*(np.sin(np.pi*Y/(h-1.0)))
    return window




