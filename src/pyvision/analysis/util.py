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

from numpy import array

import logging
import sys

pyvis_log = logging.getLogger("pyvis")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(name)s:%(levelname)-5s %(filename)15s:%(lineno)-5d - %(message)s")
handler.setFormatter(formatter)
pyvis_log.addHandler(handler)



    
def histogram(data, bins=1.0):
    ''' create a histogram of the data '''
    tmp = []
    for each in data:
        tmp.append([each[0],each[1]])
        
    tmp.sort(lambda x,y: cmp(x[0],y[0]) )
    
    min = int(tmp[ 0][0]/bins)
    max = int(tmp[-1][0]/bins) 
    print tmp[-1]
    size = max+1-min
    print size
    
    hist = []
    for i in range(size):
        hist.append([])
        
    for each in tmp:
        id = int(each[0]/bins)-min
        hist[id].append(each[1])
        #print id,each
        
    #for id in range(len(hist)):
    for each in hist:
        #each = hist[id]
        each.sort()
        min = each[0]
        select05 = int(0.05*len(each))
        select05 = each[select05]
        select25 = int(0.25*len(each))
        select25 = each[select25]
        select50 = int(0.5*len(each))
        select50 = each[select50]
        select75 = int(0.75*len(each))
        select75 = each[select75]
        select95 = int(0.95*len(each))
        select95 = each[select95]
        max = each[-1]
        
        #print each
        print "%f\t%f\t%f\t%f\t%f\t%f\t%f"%(min,select05,select25,select50,select75,select95,max)
    