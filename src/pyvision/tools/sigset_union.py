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
'''
Created on May 10, 2011

@author: bolme

Compute the union of two sigsets.
'''
import pyvision as pv
import optparse
import csv
import os.path


def parseOptions():
    usage = "usage: %prog [options] <SIGSET_A.xml> <SIGSET_B.xml> <union.xml>\nReads in two sigsets and computes the union as output."
    
    parser = optparse.OptionParser(usage)
    #parser.add_option("-v", "--verbose",
    #                  action="store_true", dest="verbose",
    #                  help="Turn on more verbose output.")
    (options, args) = parser.parse_args()
    
    if len(args) < 3:
        parser.error("This program requires two sigset arguments.")

    return options, args



if __name__ == '__main__':
    # Parse command line arguments
    options,args = parseOptions()

    
    # Compute the set of recordings and subjects in A
    both = []
    both_rec = set()
    for arg in args[:-1]:
        # Read sigsets
        sigset = pv.parseSigSet(arg)
        
        # Compute union
        for row in sigset:
            rec_id = row[1][0]['name']
            if rec_id in both_rec:
                continue
            both.append(row)
            both_rec.add(rec_id)
            
    pv.saveSigset(both, args[-1])
            
    
        
    
    
    
    
    
