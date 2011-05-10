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

Reads in two sigsets and computes overlap statistics.
'''
import pyvision as pv
import optparse
import csv
import os.path


def parseOptions():
    usage = "usage: %prog [options] <SIGSET_A.xml> <SIGSET_B.xml> \nReads in two sigsets and computes overlap statistics."
    
    parser = optparse.OptionParser(usage)
    #parser.add_option("-v", "--verbose",
    #                  action="store_true", dest="verbose",
    #                  help="Turn on more verbose output.")
    (options, args) = parser.parse_args()
    
    if len(args) not in [2]:
        parser.error("This program requires two sigset arguments.")

    return options, args



if __name__ == '__main__':
    # Parse command line arguments
    options,args = parseOptions()
    sigset_a = args[0]
    sigset_b = args[1]

    # Read in sigsets
    a = pv.parseSigSet(sigset_a)
    b = pv.parseSigSet(sigset_b)
    
    # Compute the set of recordings and subjects in A
    sub_a = []
    rec_a = []
    for row in a:
        sub_id = row[0]
        rec_id = row[1][0]['name']
        sub_a.append(sub_id)
        rec_a.append(rec_id)
    sub_a = set(sub_a)
    rec_a = set(rec_a)
    
    # Compute the set of recordings and subjects in B
    sub_b = []
    rec_b = []
    for row in b:
        sub_id = row[0]
        rec_id = row[1][0]['name']
        sub_b.append(sub_id)
        rec_b.append(rec_id)
    sub_b = set(sub_b)
    rec_b = set(rec_b)
    

    # Print overlap statistics
    print "Total Subjects A:  ",len(sub_a)
    print "Total Subjects B:  ",len(sub_b)
    print "Common Subjects:   ",len(sub_a&sub_b)
    print "Unique Subjects A: ",len(sub_a-sub_b)
    print "Unique Subjects B: ",len(sub_b-sub_a)
    print "Total Recordings A:",len(rec_a)
    print "Total Recordings B:",len(rec_b)
    print "Common Recordings: ",len(rec_a&rec_b)
    print "Unique Recordings A:",len(rec_a-rec_b)
    print "Unique Recordings B:",len(rec_b-rec_a)
    
        
    
    
    
    
    
