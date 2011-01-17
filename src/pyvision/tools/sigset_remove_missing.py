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
Created on Jan 14, 2011

@author: bolme

Convert a simple sigset into a comma sep value file.  Does not currently support complex sigests.
'''
#import pyvision as pv
import pyvision.analysis.FaceAnalysis.BEE as bee
import optparse
import csv



def parseOptions():
    usage = "usage: %prog [options] <input.xml> <image_directory> <output.xml>\nReads a sigset and removes any entries that cannot be associated with an image."
    
    parser = optparse.OptionParser(usage)
    #parser.add_option("-v", "--verbose",
    #                  action="store_true", dest="verbose",
    #                  help="Turn on more verbose output.")
    (options, args) = parser.parse_args()
    
    if len(args) != 3:
        parser.error("This program requires three arguments: an input sigest, an image directory, and an output sigset.")

    return options, args



if __name__ == '__main__':
    options,args = parseOptions()

    sigset = bee.parseSigSet(args[0])
    
    rows = []
    header = ["sub_id","rec_id","filename","modality","file_format"]
    rows.append(header)
    for each in sigset:
        sub_id = each[0]
        rec_id = each[1][0]['name']
        filename = each[1][0]['file-name']
        modality = each[1][0]['modality']
        file_format = each[1][0]['file-format']
        row = [sub_id,rec_id,filename,modality,file_format]
        rows.append(row)

    f = csv.writer(open(args[1],'wb'))
    f.writerows(rows)
    
    
    
