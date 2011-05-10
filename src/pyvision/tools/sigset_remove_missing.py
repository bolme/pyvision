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


import os.path


IMAGE_EXTENSIONS = set(['.TIF','.TIFF','.JPG','.JPEG','.PNG','.GIF','.PPM','.PGM','.BMP'])



def parseBasename(path):
    '''
    This section parses the basename without file extensions from the path.
    
    @returns basename from /path/to/file/basename.eee
    '''
    if path == None:
        return None
    
    dirname,filename = os.path.split(path)
    # Adaptation for FERET sigsets
    basename = filename.split('.')[0]
    #basename,extname = os.path.splitext(filename)
    return basename




def locateFiles(sigset,imdir):
    '''
    This function scans the image directories for the files listed in the
    sigset.  If using the reduced image format, the images should be named
    using the recording ID with a standard image extension such as "jpg".
    Otherwise, the directories are scanned for images matching the basename
    of the filename.  This means that the paths in the sigset do not have
    to be specified accuratly but problems do arise if multiple copies
    of an image are in the image directory.
    '''
    image_map = {}
    file_map = {}
    rec_map = {}

    
    for each in sigset:
        rec_id = each[1][0]['name']
        basename = parseBasename(each[1][0]['file-name'])
        image_map[rec_id] = None
        file_map[basename] = rec_id
        rec_map[rec_id] = basename
        
            
    n_images = 0
    
    for rootdir,dirs,files in os.walk(imdir):
        for filename in files:
            basename,ext = os.path.splitext(filename)
            if ext.upper() not in IMAGE_EXTENSIONS:
                continue
            print filename
            if file_map.has_key(basename) and image_map[file_map[basename]] == None:
                image_map[file_map[basename]] = os.path.join(rootdir,filename)
                n_images += 1
            elif file_map.has_key(basename) and image_map[file_map[basename]] != None:
                raise ValueError("Multiple images found matching recording id %s:\n   First instance:  %s\n   Second instance: %s"%(file_map[basename],image_map[file_map[basename]],os.path.join(rootdir,filename)))
            if rec_map.has_key(basename) and rec_map[file_map[basename]] == None:
                image_map[file_map[basename]] = os.path.join(rootdir,filename)
                n_images += 1
            elif rec_map.has_key(basename) and image_map[file_map[basename]] != None:
                raise ValueError("Multiple images found matching recording id %s:\n   First instance:  %s\n   Second instance: %s"%(file_map[basename],image_map[file_map[basename]],os.path.join(rootdir,filename)))
                    
                    
    if True: print "Found %d of %d images."%(n_images,len(image_map))
    
    missing = []
    found = []
    for item in sigset:
        rec_id = item[1][0]['name']
        filename = image_map[rec_id]
        if filename == None:
            missing.append(item)
        else:
            found.append(item)
            
    
    return found,missing




def parseOptions():
    usage = "usage: %prog [options] <input.xml> <image_directory> <found.xml> [<missing.xml>]\nReads a sigset and removes any entries that cannot be associated with an image."
    
    parser = optparse.OptionParser(usage)
    #parser.add_option("-v", "--verbose",
    #                  action="store_true", dest="verbose",
    #                  help="Turn on more verbose output.")
    (options, args) = parser.parse_args()
    
    if len(args) not in [3,4]:
        parser.error("This program requires at least three arguments: an input sigest, an image directory, and an output sigset.")

    return options, args



if __name__ == '__main__':
    options,args = parseOptions()

    sigset = bee.parseSigSet(args[0])
    imdir = args[1]
    found,missing = locateFiles(sigset,imdir)
    bee.saveSigset(found, args[2])
    if len(args) >= 4:
        bee.saveSigset(missing, args[3])
    
    
    
