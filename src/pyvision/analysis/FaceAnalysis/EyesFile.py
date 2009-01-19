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

from os.path import basename,splitext
from pyvision.types.Point import Point
from pyvision.types.Rect import Rect,BoundingRect

##
# Parses and interfaces with an eyes file.  Eyes files can be comma separated 
# (.csv) files or white space separated files.
# 
# <p>Each line of the file should have an image name and then the left X, 
# left Y, right X, and right Y.  Extensions will be automatically striped from 
# image filenames.
# 
# <p>Example for csv:
# <pre>
# IMG392.jpg,100,122,153,120
# </pre>
#
# <p>Example for whitespace delemeted:
# <pre>
# IMG392.jpg 100 122 153 120
# </pre>
#
# @param the filename for the eye coordinate file.
class EyesFile:
    
    def __init__(self,filename):
        '''
        Inits and reads in the data.
        '''
        self.filename = filename
        self.images = {}
        
        self._readEyesFile()
    
    ##
    # @return a list of all image filenames 
    def files(self):
        '''
        Returns the list of file names.
        '''
        names = self.images.keys()
        names.sort()
        return names
    
    def hasFile(self,filename):
        '''@returns True if filename is in index or False otherwise'''
        fname = self._parseName(filename)
        return self.images.has_key(fname)
    
    ##
    # Returns the eye coordinates given a face detection rectangle.  This is 
    # useful if you have a face detector and want to simulate eye detection
    #
    # @param filename image filename
    # @param rect rectangle that bounds the face that you want eye coordinates.
    # @return [image_name, leye, reye, face_rect] or None if the rectangle is
    #         not near a face.
    def findFace(self,filename,rect):
        fname = self._parseName(filename)
        if self.images.has_key(fname):
            faces = self.images[fname]
            for each in faces:
                truth_rect = each[3]
                if is_success(truth_rect,rect):
                    return each
        return None

    ##
    # Get a list of face bounding boxes.
    #
    # @param filename name of the image file
    # @return list of face rectangles
    def getFaces(self,filename):
        fname = self._parseName(filename)
        if self.images.has_key(fname):
            faces = self.images[fname]
            boxes = []
            for img,left,right,box in faces:
                boxes.append(box)
            return boxes
        return []
       
    ##
    # Get a list of eye points.
    #
    # @param filename name of the image file
    # @returns a list of eye coordinates: [(leye,reye),(leye,reye),...]
    def getEyes(self,filename):
        fname = self._parseName(filename)
        if self.images.has_key(fname):
            faces = self.images[fname]
            eyes = []
            for img,left,right,box in faces:
                eyes.append([left,right])
            return eyes
        return []

    def _readEyesFile(self):
        '''
        Private: Do not call directly. Reads the eye file.  
        '''
        if self.filename[-4:] == '.csv':
            f = open(self.filename,'r')
            for line in f:
                #print line
                line = line.split(',')
                if len(line) < 5:
                    continue
                for i in range(1,len(line),4):
                    fname = self._parseName(line[0])
                    if len(line) < i+4:
                        print "Warning in %s image %s: Count of numbers is not a multiple of four."%(self.filename,fname)
                        break
                    eye1 = Point(float(line[i+0]),float(line[i+1]))
                    eye2 = Point(float(line[i+2]),float(line[i+3]))
                    
                    truth_rect = BoundingRect(eye1,eye2)
                    truth_rect.w = 2.0 * truth_rect.w
                    truth_rect.h = truth_rect.w
                    truth_rect.x = truth_rect.x - 0.25*truth_rect.w
                    truth_rect.y = truth_rect.y - 0.3*truth_rect.w
        
                    #print fname,eye1,eye2,truth_rect
                    
                    if not self.images.has_key(fname):
                        self.images[fname] = []
                        
                    self.images[fname].append([fname,eye1,eye2,truth_rect])
            
        else:
            f = open(self.filename,'r')
            for line in f:
                #print line,
                line = line.split()
                fname = self._parseName(line[0])
                eye1 = Point(float(line[1]),float(line[2]))
                eye2 = Point(float(line[3]),float(line[4]))
                
                truth_rect = BoundingRect(eye1,eye2)
                truth_rect.w = 2.0 * truth_rect.w
                truth_rect.h = truth_rect.w
                truth_rect.x = truth_rect.x - 0.25*truth_rect.w
                truth_rect.y = truth_rect.y - 0.3*truth_rect.w
    
                #print fname,eye1,eye2,truth_rect
                
                if not self.images.has_key(fname):
                    self.images[fname] = []
                    
                self.images[fname].append([fname,eye1,eye2,truth_rect])
            
    def _parseName(self,fname):
        '''
        Private: Do not call directly.  Parses the base filename.
        '''
        fname = basename(fname)
        fname = splitext(fname)[0]
        return fname
    
