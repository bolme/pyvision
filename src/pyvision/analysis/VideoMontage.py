'''
Created on Dec 7, 2010
@author: svohara
'''
# PyVision License
#
# Copyright (c) 2006-2008 Stephen O'Hara
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
import cv

class VideoMontage:
    '''
    Provides a visualization of several videos playing back in
    a single window. This can be very handy, for example, to
    show tracking results of multiple objects from a single video,
    or for minimizing screen real-estate when showing multiple
    video sources.
    
    A video montage object is an iterator, so you "play" the
    montage by iterating through all the frames, just as with
    a standard video object.
    '''
    def __init__(self, videoDict, layout, size=(100,100) ):
        '''
        @param videoDict: A dictionary of videos to display in the montage. The keys are the video labels, and 
        the values are objects adhering to the pyvision video interface. (pv.Video, pv.VideoFromImages, etc.)
        @param layout: A tuple of (rows,cols) to indicate the layout of the montage. Videos will be separated by
        a one-pixel gutter. Videos will be drawn to the montage such that a row is filled up prior to moving
        to the next. The videos are drawn to the montage in the sorted order of the video keys in the dictionary.
        @param size: The window size to display each video in the montage. If the video frame sizes are larger than
        this size, it will be cropped. If you wish to resize, use the size option in the pv.Video class to have
        the output size of the video resized appropriately.
        '''
        if len(videoDict) < 1:
            raise ValueError("You must provide at least one video in the videoDict variable.")
        
        self.vids = videoDict
        self.layout = layout
        self.vidsize = size
        
        #build montage image canvas
        # size based on layout and vidsize
        w = (size[1]+1)*layout[1]
        h = (size[0]+1)*layout[0]
        self.montage = pv.Image( cv.CreateImage((w,h), cv.IPL_DEPTH_8U, 3) )
        
        self.imgs = {}
        self.stopped = []
        
    def __iter__(self):
        ''' Return an iterator for this video '''
        return self  #may not be the best/safest thing to do here
    
    def next(self):
        if len(self.stopped) == len(self.vids.keys()):
            print "All Videos in the Video Montage Have Completed."
            raise StopIteration

        #get next image from each video and put on montage
        #if video has ended, continue to display last image
        #stop when all videos are done.  
        for key in self.vids.keys():
            if key in self.stopped: continue #this video has already reached its end.
            v = self.vids[key]
            try:
                tmp = v.next()
                self.imgs[key] = tmp
            except StopIteration:
                #print "End of a Video %s Reached"%key
                self.stopped.append(key)
            
        #clear previous montage image
        cv.SetZero(self.montage.asOpenCV())
    
        #generate new PIL handle (required...the PIL data seems to get disconnected from the OpenCV/pyvision data)
        montagePIL = self.montage.asPIL()
        indx = 0
        keys = sorted(self.imgs.keys())
        for row in range( self.layout[0] ):
            for col in range( self.layout[1] ):
                (w,h) = self.vidsize
                dx = (w+1)*col
                dy = (h+1)*row
                try:
                    key = keys[indx]
                except IndexError:  #when there are more layout positions than videos...
                    continue                
                tmp = self.imgs[key]
                #tmp.show(window=key, delay=0)
                #print tmp.size
                tmpPIL = tmp.asPIL()
                montagePIL.paste(tmpPIL,box=(dx,dy))
                indx = indx + 1

        return pv.Image(montagePIL)
      
if __name__ == '__main__':
    pass  

#Example usage
#build dictionary of videos to show as montage
#vidsize = (100,100)
#vids = {}
#for i in range(4):
#    label = "Video%d"%i 
#    vids[label] = ...some video object...
#    
#layout=(2,2)
#vm = VideoMontage(vids, layout, vidsize)
#for imgMontage in vm:
#    imgMontage.show(window="VideoMontage", delay=40)
    
    
    