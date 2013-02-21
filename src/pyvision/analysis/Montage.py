'''
Created on Mar 14, 2011
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
import PIL.Image
import weakref

class ImageMontage(object):
    '''
    Displays thumbnails of a list of input images as a single
    'montage' image. Supports scrolling if there are more images
    than "viewports" in the layout.
    '''

    def __init__(self, image_list, layout=(2,4), tile_size=(64,48), gutter=2, by_row=True, labels='index', 
                keep_aspect=True):
        '''
        Constructor
        @param image_list: A list of pyvision images that you wish to display
        as a montage.
        @param rows: The number of rows in the montage layout
        @param cols: The number of columns in the montage layout
        @param tile_size: The size of each thumbnail image to display in the montage.
        @param gutter: The width in pixels of the gutter between thumbnails.
        @param by_row: If true, the image tiles are placed in row-major order, that
        is, one row of the montage is filled before moving to the next. If false,
        then column order is used instead.
        @param labels: Used to show a label at the lower left corner of each image in the montage.
        If this parameter is a list, then it should be the same length as len(image_list) and contain
        the label to be used for the corresponding image. If labels == 'index', then the image
        montage will simply display the index of the image in image_list. Set labels to None to suppress labels.
        @param keep_aspect: If true the original image aspect ratio will be preserved.
        '''
        self._tileSize = tile_size
        self._rows = layout[0]
        self._cols = layout[1]
        self._images = image_list
        self._gutter = gutter
        self._by_row = by_row
        self._txtfont = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5,0.5)
        self._txtcolor = (255,255,255)   
        self._imgPtr = 0
        self._labels = labels
        self._clickHandler = clickHandler(self)
        self._keep_aspect = keep_aspect
        #check if we need to allow for scroll-arrow padding
        if self._rows * self._cols < len(image_list):
            if by_row:
                self._xpad = 0
                self._ypad = 25
            else: 
                self._ypad = 0
                self._xpad = 25
        else:
            #there will be no scrolling required
            self._xpad = 0
            self._ypad = 0
                
        imgWidth = self._cols*( tile_size[0] + gutter ) + gutter + 2*self._xpad
        imgHeight = self._rows * (tile_size[1] + gutter) + gutter + 2*self._ypad
        self._size = (imgWidth, imgHeight)

        cvimg = cv.CreateImage(self._size, cv.IPL_DEPTH_8U, 3)
        self._cvMontageImage = cvimg 
        
        self._initDecrementArrow()  #build the polygon for the decrement arrow
        self._initIncrementArrow()  #build the polygon for the increment arrow
        self.draw()  #compute the initial montage image
        
    def draw(self, mousePos=None):
        '''
        Computes the image montage from the source images based on the current
        image pointer (position in list of images), etc. This internally constructs
        the montage, but show() is required for display and mouse-click handling.
        '''
        cv.SetZero(self._cvMontageImage)
        
        img_ptr = self._imgPtr
        if img_ptr > 0:
            #we are not showing the first few images in imageList
            #so display the decrement arrow
            cv.FillConvexPoly(self._cvMontageImage, self._decrArrow, (125,125,125))
            
        if img_ptr + (self._rows*self._cols) < len(self._images):
            #we are not showing the last images in imageList
            #so display increment arrow
            cv.FillConvexPoly(self._cvMontageImage, self._incrArrow, (125,125,125))
        
        
        if self._by_row:
            for row in range(self._rows):
                for col in range(self._cols):
                    if img_ptr > len(self._images)-1: break
                    tile = pv.Image(self._images[img_ptr].asAnnotated())
                    self._composite(tile, (row,col), img_ptr)
                    img_ptr += 1
        else:
            for col in range(self._cols):
                for row in range(self._rows):
                    if img_ptr > len(self._images)-1: break
                    tile = pv.Image(self._images[img_ptr].asAnnotated())
                    self._composite(tile, (row,col), img_ptr)
                    img_ptr += 1
        
        #if mousePos != None:
        #    (x,y) = mousePos
        #    cv.Rectangle(self._cvMontageImage, (x-2,y-2), (x+2,y+2), (0,0,255), thickness=cv.CV_FILLED)
                
             
    def asImage(self):
        '''
        If you don't want to use the montage's built-in mouse-click handling by calling
        the ImageMontage.show() method, then this method will return the montage image
        computed from the last call to draw().
        '''
        return pv.Image(self._cvMontageImage)
    
    def show(self, window="Image Montage", pos=None, delay=0):
        '''
        Will display the montage image, as well as register the mouse handling callback
        function so that the user can scroll the montage by clicking the increment/decrement
        arrows.
        '''
        img = self.asImage()
        cv.NamedWindow(window)
        cv.SetMouseCallback(window, self._clickHandler.onClick, window)
        img.show(window=window, pos=pos, delay=delay)
 
    def _checkClickRegion(self, x,y):
        '''
        internal method to determine the clicked region of the montage.
        @return: -1 for decrement region, 1 for increment region, and 0 otherwise
        '''
        if self._by_row:
            #scroll up/down to expose next/prev row
            decr_rect = pv.Rect(0,0, self._size[0], self._ypad)
            incr_rect = pv.Rect(0, self._size[1]-self._ypad, self._size[0], self._ypad)
        else:
            #scroll left/right to expose next/prev col
            decr_rect = pv.Rect(0,0, self._xpad, self._size[1])
            incr_rect = pv.Rect(self._size[0]-self._xpad, 0, self._xpad, self._size[1])
            
        pt = pv.Point(x,y)
        if incr_rect.containsPoint(pt):
            #print "DEBUG: Increment Region"
            return 1
        elif decr_rect.containsPoint(pt):
            #print "DEBUG: Decrement Region"
            return -1
        else:
            #print "DEBUG: Neither Region"
            return 0
               
    def _initDecrementArrow(self):
        '''
        internal method to compute the list of points that represents
        the appropriate decrement arrow (leftwards or upwards) depending
        on the image montage layout.
        '''
        if self._by_row:
            #decrement upwards
            x1 = self._size[0]/2
            y1 = 2
            halfpad = self._ypad / 2
            self._decrArrow = [(x1,y1),(x1+halfpad,self._ypad-2),(x1-halfpad,self._ypad-2)]
        else:
            #decrement leftwards
            x1 = 2
            y1 = self._size[1]/2
            halfpad = self._xpad / 2
            self._decrArrow = [(x1,y1),(x1+self._xpad-3, y1-halfpad),(x1+self._xpad-3,y1+halfpad)]
            
    def _initIncrementArrow(self):
        '''
        internal method to compute the list of points that represents
        the appropriate increment arrow (rightwards or downwards) depending
        on the image montage layout.
        '''
        if self._by_row:
            #increment downwards
            x1 = self._size[0]/2
            y1 = self._size[1] - 3
            halfpad = self._ypad / 2
            self._incrArrow = [(x1,y1),(x1+halfpad,y1-self._ypad+3),(x1-halfpad,y1-self._ypad+3)]
        else:
            #increment rightwards
            x1 = self._size[0] - 2
            y1 = self._size[1]/2
            halfpad = self._xpad / 2
            self._incrArrow = [(x1,y1),(x1-self._xpad+2, y1-halfpad),(x1-self._xpad+2,y1+halfpad)]
        
    def _decr(self):
        '''
        internal method used by _onClick to compute the new imgPtr location after a decrement
        '''
        tmp_ptr = self._imgPtr        
        if self._by_row:
            tmp_ptr -= self._cols
        else:
            tmp_ptr -= self._rows            
        if tmp_ptr < 0:
            self._imgPtr = 0
        else:
            self._imgPtr = tmp_ptr
            
    def _incr(self):
        '''
        internal method used by _onClick to compute the new imgPtr location after an increment
        '''
        tmp_ptr = self._imgPtr        
        if self._by_row:
            tmp_ptr += self._cols
        else:
            tmp_ptr += self._rows
            
        self._imgPtr = tmp_ptr
            
            
    def _composite(self, img, pos, imgNum):
        '''
        Internal method to composite the thumbnail of a given image into the
        correct position, given by (row,col).
        @param img: The image from which a thumbnail will be composited onto the montage
        @param pos: A tuple (row,col) for the position in the montage layout
        @param imgNum: The image index of the tile being drawn, this helps us display the
        appropriate label in the lower left corner if self._labels is not None.
        '''
        (row,col) = pos
        
        if self._keep_aspect:
            # Get the current size
            w,h = img.size
            
            # Find the scale
            scale = min(1.0*self._tileSize[0]/w,1.0*self._tileSize[1]/h)
            w = int(scale*w)
            h = int(scale*h)
            
            # Resize preserving aspect
            img = img.resize((w,h)).asPIL()
            
            # Create a new image with the old image centered
            x = (self._tileSize[0]-w)/2
            y = (self._tileSize[1]-h)/2
            pil = PIL.Image.new('RGB',self._tileSize,"#000000")
            pil.paste(img,(x,y,x+w,y+h))
            
            # Generate the tile
            tile = pv.Image(pil)
        else:
            tile = img.resize(self._tileSize)
        
        pos_x = col*(self._tileSize[0] + self._gutter) + self._gutter + self._xpad
        pos_y = row*(self._tileSize[1] + self._gutter) + self._gutter + self._ypad 
        
        cvImg = self._cvMontageImage
        cvTile = tile.asOpenCV()
        cv.SetImageROI(cvImg, (pos_x,pos_y,self._tileSize[0],self._tileSize[1]))
        
        depth = cvTile.nChannels
        if depth==1:
            cvTileBGR = cv.CreateImage(self._tileSize, cv.IPL_DEPTH_8U, 3)
            cv.CvtColor(cvTile, cvTileBGR, cv.CV_GRAY2BGR)
            cv.Copy(cvTileBGR,cvImg)  #should respect the ROI
        else:
            cv.Copy(cvTile,cvImg)  #should respect the ROI

        if self._labels == 'index':
            #draw image number in lower left corner, respective to ROI
            lbltext = "%d"%imgNum
        elif type(self._labels) == list:
            lbltext = str( self._labels[imgNum])
        else:
            lbltext = None
            
        if not lbltext is None:
            ((tw,th),_) = cv.GetTextSize(lbltext, self._txtfont)
            #print "DEBUG: tw, th = %d,%d"%(tw,th)
            if tw>0 and th>0:
                cv.Rectangle(cvImg, (0,self._tileSize[1]-1),(tw+1,self._tileSize[1]-(th+1)-self._gutter), (0,0,0), thickness=cv.CV_FILLED )
                font = self._txtfont
                color = self._txtcolor
                cv.PutText(cvImg, lbltext, (1,self._tileSize[1]-self._gutter-2), font, color)                        
                   
        #reset ROI 
        cv.SetImageROI(cvImg, (0,0,self._size[0],self._size[1]))


class clickHandler(object):
    '''
    A class for objects designed to handle click events on ImageMontage objects.
    We separate this out from the ImageMontage object to address a memory leak
    when using cv.SetMouseCallback(window, self._onClick, window), because we
    don't want the image data associated with the click handler
    '''
    def __init__(self, IM_Object):
        
        self.IM = weakref.ref(IM_Object)
        
    def onClick(self, event, x, y, flags, window):
        '''
        Handle the mouse click for an image montage object.
        Increment or Decrement the set of images shown in the montage
        if appropriate.
        '''
        IM = self.IM()  #IM object is obtained via weak reference to image montage
        if IM is None: return #if the reference was deleted already...
        
        if event == cv.CV_EVENT_LBUTTONDOWN:
            rc = IM._checkClickRegion(x, y)
            if rc == -1 and IM._imgPtr > 0:
                #user clicked in the decrement region
                IM._decr()
            elif rc == 1 and IM._imgPtr < (len(IM._images)-(IM._rows*IM._cols)):
                IM._incr()
            else:
                pass #do nothing
            
            IM.draw((x,y))
            cv.ShowImage(window, IM._cvMontageImage)
        
class VideoMontage(pv.Video):
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
    def __init__(self, videoDict, layout=(2,4), tileSize=(64,48) ):
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
        self.vidsize = tileSize 
        self.imgs = {}
        self.stopped = []
        
    def __iter__(self):
        ''' Return an iterator for this video '''
        return VideoMontage(self.vids, layout=self.layout, tileSize=self.vidsize)
    
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
            
        keys = sorted(self.imgs.keys())
        imageList = []
        for k in keys:
            imageList.append( self.imgs[k] )
            
        im = ImageMontage(imageList, self.layout, self.vidsize, gutter=2, byrow=True, labels=keys)      
        return im.asImage()
    
def demo_imageMontage():
    import os
    imageList = []
    counter = 0
    
    #get all the jpgs in the data/misc directory
    JPGDIR = os.path.join(pv.__path__[0],'data','misc')
    filenames = os.listdir(JPGDIR)
    jpgs = [os.path.join(JPGDIR,f) for f in filenames if f.endswith(".jpg")]
    
    for fn in jpgs:
        print counter
        if counter > 8: break
        imageList.append( pv.Image(fn) )
        counter += 1
        
    im = ImageMontage(imageList, (2, 3), tileSize=(128,96), gutter=2, byrow=False)
    im.show(window="Image Montage",delay=0)

    
def demo_videoMontage():
    import os

    TOYCAR_VIDEO = os.path.join(pv.__path__[0],'data','test','toy_car.m4v')
    TAZ_VIDEO = os.path.join(pv.__path__[0],'data','test','TazSample.m4v')
    
    vid1 = pv.Video(TOYCAR_VIDEO)
    vid2 = pv.Video(TAZ_VIDEO)
    #vid3 = pv.Video(TOYCAR_VIDEO)
    #vid4 = pv.Video(TAZ_VIDEO)
    vid_dict = {"V1":vid1, "V2":vid2} #, "V3":vid3, "V4":vid4}
    vm = VideoMontage(vid_dict, layout=(2,1), tileSize=(256,192))
    for img in vm:
        img.show("Video Montage", delay=60, pos=(10,10))
    
#if __name__ == '__main__':

#    print "Demo of an Image Montage..."
#    demo_imageMontage()
    
#    print "Demo of a Video Montage..."
#    demo_videoMontage()




