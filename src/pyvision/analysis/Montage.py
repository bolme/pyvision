"""
Created on Mar 14, 2011
@author: svohara
"""
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
import PIL.Image
import weakref
import cv2
import numpy as np

class ImageMontage(object):
    """
    Displays thumbnails of a list of input images as a single
    'montage' image. Supports scrolling if there are more images
    than "viewports" in the layout.
    """

    def __init__(self, image_list, layout=(2, 4), tile_size=(64, 48), gutter=2, by_row=True, labels='index',
                 keep_aspect=True, highlight_selected=False):
        """
        Constructor
        @param image_list: A list of pyvision images that you wish to display
        as a montage.
        @param layout: A tuple (rows,cols) that indicates the number of tiles to
        show in a single montage page, oriented in a grid.
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
        @param highlight_selected: If true, any image tile in the montage which has been clicked will
        be drawn with a rectangular highlight. This will toggle, such that if an image is clicked a second
        time, the highlighting will be removed.
        """
        self._tileSize = tile_size
        self._rows = layout[0]
        self._cols = layout[1]
        self._images = image_list
        self._gutter = gutter
        self._by_row = by_row
        
        self._txtfont = (cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        self._txtcolor = (255, 255, 255)
        
        self._image_list = image_list
        
        self._labels = labels
        self._keep_aspect = keep_aspect
        
        width = tile_size[0]*self._cols
        height = tile_size[1]*self._rows
        im_size = (height,width,3)

        cvimg = np.zeros(im_size,dtype=np.uint8) #cv.CreateImage(self._size, cv.IPL_DEPTH_8U, 3)
        self._cvMontageImage = cvimg

        self.draw()  #compute the initial montage image

    def draw(self):
        """
        Computes the image montage from the source images based on the current
        image pointer (position in list of images), etc. This internally constructs
        the montage, but show() is required for display and mouse-click handling.
        """
        # initialize to zeros
        self._cvMontageImage[:,:,:] = 0

        for i in range(self._rows*self._cols):
            if i > len(self._image_list):
                break
            row = i / self._cols
            col = i % self._cols
            if not self._by_row:
                row = i % self._rows
                col = i / self._rows
            tw = self._tileSize[0]
            th = self._tileSize[1]
            xpos = col*tw
            ypos = row*th
            
            im = self._image_list[i]
            
            cv_tile = im.thumbnail((tw,th)).asOpenCV2()
            ih,iw,ch = cv_tile.shape

            assert ch==3
            
            # Draw Image
            xpos += (tw - iw)/2
            ypos += (th - ih)/2
            self._cvMontageImage[ypos:ypos+ih,xpos:xpos+iw,:] = cv_tile

            # Draw Text
            xpos = col*tw
            ypos = row*th
            lbltext = "%d"%(i)
            ((txtw, txth), _) = cv2.getTextSize(lbltext, *self._txtfont)
            print "DEBUG: tw, th = %d,%d"%(txtw,txth)
            if txtw > 0 and txth > 0:
                cv2.rectangle(self._cvMontageImage, (xpos, ypos), (xpos+txtw + 3, ypos+txth + 3),
                             (0, 0, 0), thickness=cv2.FILLED)
                font = self._txtfont[0]
                font_size = self._txtfont[1]
                font_thick = self._txtfont[2]
                color = self._txtcolor
                cv2.putText(self._cvMontageImage, lbltext, (xpos+1, ypos+1+txth), font, font_size, color,font_thick)
        return
            
            
            
                
        img_ptr = self._imgPtr
        if img_ptr > 0:
            #we are not showing the first few images in imageList
            #so display the decrement arrow
            #cv.FillConvexPoly()
            cv2.fillConvexPoly(self._cvMontageImage, self._decrArrow, (125, 125, 125))
        if img_ptr + (self._rows * self._cols) < len(self._images):
            #we are not showing the last images in imageList
            #so display increment arrow
            cv2.fillConvexPoly(self._cvMontageImage, self._incrArrow, (125, 125, 125))

        self._image_positions = []
        if self._by_row:
            for row in range(self._rows):
                for col in range(self._cols):
                    if img_ptr > len(self._images) - 1: break
                    tile = pv.Image(self._images[img_ptr].asAnnotated())
                    self._composite(tile, (row, col), img_ptr)
                    img_ptr += 1
        else:
            for col in range(self._cols):
                for row in range(self._rows):
                    if img_ptr > len(self._images) - 1: break
                    tile = pv.Image(self._images[img_ptr].asAnnotated())
                    self._composite(tile, (row, col), img_ptr)
                    img_ptr += 1

                    #if mousePos != None:
                    #    (x,y) = mousePos
                    #    cv.Rectangle(self._cvMontageImage, (x-2,y-2), (x+2,y+2), (0,0,255), thickness=cv.CV_FILLED)


    def asImage(self):
        """
        If you don't want to use the montage's built-in mouse-click handling by calling
        the ImageMontage.show() method, then this method will return the montage image
        computed from the last call to draw().
        """
        return pv.Image(self._cvMontageImage)

    def show(self, window="Image Montage", pos=None, delay=0):
        """
        Will display the montage image, as well as register the mouse handling callback
        function so that the user can scroll the montage by clicking the increment/decrement
        arrows.
        @return: The key code of the key pressed, if any, that dismissed the window.
        """
        img = self.asImage()
        key = img.show(window=window, pos=pos, delay=delay)
        return key



    def _composite(self, img, pos, imgNum):
        """
        Internal method to composite the thumbnail of a given image into the
        correct position, given by (row,col).
        @param img: The image from which a thumbnail will be composited onto the montage
        @param pos: A tuple (row,col) for the position in the montage layout
        @param imgNum: The image index of the tile being drawn, this helps us display the
        appropriate label in the lower left corner if self._labels is not None.
        """
        (row, col) = pos

        if self._keep_aspect:
            # Get the current size
            w, h = img.size

            # Find the scale
            scale = min(1.0 * self._tileSize[1] / w, 1.0 * self._tileSize[0] / h)
            w = int(scale * w)
            h = int(scale * h)

            # Resize preserving aspect
            img2 = img.resize((w, h)).asPIL()

            # Create a new image with the old image centered
            x = (self._tileSize[0] - w) / 2
            y = (self._tileSize[1] - h) / 2
            pil = PIL.Image.new('RGB', self._tileSize, "#000000")
            pil.paste(img2, (x, y, x + w, y + h))

            # Generate the tile
            tile = pv.Image(pil)
        else:
            tile = img.resize(self._tileSize)

        pos_x = col * (self._tileSize[0] + self._gutter) + self._gutter + self._xpad
        pos_y = row * (self._tileSize[1] + self._gutter) + self._gutter + self._ypad

        cvImg = self._cvMontageImage
        cvTile = tile.asOpenCV2()
        cvImg[pos_x:pos_x+self._tileSize[0], pos_y:pos_y+self._tileSize[1],:] = np.transpose(cvTile,axes=(1,0,2))

        # Save the position of this image
        self._image_positions.append(
            [self._images[imgNum], imgNum, pv.Rect(pos_x, pos_y, self._tileSize[0], self._tileSize[1])])

        depth = cvTile.shape[2]
        #if depth == 1:
        #    cvTileBGR = cv.CreateImage(self._tileSize, cv.IPL_DEPTH_8U, 3)
        #    cv.CvtColor(cvTile, cvTileBGR, cv.CV_GRAY2BGR)
        #    cv.Copy(cvTileBGR, cvImg)  #should respect the ROI
        #else:
        #    cv.Copy(cvTile, cvImg)  #should respect the ROI

        if self._labels == 'index':
            #draw image number in lower left corner, respective to ROI
            lbltext = "%d" % imgNum
        elif type(self._labels) == list:
            lbltext = str(self._labels[imgNum])
        else:
            lbltext = None

        if not lbltext is None:
            ((tw, th), _) = cv2.getTextSize(lbltext, *self._txtfont)
            #print "DEBUG: tw, th = %d,%d"%(tw,th)
            if tw > 0 and th > 0:
                cv2.rectangle(cvImg, (0, self._tileSize[1] - 1), (tw + 1, self._tileSize[1] - (th + 1) - self._gutter),
                             (0, 0, 0), thickness=cv2.FILLED)
                font = self._txtfont[0]
                font_size = self._txtfont[1]
                font_thick = self._txtfont[2]
                color = self._txtcolor
                cv2.putText(cvImg, lbltext, (1, self._tileSize[1] - self._gutter - 2), font, font_size, color,font_thick)

        if self._highlighted and (imgNum in self._selected_tiles):
            #draw a highlight around this image
            cv2.rectangle(cvImg, (0, 0), (self._tileSize[0], self._tileSize[1]), (0, 255, 255), thickness=4)

                #reset ROI
        #cv2.setImageROI(cvImg, (0, 0, self._size[0], self._size[1]))


class clickHandler(object):
    """
    A class for objects designed to handle click events on ImageMontage objects.
    We separate this out from the ImageMontage object to address a memory leak
    when using cv.SetMouseCallback(window, self._onClick, window), because we
    don't want the image data associated with the click handler
    """

    def __init__(self, IM_Object):

        self.IM = weakref.ref(IM_Object)

    def onClick(self, event, x, y, flags, window):
        """
        Handle the mouse click for an image montage object.
        Increment or Decrement the set of images shown in the montage
        if appropriate.
        """
        IM = self.IM()  #IM object is obtained via weak reference to image montage
        if IM is None: return #if the reference was deleted already...

        #print "event",event
        if event == cv2.EVENT_LBUTTONDOWN:
            rc = IM._checkClickRegion(x, y)
            if rc == -1 and IM._imgPtr > 0:
                #user clicked in the decrement region
                IM._decr()
            elif rc == 1 and IM._imgPtr < (len(IM._images) - (IM._rows * IM._cols)):
                IM._incr()
            else:
                pass #do nothing

            IM.draw((x, y))
            cv2.imshow(window, IM._cvMontageImage)


class VideoMontage(pv.Video):
    """
    Provides a visualization of several videos playing back in
    a single window. This can be very handy, for example, to
    show tracking results of multiple objects from a single video,
    or for minimizing screen real-estate when showing multiple
    video sources.
    
    A video montage object is an iterator, so you "play" the
    montage by iterating through all the frames, just as with
    a standard video object.
    """

    def __init__(self, videoDict, layout=(2, 4), tile_size=(64, 48)):
        """
        @param videoDict: A dictionary of videos to display in the montage. The keys are the video labels, and 
        the values are objects adhering to the pyvision video interface. (pv.Video, pv.VideoFromImages, etc.)
        @param layout: A tuple of (rows,cols) to indicate the layout of the montage. Videos will be separated by
        a one-pixel gutter. Videos will be drawn to the montage such that a row is filled up prior to moving
        to the next. The videos are drawn to the montage in the sorted order of the video keys in the dictionary.
        @param tile_size: The window size to display each video in the montage. If the video frame sizes are larger than
        this size, it will be cropped. If you wish to resize, use the size option in the pv.Video class to have
        the output size of the video resized appropriately.
        """
        if len(videoDict) < 1:
            raise ValueError("You must provide at least one video in the videoDict variable.")

        self.vids = videoDict
        self.layout = layout
        self.vidsize = tile_size
        self.imgs = {}
        self.stopped = []

    def __iter__(self):
        """ Return an iterator for this video """
        return VideoMontage(self.vids, layout=self.layout, tile_size=self.vidsize)

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
            imageList.append(self.imgs[k])

        im = ImageMontage(imageList, self.layout, self.vidsize, gutter=2, by_row=True, labels=keys)
        return im.asImage()


def demo_imageMontage():
    import os

    imageList = []
    counter = 0

    #get all the jpgs in the data/misc directory
    JPGDIR = os.path.join(pv.__path__[0], 'data', 'misc')
    filenames = os.listdir(JPGDIR)
    jpgs = [os.path.join(JPGDIR, f) for f in filenames if f.endswith(".jpg")]

    for fn in jpgs:
        print counter
        if counter > 8: break
        imageList.append(pv.Image(fn))
        counter += 1

    im = ImageMontage(imageList, (2, 3), tile_size=(128, 96), gutter=2, by_row=False)
    im.show(window="Image Montage", delay=0)
    cv2.destroyWindow('Image Montage')


def demo_videoMontage():
    import os

    TOYCAR_VIDEO = os.path.join(pv.__path__[0], 'data', 'test', 'toy_car.m4v')
    TAZ_VIDEO = os.path.join(pv.__path__[0], 'data', 'test', 'TazSample.m4v')

    vid1 = pv.Video(TOYCAR_VIDEO)
    vid2 = pv.Video(TAZ_VIDEO)
    #vid3 = pv.Video(TOYCAR_VIDEO)
    #vid4 = pv.Video(TAZ_VIDEO)
    vid_dict = {"V1": vid1, "V2": vid2} #, "V3":vid3, "V4":vid4}
    vm = VideoMontage(vid_dict, layout=(2, 1), tile_size=(256, 192))
    vm.play("Video Montage", delay=60, pos=(10, 10))

if __name__ == '__main__':

#    print "Demo of an Image Montage..."
    demo_imageMontage()

#    print "Demo of a Video Montage..."
#    demo_videoMontage()




