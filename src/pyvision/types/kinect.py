'''
Created on Jan 28, 2012

@author: bolme
'''
#import freenect
import cv
import numpy as np
import pyvision as pv


# Here is a nice pythonic interface to the kinect sensor for pyvision
class Kinect(object):
    '''
    This class provides a simple interface to the kinect sensor.  When operating 
    properly the device produces an color RGB video frame (8bit 640x480) and a 
    depth image (11-bit, 640x480) at around 20-24 frames per second with low latency.

    
    The kinect drivers to not appear to be robust.  Some trial and error may be
    needed to determine how to best interface with the device. Here are some problem 
    behaviours that I have seen:
      * The Kinect does not seem to start properly.
      * Calling some functions in the freenect module results in segmentation
        faults
      * Connecting to the sensor temporarily or perminatly interupts the usb
        bus. (The keyboard and mouse stop working.)
      * The device seems to work better when connected to one of the ports on
        my laptop but not the other.
      * The device sometimes prints out error/warning messages which do not seem
        to interfere with the operation of the device.
        "[Stream 70] Expected 1748 data bytes, but got 948"
        
    After connecting the device to the usb port it may be helpful to initialize 
    the device using ipython.  Try running these commands before connecting through
    pyvision or attempting to grab frames:
    
    import freenect
    freenect.init()
    exit
    
    The current implementation initializes the device and returns RGB and depth
    frames.  No advanced features such as LED, Audio, or Tilt are supported because
    they cause sementation violations.
    
    '''
    
    def __init__(self):
        '''
        Initialize a Kinect device.
        '''
        # Try importing the freenect library.  
        import freenect as fn
        self.fn = fn
        self.fn.init()
        
    def __iter__(self):
        ''' Return an iterator for this video '''
        return self

    def next(self):
        '''
        Get the next frame and depth image from the sensor.
        '''
        # This is an 11bit numpy array
        depth = self.fn.sync_get_depth()[0]
        
        # This is an 8 bit numpy array
        frame = self.fn.sync_get_video()[0]
        
        r,c,chan = frame.shape
        
        cvframe = cv.CreateImageHeader((c,r),cv.IPL_DEPTH_8U,chan)
        cv.SetData(cvframe,frame.tostring(),3*c)
        cv.CvtColor(cvframe,cvframe,cv.CV_RGB2BGR)
        #video = video_cv(frame)
        return pv.Image(cvframe),pv.Image(np.array(depth.T,dtype=np.float))
    
    
if __name__ == '__main__':
    print "Testing the kinect sensor."
    print "Press any key to quit"
    
    kinect = pv.Kinect()
    for frame, depth in kinect:
        # Display the images
        k = max(
                depth.show(window="Depth",delay=1,pos=(100,100)),
                frame.show(window="Video",delay=1,pos=(740,100))
                )
        
        # Check to see if a key was pressed
        if k != -1:
            break       
        
