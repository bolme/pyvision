'''
Created on Jan 13, 2012

@author: bolme
'''
import pyvision as pv
from pyvision.face.CascadeDetector import CascadeDetector
from vtm import *
import numpy as np

class ChangeDetectionVT(VideoTask):
    
    def __init__(self,frame_id):
        '''
        This is a change detection task that computes the difference between the current and previous frames.
        '''
        # request the current and previous frames.
        VideoTask.__init__(self, frame_id, required_args=[("FRAME",frame_id),("FRAME",frame_id-1)])
        
    def execute(self, curr, prev):
        '''
        Perform change detection between two frames.
        '''
        # Get the frames as matrices
        curr = curr.asMatrix2D()
        prev = prev.asMatrix2D()
        
        # Compute changes
        diff = np.abs(curr - prev)
        mask = diff > 20.0
        
        # Create a new mask data item.
        return [("CHANGE_MASK",self.getFrameId(),mask)]
    
    
class ChangeDetectionAnnotationVT(VideoTask):
    
    def __init__(self,frame_id):
        '''
        Register for the current and previous frames.
        '''
        VideoTask.__init__(self, frame_id, required_args=[("FRAME",frame_id),("CHANGE_MASK",frame_id)])
        
    def execute(self, frame, mask):
        '''
        Perform change detection between two frames.
        '''
        # Annotate the frame with the mask.
        frame.annotateMask(mask,color='red')
        
        # Return an empty list because no new items were created.
        return []
    
class FaceDetectorVT(VideoTask):
    '''
    This tasks illustrates one way to initialize data in the first frame by 
    changing the number of required arguments required by additional frames.
    '''
    def __init__(self,frame_id):
        if frame_id == 0:
            # The first frame only requires the frame
            VideoTask.__init__(self,frame_id,required_args=[("FRAME",frame_id)])
        else:
            # Each additional frame requires an initialized detector.
            # The underscore in _FACE_DETECTOR indicates this is not shared
            # data.
            VideoTask.__init__(self,frame_id,required_args=[("FRAME",frame_id),("_FACE_DETECTOR",frame_id-1)])
    
    def execute(self,frame,detector=None):
        if detector == None:
            print "Initializing Face Detector."
            detector = CascadeDetector(min_size=(128,128))
        
        faces = detector(frame)
        for rect in faces:
            frame.annotateRect(rect)
        
        return [('FACES',self.getFrameId(),faces),("_FACE_DETECTOR",self.getFrameId(),detector)]


def runChangeDetectionExample():
    video = pv.Video(pv.BUGS_VIDEO)
    
    vtm = VideoTaskManager(buffer_size=2,debug_level=2)
    vtm.addTaskFactory(ChangeDetectionVT)
    vtm.addTaskFactory(ChangeDetectionAnnotationVT)
    
    for frame in video:
        vtm.addFrame(frame)

def runFaceDetectionExample():
    video = pv.Webcam()
    
    vtm = VideoTaskManager(buffer_size=2,debug_level=3)
    vtm.addTaskFactory(FaceDetectorVT)
    
    for frame in video:
        vtm.addFrame(frame)

if __name__ == '__main__':
    #runChangeDetectionExample()
    runFaceDetectionExample()
    
    
    