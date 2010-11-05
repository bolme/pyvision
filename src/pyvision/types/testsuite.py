'''
Copyright David S. Bolme

Created on Nov 5, 2010

@author: bolme
'''
import unittest

import pyvision as pv
import os.path

DATA_DIR = os.path.join(pv.__path__[0],'data','test')
SYNC_VIDEO = 'video_sync.mov'
SYNC_FRAMES = ['video_sync_0001.jpg', 'video_sync_0002.jpg', 'video_sync_0003.jpg', 'video_sync_0004.jpg', 'video_sync_0005.jpg',]
class TestVideo(unittest.TestCase):
    '''Tests for the video class.'''


    def testSync(self):
        # Uncomment next line to show image diagnostics
        ilog = None # pv.ImageLog()
        video_path = os.path.join(DATA_DIR,SYNC_VIDEO)
        video = pv.Video(video_path)
        
        frame_num = 0
        for frame_name in SYNC_FRAMES:
            frame_path = os.path.join(DATA_DIR,frame_name)
            ffmpeg_frame = pv.Image(frame_path)
            opencv_frame = video.next()
            diff = ffmpeg_frame.asMatrix3D() - opencv_frame.asMatrix3D()
            diff_max = max(abs(diff.max()),abs(diff.min()))
            self.assert_(diff_max < 30.0) # Test on MacOS never exceeds 25
            diff = pv.Image(diff)
            if ilog != None:
                print frame_name,diff_max
                ilog(ffmpeg_frame,"ffmpeg_%04d"%frame_num)
                ilog(opencv_frame,"opencv_%04d"%frame_num)
                ilog(diff,"diff_%04d"%frame_num)
            frame_num += 1

        # Make sure that this is the last frame of the video
        self.assertRaises(StopIteration, video.next)
            
        if ilog != None:
            ilog.show()
        
    def testFrameCount(self):
        video_path = os.path.join(DATA_DIR,SYNC_VIDEO)
        video = pv.Video(video_path)
        
        self.assertEquals(len(video),5)
        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()