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

BUGS_VIDEO = os.path.join(pv.__path__[0],'data','test','BugsSample.m4v')
TAZ_VIDEO = os.path.join(pv.__path__[0],'data','test','TazSample.m4v')
TOYCAR_VIDEO = os.path.join(pv.__path__[0],'data','test','toy_car.m4v')


class _TestVideo(unittest.TestCase):
    '''Tests for the video class.'''


    def testSync(self):
        """Video Sync Test"""
        # Tests a kludge that makes sure the first frame of video is read properly.
        
        # Uncomment next line to show image diagnostics
        ilog = None # pv.ImageLog()
        video_path = os.path.join(DATA_DIR,SYNC_VIDEO)
        video = pv.Video(video_path)
        
        frame_num = 0
        for frame_name in SYNC_FRAMES:
            frame_path = os.path.join(DATA_DIR,frame_name)
            ffmpeg_frame = pv.Image(frame_path)
            opencv_frame = video.next()
            #print ffmpeg_frame.asMatrix3D().shape
            #print opencv_frame.asMatrix3D().shape
            diff = ffmpeg_frame.asMatrix3D() - opencv_frame.asMatrix3D()
            diff_max = max(abs(diff.max()),abs(diff.min()))
            self.assert_(diff_max < 30.0) # Test on MacOS never exceeds 25
            diff = pv.Image(diff)
            if ilog != None:
                #print frame_name,diff_max
                ilog(ffmpeg_frame,"ffmpeg_%04d"%frame_num)
                ilog(opencv_frame,"opencv_%04d"%frame_num)
                ilog(diff,"diff_%04d"%frame_num)
            frame_num += 1

        # Make sure that this is the last frame of the video
        self.assertRaises(StopIteration, video.next)
            
        if ilog != None:
            ilog.show()
        
    def testVideoFrameCount(self):
        """Frame Count Test"""
        video_path = os.path.join(DATA_DIR,SYNC_VIDEO)
        video = pv.Video(video_path)
        
        count = 0
        for _ in video:
            #_.show(delay=0)
            count += 1
        
        self.assertEquals(count,5)
        
    def testFFMPEGFrameCount(self):
        """Frame Count Test"""
        video_path = os.path.join(DATA_DIR,SYNC_VIDEO)
        video = pv.FFMPEGVideo(video_path)
        
        count = 0
        for _ in video:
            #_.show(delay=0)
            count += 1
        
        self.assertEquals(count,5)
        
    def testFFMPEGBugsVideo(self):
        #ilog = pv.ImageLog()
        ilog = None
        
        video = pv.FFMPEGVideo(BUGS_VIDEO)
        
        i = 0
        for frame in video:            
            
            if ilog != None:
                print "Processing Frame:",i
                
                #if frame != None:
                ilog(frame,format='jpg')
                            
            i += 1
        
        if ilog != None:
            ilog.show()

    


       
        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()