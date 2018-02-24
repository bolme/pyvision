'''
Created on Jul 22, 2011
@author: Stephen O'Hara

This demonstration will play the streaming video from a compliant IP (network) video camera.
Getting the url correct for your make/model camera is critical, and you'll also need to
have OpenCV built with ffmpeg support. If the dependencies are met, you can see that this
is two lines of trivial pyvision code!
'''
import pyvision as pv

#The following is the rtsp url for a linksys WVC54GCA IP camera,
# which can be purchased for less than $100
# Of course the ip address in the middle of the URL below will
# need to be changed as appropriate for your local network.
cam_url = "rtsp://192.168.2.55/img/video.sav"

if __name__ == '__main__':
    pass

print("Please be patient, it can take several seconds to buffer live video...")
print("When video is playing, if you click on the video window and then hold down the spacebar, you can pause it.")
print("When paused, you can hit 's' to step one frame at a time, 'c' to continue playback, or 'q' to quit.")
vid = pv.Video(cam_url)
vid.play()