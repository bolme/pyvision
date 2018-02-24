'''
This file provides a basic framework for a live demo.  In this case the 
demo is for face and eye detection.

Copyright David S. Bolme

Created on Jul 9, 2011

@author: bolme
'''
import pyvision as pv
import cv2
from pyvision.face.CascadeDetector import CascadeDetector
from pyvision.face.FilterEyeLocator import FilterEyeLocator

def mouseCallback(event, x, y, flags, param):
    if event in [cv2.EVENT_LBUTTONDOWN,cv2.EVENT_LBUTTONUP]:
        print "Mouse Event:",event,x,y

if __name__ == '__main__':
    
    # Setup the webcam
    webcam  = pv.Webcam(size=(640,360))
    
    # Setup the face and eye detectors
    cd = CascadeDetector(min_size=(100,100))
    el = FilterEyeLocator()
    
    # Setup the mouse callback to handle mause events (optional)
    cv2.namedWindow("PyVision Live Demo")
    cv2.setMouseCallback("PyVision Live Demo", mouseCallback)
    
    while True:
        # Grab a frame from the webcam
        frame = webcam.query()
        
        # Run Face and Eye Detection
        rects = cd(frame)
        eyes = el(frame,rects)

        # Annotate the result
        for rect,leye,reye in eyes:
            frame.annotateThickRect(rect, 'green', width=3)
            frame.annotatePoint(leye, color='green')
            frame.annotatePoint(reye, color='green')
        
        # Annotate instructions
        frame.annotateLabel(pv.Point(10,10), "Press 'q' to quit.")
        
        # Show the frame (uses opencv highgui)
        key_press = frame.show("PyVision Live Demo",delay=1)
        
        # Handle key press events.
        if key_press == ord('q'):
            break