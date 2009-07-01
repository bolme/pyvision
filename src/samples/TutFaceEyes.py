"""
This is some sample code that shows how to use the face detector
and eye locator.
"""

import os.path
import pyvision as pv

# The ASEF eye locator has patent complications.  This next line
# disables those warnings.
pv.disableCommercialUseWarnings()

from pyvision.face.CascadeDetector import CascadeDetector
from pyvision.face.FilterEyeLocator import FilterEyeLocator

if __name__ == "__main__":
    ilog = pv.ImageLog()

    # Load the face image file
    fname = os.path.join(pv.__path__[0],'data','misc','FaceSample.jpg')
    
    # Create the annotation image in black and white so that color
    # annotations show up better.
    im = pv.Image(fname,bw_annotate=True)

    # Create a OpenCV cascade face detector object
    cd = CascadeDetector()
    
    # Create an eye detector object
    el = FilterEyeLocator()

    # Call the face detector like a function to get a list of face rectangles
    rects = cd(im)
    
    # print the list of rectangles
    print "Face Detection Output:",rects
    
    # Also call the eye detector like a function with the original image and
    # the list of face detections to locate the eyes.
    eyes = el(im,rects)
    
    # print the list of eyes.  Format [ [ face_rect, left_eye, right_eye], ...]
    print "Eye Locator Output:",eyes
    
    # Now you can process the detection and eye data for each face detected in the
    # image.  Here we  annotate the image with the face detection box and 
    # eye coordinates and we create create a normalized face image by translating
    # rotating and scaling the face using pv.AffineFromPoints
    for face_rect,left_eye,right_eye in eyes:
        # Annotate the original image
        im.annotateRect(face_rect, color='red')
        im.annotatePoint(left_eye, color='yellow')
        im.annotatePoint(right_eye, color='yellow')
        
        # Align the eye coordinates to produce a face tile.  This is a typical 
        # step before running a face verification algorithm.
        affine = pv.AffineFromPoints(left_eye,right_eye,pv.Point(32.0,64.0),pv.Point(96.0,64.0),(128,160))
        tile = affine.transformImage(im)
        ilog.log(tile)
        
    # Finally, display the annotate image.
    ilog.log(im)
    ilog.show()
