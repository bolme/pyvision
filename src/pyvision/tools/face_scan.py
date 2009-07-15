# PyVision License
#
# Copyright (c) 2006-2008 David S. Bolme
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

'''
This tool scans a directory for image files and detects faces and eyes in those 
images.  A CSV file is generated that has each face detection and ASEF eye 
coordinates for each face detection.
'''


import optparse
import csv
import os
import pyvision as pv
from pyvision.face.CascadeDetector import CascadeDetector
from pyvision.face.FilterEyeLocator import FilterEyeLocator
import PIL
import random

EXTENSIONS = ["PGM","PPM","BMP","JPG","JPEG","GIF","PNG","TIF","TIFF"]

def parseOptions():
    usage = "usage: %prog [options] <image_directory> <output.csv>"
    parser = optparse.OptionParser(usage)
    parser.add_option("--rotate", dest="rotate",default=False,
                      action="store_true",
                      help="Used to detection faces in images where the camera was turn while taking the photo.  Tests all four rotations.")
    parser.add_option("--scale", dest="scale",default=1.0,type='float',
                      help="Rescale the image before detection to improve performance.")
    parser.add_option("--extension", dest="extension",default=None,
                      help="Attempt to process images with this extension.")
    parser.add_option("--log", dest="log_dir",default=None,
                      help="Create a directory containing annotated images.")
    parser.add_option("--log-scale", dest="log_scale",default=1.0,type='float',
                      help="Rescale images before they are logged.")
    parser.add_option("--sample", dest="sample",default=None,type='int',
                      help="Randomly sample n images to process.")
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose",
                      help="Turn on more verbose output.")
    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error("This program requires two arguments: a directory containing images and the name of a file to use for output.")

    return options, args


def processFaces(im,face_detect,locate_eyes):
    
    # Run face detection
    faces = face_detect(im)
    return locate_eyes(im,faces)
    #results = []
    #i = 0
    #for face in faces:
    # Run eye detection
    #    affine = pv.AffineFromRect(face, (128, 128))
    #    face_im = affine.transformImage(im)
    #    cv_im = face_im.asOpenCVBW()
    #    eye1, eye2, corr1, corr2 = locate_eyes.locateEyes(cv_im)
    #    eye1, eye2 = affine.invertPoints([pv.Point(eye1), pv.Point(eye2)])
    #    results.append([face,eye1,eye2])
    #    i += 1
    #
    #return results

if __name__ == "__main__":
    # Read in program arguments and options.
    options, args = parseOptions()
    
    #locator_filename = os.path.join(csu.__path__[0],'data','EyeLocatorASEF128x128.fel')
    
    # Scan the directory for image files.
    image_names = []
    for dirpath, dirnames, filenames in os.walk(args[0]):
        for filename in filenames:
            extension = filename.split('.')[-1]
            extension = extension.upper()

            if (options.extension==None and extension in EXTENSIONS) or (options.extension != None and options.extension.upper() == extension):
                pathname = os.path.join(dirpath,filename)
                image_names.append(pathname)
                           
    # If an integer is passed to the sample option then subselect the image names.
    if options.sample != None:
        image_names = random.sample(image_names,options.sample)
    
    
    # Open the file to use as output.
    f = open(args[1],'wb')
    csv_file = csv.writer(f)
    headers = ['image_name','detect_number','detect_x','detect_y','detect_width','detect_height','eye1_x','eye1_y','eye2_x','eye2_y']
    csv_file.writerow(headers)
    
    # Create an image log if this is being saved to a file.
    ilog = None
    if options.log_dir != None:
        print "Creating Image Log..."
        ilog = pv.ImageLog(options.log_dir)
    
    # For each image run face and eye detection
    face_detect = CascadeDetector(image_scale=1.3*options.scale)
    locate_eyes = FilterEyeLocator()#locator_filename)
    
    c = 0
    for pathname in image_names:
        c += 1
        
        im = pv.Image(pathname)

        scale = options.log_scale
        log_im = pv.AffineScale(scale,(int(scale*im.width),int(scale*im.height))).transformImage(im)
        
            
 
        results = processFaces(im,face_detect,locate_eyes)

        if options.rotate:
            
            rot_image = pv.Image(im.asPIL().transpose(PIL.Image.ROTATE_90))
            more_results = processFaces(rot_image,face_detect,locate_eyes)
            for face,eye1,eye2 in more_results:
                results.append([pv.Rect(im.width-face.y-face.h, face.x, face.h, face.w),
                               pv.Point(im.width-eye1.Y(),eye1.X()),
                               pv.Point(im.width-eye2.Y(),eye2.X())])

            rot_image = pv.Image(im.asPIL().transpose(PIL.Image.ROTATE_180))
            more_results = processFaces(rot_image,face_detect,locate_eyes)            
            for face,eye1,eye2 in more_results:
                results.append([pv.Rect(im.width - face.x - face.w, im.height-face.y-face.h, face.w, face.h),
                               pv.Point(im.width-eye1.X(),im.height-eye1.Y()),
                               pv.Point(im.width-eye2.X(),im.height-eye2.Y())])
                
            rot_image = pv.Image(im.asPIL().transpose(PIL.Image.ROTATE_270))
            more_results = processFaces(rot_image,face_detect,locate_eyes)
            for face,eye1,eye2 in more_results:
                results.append([pv.Rect(face.y, im.height-face.x-face.w, face.h, face.w),
                               pv.Point(eye1.Y(),im.height-eye1.X()),
                               pv.Point(eye2.Y(),im.height-eye2.X())])
                

        n_faces = 0
        for face,eye1,eye2 in results:
            csv_file.writerow([pathname,n_faces,face.x,face.y,face.w,face.h,eye1.X(),eye1.Y(),eye2.X(),eye2.Y()])
            if ilog != None:
                log_im.annotateRect(scale*face)
                log_im.annotatePoint(scale*eye1)
                log_im.annotatePoint(scale*eye2)
            n_faces += 1

        #else:
        #    csv_file.writerow([pathname,"NA","NA","NA","NA","NA","NA","NA","NA","NA"])
        
        print "Processed %5d of %d: [%2d faces] %s "%(c,len(image_names),n_faces,pathname)
        
        if ilog != None:
            basename = os.path.basename(pathname)
            basename = basename.split('.')[0]
            ilog.log(log_im,label=basename)
        
    if ilog != None:
        ilog.show()
    
    
    
    

