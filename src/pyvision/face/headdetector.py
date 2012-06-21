'''
Created on Mar 27, 2012

@author: bolme
'''
import pyvision as pv
import os
#import random
#import time
#import csv
import numpy as np
#import copy

import pyvision.face.CascadeDetector as cd

WEIGHTS = np.array([[  5.62758656e+00],
       [  1.43633799e+00],
       [  1.64466919e+01],
       [  1.39618695e+02],
       [  6.65913984e+00],
       [ -2.92879480e+01],
       [  4.61796737e+00],
       [ -5.88458117e+00],
       [  1.18577036e+03],
       [  1.17193154e+03],
       [  1.18310418e+03],
       [  1.00350301e+06],
       [  1.98506471e+06],
       [  2.01482871e+06],
       [  9.81550348e+05],
       [  1.99287479e+06],
       [  1.01131996e+06],
       [ -5.82410452e+00]])

class HeadDetector(object):
    '''
    A detector that uses multiple detectors and quality measures to accuratly 
    detect faces.  The goal is to be slow but accurate.
    '''
    
    def __init__(self,prescale=0.25,weights=WEIGHTS):
        ''' Initialize the detector with default parameters. '''
        # Look for very small faces
        self.fd = cd.CascadeDetector(min_size=(30,30))
        
        # Look for larger heads and shoulders
        self.hd = cd.CascadeDetector(cd.UPPERBODY_MCS_CASCADE,min_size=(60,60))

        # Scale
        self.prescale = prescale
        
        # This regression will provide a quality score for the detector.
        self.quality = pv.LogisticRegression()
        self.quality.params = weights
        
    def detect(self,im,annotate=True):
        '''
        This performs face detection and returns an ordered list of faces sorted by confidence scores.
        '''
        # Compute the raw detections  .          
        detections = self.raw_detections(im)
        
        # Produce a list of faces.
        faces = []
        for each in detections:
            rect = each[0]
            
            # Assign a qualty score to each detection.
            score = self.quality.predict(each[2:])
            rect.detector = each[1]
            rect.score = score
            faces.append(rect)
            
        # Order the list by score.            
        faces.sort(lambda x,y: -cmp(x.score,y.score))
        
        return faces
    
    def raw_detections(self,im):
        '''
        Run the face detectors with additional quality parameters.
        '''
        W,H = im.size
        
        scale = 1.0/self.prescale
        im = im.scale(self.prescale)
            
        faces = self.fd(im)
        faces = [[scale*rect,'FACE'] for rect in faces]
        
        heads = self.hd(im)
        
        # Approximate face locations from head detections
        hfaces = []
        for each in heads:
            # Get the center of the head location
            x,y,w,h = each.asCenteredTuple()
            y = y - 0.10*w
            w = 0.33*w
            hfaces.append([scale*pv.CenteredRect(x,y,w,w),'HEAD'])
            
        # Compute when face and head detections overlap    
        for face in faces:
            best_overlap = 0.0
            for head in hfaces:
                best_overlap = max(best_overlap,face[0].similarity(head[0]))
            if best_overlap > 0.7:
                face.append(1.0)
            else:
                face.append(0.0)

        # Compute when face and head detections overlap    
        for head in hfaces:
            best_overlap = 0.0
            for face in faces:
                best_overlap = max(best_overlap,head[0].similarity(face[0]))
            if best_overlap > 0.7:
                head.append(1.0)
            else:
                head.append(0.0)

        detections = faces + hfaces  
        
        # Compute some simple statistics
        for each in detections:
            tile = pv.AffineFromRect(self.prescale*each[0],(128,128))(im)
            #tile.show()
            
            # face vs head detection
            each.append(1.0*(each[1] == 'FACE'))
            
            # size relative to image
            each.append(np.sqrt(each[0].area())/np.sqrt(W*H))
            each.append(np.sqrt(each[0].area())/np.sqrt(W*H)**2)
            
            # scaled contrast
            each.append(tile.asMatrix2D().std()/255.0)
            each.append((tile.asMatrix2D().std()/255.0)**2)
            
            # scaled brightness
            each.append(tile.asMatrix2D().mean()/255.0)
            each.append((tile.asMatrix2D().mean()/255.0)**2)
            
            # relative rgb intensity
            rgb = tile.asMatrix3D()
            t = rgb.mean() + 0.001 # grand mean regularized
            
            # rgb relative to grand mean
            r = -1+rgb[0,:,:].mean()/t
            g = -1+rgb[1,:,:].mean()/t
            b = -1+rgb[2,:,:].mean()/t
            
            # create a quadradic model with interactions for rgb
            each += [r,g,b,r*r,r*g,r*b,g*g,g*b,b*b]            
                  
        return detections
        
        
    
    def train(self, image_dir, eye_data):
        '''
        This function trains the logistic regression model to score the meta-detections.
        
        Images must be oriented so that the face is upright.
        
        @param image_dir: A pathname containing images.
        @param eye_data: a list of tuples (from csv) filename,eye1x,eye1y,eye2x,eye2y
        '''
        print "Training"
        
        data_set = []
        
        progress = pv.ProgressBar(maxValue=len(eye_data))
        for row in eye_data:
            filename = row[0]
            print "Processing",row
            points = [float(val) for val in row[1:]]
            eye1 = pv.Point(points[0],points[1])
            eye2 = pv.Point(points[2],points[3])
            
            # Compute the truth rectangle from the eye coordinates
            ave_dist = np.abs(cd.AVE_LEFT_EYE.X() - cd.AVE_RIGHT_EYE.X())
            y_height = 0.5*(cd.AVE_LEFT_EYE.Y() + cd.AVE_RIGHT_EYE.Y())
            x_center = 0.5*(eye1.X() + eye2.X())
            x_dist = np.abs(eye1.X() - eye2.X())
            width = x_dist/ave_dist
            y_center = 0.5*(eye1.Y() + eye2.Y()) + (0.5-y_height)*width
            truth = pv.CenteredRect(x_center,y_center,width,width)
            
            # Read the image
            im = pv.Image(os.path.join(image_dir,filename))

            # Compute the detections            
            detections = self.raw_detections(im)
            #print detections
            
            # Score the detections  
            # Similarity above 0.7 count as correct and get a value of 1.0 in the logistic regression
            # Incorrect detections get a value of 0.0
            scores = [truth.similarity(each[0]) for each in detections]
            
            for i in range(len(scores)):
                score = scores[i]
                detection = detections[i]
                success = 0.0
                if score > 0.7:
                    success = 1.0
                row = detection[1],success,detection[2:]
                print row
                data_set.append(row)                    
                            
            # Display the results
            im = im.scale(self.prescale)
            colors = {'FACE':'yellow','HEAD':'blue'}
            for detection in detections:
                #print detection
                rect = self.prescale*detection[0]
                im.annotateRect(rect,color=colors[detection[1]])
            im.annotateRect(self.prescale*truth,color='red')
            progress.updateAmount()
            progress.show()
            print
            #im.show(delay=1)
        progress.finish()
        obs = [each[1] for each in data_set]   
        data = [each[2] for each in data_set] 
        
        print obs
        print data
        
        self.quality.train(obs,data)
        
        return 
      
        for each in data_set:
            self.quality[each[0]][1].append(each[1])
            self.quality[each[0]][2].append(each[2])
        
        for key,value in self.quality.iteritems():
            print "Training:",key
            obs = value[1]
            data = value[2]
            assert len(obs) == len(data)
            value[0].train(obs,data)
            print value[0].params
            
        print "Done Training"
        
        
    def __call__(self,*args,**kwargs):
        return self.detect(*args,**kwargs)

#if __name__ == '__main__':
#    # Create the face detector
#    md = HeadDetector()
#    
#    # Read the eye data into memory
#    f = csv.reader(open('eye_coords_training.csv','rb'))
#    eye_data = [row for row in f]
#    
#    # Train on 800 images.
#    md.train("/FaceData/BFRC_Training/", eye_data[:1000])
#    
#    # Save the detector to disk
#    pickle.dump(md,open('metadetector2.pkl','wb'),protocol=2)
#    
    