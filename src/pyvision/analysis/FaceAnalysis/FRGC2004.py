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
Created on May 15, 2009

@author: bolme
'''

import BEE
from FaceDatabase import FaceDatabase
import os.path
import os
import copy
import pyvision as pv
import xml.etree.cElementTree as et
import shutil

REDUCED_LEYE = pv.Point(128+64,256-20)
REDUCED_REYE = pv.Point(256+128-64,256-20)
REDUCED_SIZE = (512,512)

class FRGCMetadata:
    def __init__(self):
        pass


class FRGC_Exp1(FaceDatabase):
    '''
    FRGC Experiment 1 is a controlled to controlled scenario
    '''
    
    def __init__(self,location):
        pass
    
class FRGC_Exp4(FaceDatabase):
    '''
    FRGC Experiment 4 is a controled to uncontrolled scenario
    
    Note:  The left and right eye in the metadata is relative to the 
           person.  For the face structure this is reversed and follows 
           the pyvision convention which is relative to the image.
    '''
    
    def __init__(self,location):
        self.location = location
        
        self.data_path            = os.path.join(location,"nd1")
        self.metadata_path        = os.path.join(location,"BEE_DIST","FRGC2.0","metadata","FRGC_2.0_Metadata.xml")
                
        self.readSigsets()
        self.readEyeData()
        
        
    def readSigsets(self):
        self.orig_sigset_path     = os.path.join(self.location,"BEE_DIST","FRGC2.0","signature_sets","experiments","FRGC_Exp_2.0.4_Orig.xml")
        self.query_sigset_path    = os.path.join(self.location,"BEE_DIST","FRGC2.0","signature_sets","experiments","FRGC_Exp_2.0.4_Query.xml")
        self.target_sigset_path   = os.path.join(self.location,"BEE_DIST","FRGC2.0","signature_sets","experiments","FRGC_Exp_2.0.4_Target.xml")
        self.training_sigset_path = os.path.join(self.location,"BEE_DIST","FRGC2.0","signature_sets","experiments","FRGC_Exp_2.0.4_Training.xml")

        self.orig_sigset     = BEE.parseSigSet(self.orig_sigset_path)
        self.query_sigset    = BEE.parseSigSet(self.query_sigset_path)
        self.target_sigset   = BEE.parseSigSet(self.target_sigset_path)
        self.training_sigset = BEE.parseSigSet(self.training_sigset_path)
        
        self.orig_sigset_map     = dict([ (data[0]['name'],[key,data]) for key,data in self.orig_sigset ])
        self.query_sigset_map    = dict([ (data[0]['name'],[key,data]) for key,data in self.query_sigset ])
        self.target_sigset_map   = dict([ (data[0]['name'],[key,data]) for key,data in self.target_sigset ])
        self.training_sigset_map = dict([ (data[0]['name'],[key,data]) for key,data in self.training_sigset ])
        
        self.orig_keys     = [ data[0]['name'] for key,data in self.orig_sigset ]
        self.query_keys    = [ data[0]['name'] for key,data in self.query_sigset ]
        self.target_keys   = [ data[0]['name'] for key,data in self.target_sigset ]
        self.training_keys = [ data[0]['name'] for key,data in self.training_sigset ]
        
        # Check that everything adds up.
        assert len(set(self.orig_keys)) == len(self.orig_keys)
        assert len(set(self.query_keys + self.target_keys + self.training_keys)) == len(self.orig_keys)
        assert len(set(self.query_keys + self.target_keys + self.training_keys)) == len(self.query_keys) + len(self.target_keys) + len(self.training_keys)

    def readEyeData(self):
        f = open(self.metadata_path,'rb')
        xml = et.parse(f)
        
        self.metadata = {}
        
        for recording in xml.findall('Recording'):
            md = FRGCMetadata()
            md.rec_id = recording.get('recording_id')
            md.sub_id = recording.get('subject_id')
            md.capture_date = recording.get('capturedate')
            
            md.left_eye  = None
            md.right_eye = None
            md.nose      = None
            md.mouth     = None
            
            for leye_center in recording.findall('LeftEyeCenter'):
                x,y = float(leye_center.get('x')),float(leye_center.get('y'))
                md.left_eye = pv.Point(x,y)
                
            for leye_center in recording.findall('RightEyeCenter'):
                x,y = float(leye_center.get('x')),float(leye_center.get('y'))
                md.right_eye = pv.Point(x,y)
                
            for leye_center in recording.findall('Nose'):
                x,y = float(leye_center.get('x')),float(leye_center.get('y'))
                md.nose = pv.Point(x,y)
                
            for leye_center in recording.findall('Mouth'):
                x,y = float(leye_center.get('x')),float(leye_center.get('y'))
                md.mouth = pv.Point(x,y)
                
            self.metadata[md.rec_id] = md


    def keys(self):
        return copy.copy(self.orig_keys)
    
    def query(self):
        return copy.copy(self.query_keys)
    
    def target(self):
        return copy.copy(self.target_keys)
    
    def training(self):
        return copy.copy(self.training_keys)
    
    def getMetadata(self,key):
        meta = self.metadata[key]
        sig = self.orig_sigset_map[key]
        return key,sig[0],sig[1][0]['file-name'],meta.right_eye,meta.left_eye,meta.nose,meta.mouth

    
    def __getitem__(self,key):
        entry = self.orig_sigset_map[key]
        
        face_obj = FaceDatabase.FaceObject()
        
        face_obj.key = key
        face_obj.person_id = entry[0]
        face_obj.entry = entry
        face_obj.metadata = self.metadata[key]
        
        # The left_eye and right_eye in the metadata is relative to the subject
        # This needs to be reversed because pyvision used image left and image right.
        face_obj.left_eye  = face_obj.metadata.right_eye
        face_obj.right_eye = face_obj.metadata.left_eye
        
        im_name = os.path.join(self.location,entry[1][0]['file-name'])
        im = pv.Image(im_name)
        face_obj.image = im
        
        return face_obj


class FRGC_Exp4_Reduced(FaceDatabase):
    '''
    FRGC Experiment 4 is a controled to uncontrolled scenario
    
    Note:  The left and right eye in the metadata is relative to the 
           person.  For the face structure this is reversed and follows 
           the pyvision convention which is relative to the image.
    '''
    
    def __init__(self,location):
        self.location = location
        
        self.data_path            = os.path.join(location,"nd1")
                
        self.readSigsets()
        
        
    def readSigsets(self):
        self.orig_sigset_path     = os.path.join(self.location,"sigsets","FRGC_Exp_2.0.4_Orig.xml")
        self.query_sigset_path    = os.path.join(self.location,"sigsets","FRGC_Exp_2.0.4_Query.xml")
        self.target_sigset_path   = os.path.join(self.location,"sigsets","FRGC_Exp_2.0.4_Target.xml")
        self.training_sigset_path = os.path.join(self.location,"sigsets","FRGC_Exp_2.0.4_Training.xml")

        self.orig_sigset     = BEE.parseSigSet(self.orig_sigset_path)
        self.query_sigset    = BEE.parseSigSet(self.query_sigset_path)
        self.target_sigset   = BEE.parseSigSet(self.target_sigset_path)
        self.training_sigset = BEE.parseSigSet(self.training_sigset_path)
        
        self.orig_sigset_map     = dict([ (data[0]['name'],[key,data]) for key,data in self.orig_sigset ])
        self.query_sigset_map    = dict([ (data[0]['name'],[key,data]) for key,data in self.query_sigset ])
        self.target_sigset_map   = dict([ (data[0]['name'],[key,data]) for key,data in self.target_sigset ])
        self.training_sigset_map = dict([ (data[0]['name'],[key,data]) for key,data in self.training_sigset ])
        
        self.orig_keys     = [ data[0]['name'] for key,data in self.orig_sigset ]
        self.query_keys    = [ data[0]['name'] for key,data in self.query_sigset ]
        self.target_keys   = [ data[0]['name'] for key,data in self.target_sigset ]
        self.training_keys = [ data[0]['name'] for key,data in self.training_sigset ]
        
        # Check that everything adds up.
        assert len(set(self.orig_keys)) == len(self.orig_keys)
        assert len(set(self.query_keys + self.target_keys + self.training_keys)) == len(self.orig_keys)
        assert len(set(self.query_keys + self.target_keys + self.training_keys)) == len(self.query_keys) + len(self.target_keys) + len(self.training_keys)


    def keys(self):
        return copy.copy(self.orig_keys)
    
    def query(self):
        return copy.copy(self.query_keys)
    
    def target(self):
        return copy.copy(self.target_keys)
    
    def training(self):
        return copy.copy(self.training_keys)
    
    def __getitem__(self,key):
        entry = self.orig_sigset_map[key]
        
        face_obj = FaceDatabase.FaceObject()
        
        face_obj.key = key
        face_obj.person_id = entry[0]
        face_obj.entry = entry
        
        # The left_eye and right_eye in the metadata is relative to the subject
        # This needs to be reversed because pyvision used image left and image right.
        face_obj.left_eye  = REDUCED_LEYE
        face_obj.right_eye = REDUCED_REYE
        
        im_name = os.path.join(self.location,'recordings',key+".jpg")
        im = pv.Image(im_name)
        face_obj.image = im
        
        return face_obj


def reduce_exp4(source_dir,dest_dir):
    ''''''
    print "Creating directories."
    try:
        os.makedirs(os.path.join(dest_dir,'recordings'))
    except:
        pass
    
    try:
        os.makedirs(os.path.join(dest_dir,'sigsets'))
    except:
        pass
    
    print "Loading FRGC Information."
    frgc = FRGC_Exp4(source_dir)
    
    print "Processing Images."
    keys = frgc.keys()
    for i in xrange(len(keys)):
        key = keys[i]
        face = frgc[key]
        print "Processing %d of %d:"%(i+1,len(keys)), key,face.person_id
        
        affine = pv.AffineFromPoints(face.left_eye,face.right_eye,REDUCED_LEYE,REDUCED_REYE,REDUCED_SIZE)
        tile = affine.transformImage(face.image)
        tile.asPIL().save(os.path.join(dest_dir,'recordings',key+".jpg"),quality=95)

        #if i > 10:
        #    break
        
    print "Copying sigsets."
    shutil.copy(frgc.orig_sigset_path,     os.path.join(dest_dir,'sigsets'))
    shutil.copy(frgc.query_sigset_path,    os.path.join(dest_dir,'sigsets'))
    shutil.copy(frgc.target_sigset_path,   os.path.join(dest_dir,'sigsets'))
    shutil.copy(frgc.training_sigset_path, os.path.join(dest_dir,'sigsets'))
    
    print "Copying metadata."
    shutil.copy(frgc.metadata_path,        os.path.join(dest_dir,'sigsets'))

if __name__ == "__main__":
    import csv
    f = open('data/FRGC_Exp4_query_flat.csv','wb')
    writer = csv.writer(f)
    db = FRGC_Exp4("/Users/bolme/vision/data/FRGC_Metadata")
    keys = db.query()
    writer.writerow(['rec_id','sub_id','filename','eye1_x','eye1_y','eye2_x','eye2_y','nose_x','nose_y','mouth_x','mouth_y'])
    for key in keys:
        #print key
        rec_id,sub_id,filename,left_eye,right_eye,nose,mouth = db.getMetadata(key)
        #print "   ",rec_id,sub_id,filename,left_eye,right_eye,nose,mouth
        writer.writerow([rec_id,sub_id,filename,left_eye.X(),left_eye.Y(),right_eye.X(),right_eye.Y(),nose.X(),nose.Y(),mouth.X(),mouth.Y()])
    print "done"
        
