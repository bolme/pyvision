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
import time
from pyvision.analysis.FaceAnalysis.BEE import BEEDistanceMatrix

REDUCED_LEYE = pv.Point(128+64,256-20)
REDUCED_REYE = pv.Point(256+128-64,256-20)
REDUCED_SIZE = (512,512)

'''
This is a training set that does not include any overlap with the good bad and ugly.
'''
GOOD_BAD_UGLY_TRAINING = ['nd1S04201','nd1S04207','nd1S04211','nd1S04212','nd1S04213','nd1S04217','nd1S04219','nd1S04222',
                          'nd1S04226','nd1S04227','nd1S04228','nd1S04229','nd1S04243','nd1S04256','nd1S04265','nd1S04273',
                          'nd1S04274','nd1S04279','nd1S04282','nd1S04288','nd1S04295','nd1S04300','nd1S04305','nd1S04308',
                          'nd1S04315','nd1S04316','nd1S04320','nd1S04322','nd1S04323','nd1S04326','nd1S04331','nd1S04335',
                          'nd1S04337','nd1S04339','nd1S04344','nd1S04352','nd1S04358','nd1S04360','nd1S04361','nd1S04365',
                          'nd1S04366','nd1S04367','nd1S04368','nd1S04369','nd1S04371','nd1S04372','nd1S04374','nd1S04376',
                          'nd1S04378','nd1S04380','nd1S04381','nd1S04382','nd1S04386','nd1S04388','nd1S04392','nd1S04395',
                          'nd1S04402','nd1S04403','nd1S04409','nd1S04410','nd1S04411','nd1S04412','nd1S04414','nd1S04415',
                          'nd1S04418','nd1S04423','nd1S04424','nd1S04425','nd1S04428','nd1S04430','nd1S04431','nd1S04432',
                          'nd1S04433','nd1S04435','nd1S04437','nd1S04442','nd1S04444','nd1S04454','nd1S04460','nd1S04467',
                          'nd1S04471','nd1S04479','nd1S04487','nd1S04489','nd1S04495','nd1S04500','nd1S04513','nd1S04515',
                          'nd1S04516','nd1S04519','nd1S04520','nd1S04522','nd1S04523','nd1S04524','nd1S04525','nd1S04527',
                          'nd1S04529','nd1S04530','nd1S04533','nd1S04539','nd1S04540','nd1S04545','nd1S04548','nd1S04549',
                          'nd1S04551','nd1S04558','nd1S04559','nd1S04561','nd1S04563','nd1S04564','nd1S04572','nd1S04573',
                          'nd1S04577','nd1S04579','nd1S04582','nd1S04584','nd1S04585','nd1S04589','nd1S04598','nd1S04599',
                          'nd1S04600','nd1S04610','nd1S04618','nd1S04619','nd1S04624','nd1S04637','nd1S04638','nd1S04641',
                          'nd1S04644','nd1S04657','nd1S04675',]

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


class FRGC_V1(FaceDatabase):
    '''
    FRGC Experiment 4 is a controled to uncontrolled scenario
    
    Note:  The left and right eye in the metadata is relative to the 
    person.  For the face structure this is reversed and follows 
    the pyvision convention which is relative to the image.
    '''
    
    def __init__(self,location):
        self.location = location
        
        self.data_path            = os.path.join(location,"nd1")
        self.metadata_path        = os.path.join(location,"BEE_DIST","FRGC1.0","metadata","FRGC_1.0_Metadata.xml")
                
        self.readSigsets()
        self.readEyeData()
        
        
    def readSigsets(self):
        self.orig_sigset_path     = os.path.join(self.location,"BEE_DIST","FRGC1.0","signature_sets","all.xml")

        self.orig_sigset     = BEE.parseSigSet(self.orig_sigset_path)
        #print self.orig_sigset
        
        self.orig_sigset = filter(lambda x: len(x[1]) > 0, self.orig_sigset)

        self.orig_sigset_map     = dict([ (data[0]['name'],[key,data]) for key,data in self.orig_sigset ])
        
        self.orig_keys     = [ data[0]['name'] for key,data in self.orig_sigset ]
        

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
        self.sigset_dir = os.path.join(self.location,"sigsets")
        
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


def FRGCExp4Test(database, algorithm, face_detector=None, eye_locator=None, n=None,verbose=10.0,ilog=None):
    ''' 
    Run the FRGC Experiment 4 Test 
    
    On completion this will produce a BEE distance matrix.
    '''
    message_time = time.time()
    timer = pv.Timer()

    # Produce face records for each image in the query set
    query_keys = database.query()
    if n != None:
        query_keys = query_keys[:n]
    query_recs = []
    timer.mark("QueryStart")
    i = 0
    for key in query_keys:
        i += 1
        face = database[key]
        
        face_rec = algorithm.getFaceRecord(face.image,None,face.left_eye,face.right_eye)
        query_recs.append(face_rec)    
        if verbose:
            if time.time() - message_time > verbose:
                message_time = time.time()
                print "Processed query image %d of %d"%(i,len(query_keys))
                
    timer.mark("QueryStop",notes="Processed %d images."%len(query_keys))
    
    
    # Produce face records for each image in the target set
    message_time = time.time()
    target_keys = database.target()
    if n != None:
        target_keys = target_keys[:n]
    target_recs = []
    timer.mark("TargetStart")
    i = 0
    for key in target_keys:
        i += 1
        face = database[key]
        
        face_rec = algorithm.getFaceRecord(face.image,None,face.left_eye,face.right_eye)
        target_recs.append(face_rec)    
        if verbose:
            if time.time() - message_time > verbose:
                message_time = time.time()
                print "Processed target image %d of %d"%(i,len(target_keys))
                
    timer.mark("TargetStop",notes="Processed %d images."%len(target_keys))
    
    print "Finished processing FaceRecs (%d query, %d target)"%(len(query_keys),len(target_keys))
    
    # Compute the  matrix
    print "Computing similarity matrix..."
    timer.mark("SimilarityStart")
    mat = algorithm.similarityMatrix(query_recs,target_recs)
    timer.mark("SimilarityStop",notes="Processed %d comparisons."%(mat.shape[0]*mat.shape[1],))

    print "Completing task..."
    print mat.shape
    
    bee_mat = BEEDistanceMatrix(mat,"FRGC_Exp_2.0.4_Query.xml", "FRGC_Exp_2.0.4_Target.xml", sigset_dir=database.sigset_dir, is_distance=False)
    
    if ilog != None:
        ilog(timer)
        
    return bee_mat, timer
    
    
    
    
    
    
    