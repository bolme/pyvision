import pyvision as pv
from EyesFile import EyesFile
import os.path
import numpy as np

class FaceDatabase:
    
    class FaceObject:
        def __init__(self):
            self.key = None
            self.person_id = None
            self.image = None
            self.left_eye = None
            self.right_eye = None
            self.nose = None
            self.mouth = None
            self.face_rect = None
            
        def __str__(self):
            return "<FaceObject %s>"%(self.key,) 
            
        
    def __init__(self):
        pass
    
    def keys(self):
        pass
    
    def __getitem__(self,key):
        pass
    
    
class FERETDatabase(FaceDatabase):
    
    def __init__(self,image_path, image_ext=".pgm", coord_file=None):
        ''' Create an object that manages a FERET face database. '''
        self.image_path = image_path
        self.image_ext = image_ext
        
        if coord_file == None:
            coord_name = os.path.join(pv.__path__[0],'analysis','FaceAnalysis','data','coords.3368')
            self.eyes_file = EyesFile(coord_name)
        else:
            self.eyes_file = EyesFile(coord_name)
            

    def keys(self):
        return self.eyes_file.files()
    
    def __getitem__(self,key):
        assert self.eyes_file.hasFile(key)
        
        face_obj = FaceDatabase.FaceObject()
        face_obj.key = key
        face_obj.person_id = key[:5]
        
        leye,reye = self.eyes_file.getEyes(key)[0]
        face_obj.left_eye = leye
        face_obj.right_eye = reye
        
        im_name = os.path.join(self.image_path,key+self.image_ext)
        im = pv.Image(im_name)
        face_obj.image = im
        
        return face_obj



class PIE_ILLUM_Database(FaceDatabase):
    
    def __init__(self,image_path, image_ext=".jpg", coord_file=None):
        ''' Create an object that manages a FERET face database. '''
        self.image_path = image_path
        self.image_ext = image_ext
        
        coord_name = os.path.join(pv.__path__[0], 'analysis', 'FaceAnalysis', 'data', 'pie_illum_coords.csv')
        pie_months =['oct_2000-nov_2000','nov_2000-dec_2000']
        #pie_pose   = [27,5,29,9,7]
        #pie_illum  = [19,20,21,5,6,11,12,10,7,8,9]
        #pie_illum  = [19,20,21,6,11,12,7,8,9]
        
        # Read in PIE Eyes File
        eyes = {}
        for line in open(coord_name):
            #print line
            month,sub,pose,lx,ly,rx,ry = line.split(',')
            
            lx = float(lx)
            ly = float(ly)
            rx = float(rx)
            ry = float(ry)
            
            key = (month,sub,int(pose))
            #label = "%s %s %s"%key
            
            eyes[key] = (pv.Point(lx,ly),pv.Point(rx,ry))
            
        self.eyes = eyes


        keys = []
        # Generate PIE keys
        for month in pie_months:
            month_dir = os.path.join(self.image_path,month)
            for sub in os.listdir(month_dir):
                if sub[0] != '0' or len(sub) != 5:
                    continue
                illum_dir = os.path.join(month_dir,sub,"ILLUM")
                for filename in os.listdir(illum_dir):
                    pose,illum = filename.split('.')[0].split('_')
                    pose = int(pose)
                    illum = int(illum)
                    key = (month,sub,pose,illum)
                    keys.append(key)
        
        self.key_list = keys
            

    def keys(self):
        return self.key_list
    
    def __getitem__(self,key):
        assert key in self.key_list
        
        face_obj = FaceDatabase.FaceObject()
        month,sub,pose,illum = key
        pose = int(pose)
        illum = int(illum)
        
        face_obj.key = key
        face_obj.person_id = key[1]
        
        filename = os.path.join(self.image_path,month,sub,"ILLUM","%02d_%02d"%(pose,illum)+self.image_ext)
        
        face_obj.image = pv.Image(filename)
        
        key = (month,sub,pose)
        
        if self.eyes.has_key(key):
            leye,reye = self.eyes[key]
                    
            face_obj.left_eye = leye
            face_obj.right_eye = reye
        
        return face_obj



# TODO: Needs unit tests.
        
