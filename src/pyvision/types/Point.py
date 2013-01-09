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


from numpy import array
from math import sqrt
import numpy as np
import csv

import pyvision as pv

class Point:
    def __init__(self,x=0.0,y=0.0,z=0.0,w=1.0,scale=1.0,rotation=0.0):
        ''' 
        Create a point.
        
        Arguments:
        x: x coordinate
        y: y coordinate
        z: z coordinate
        w: homoginious coordinate
        scale: scale selection
        rotation: rotation selection
        '''
        #if isinstance(x,cv.Point):
        #    self.x = float(x.x)
        #    self.y = float(x.y)
        #    self.z = 0.0
        #    self.w = 1.0
        #    self.scale = 1.0
        #    self.rotation = 0.0
        if isinstance(x,tuple):
            self.x = float(x[0])
            self.y = float(x[1])
            self.z = 0.0
            self.w = 1.0
            if len(x) > 2:
                self.z = x[3]
            if len(x) > 3:
                self.w = x[4]
        #elif isinstance(x,cv.Point2D32f):
        #    self.x = float(x.x)
        #    self.y = float(x.y)
        #    self.z = 0.0
        #    self.w = 1.0
        #    self.scale = 1.0
        #    self.rotation = 0.0
        else:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
            self.w = float(w)
            self.scale = scale
            self.rotation = rotation
        
    def X(self):
        return float(self.x)/self.w
    
    def Y(self):
        return float(self.y)/self.w
    
    def Z(self):
        return float(self.z)/self.w
    
    def asArray(self,homogenious=False):
        '''
        returns the point data as a 4 element numpy array.
        
        if 'homogenious' == True: returns x,y,z,w
        else: return x,y,z,1.0
        '''
        if homogenious:
            return array([self.X(),self.Y(),self.Z(),1.0])
        else:
            return array([self.X(),self.Y(),self.Z(),1.0])
        
    
    def asVector2H(self):
        ''' Return a 2D homogenious vector [x,y,w] '''
        return array([[self.x],[self.y],[self.w]])
    
    def asVector3H(self):
        ''' Return a 3D homogenious vector [x,y,z,w] '''
        return array([[self.x],[self.y],[self.z],[self.w]])
    
    def asOpenCV(self):
        return (self.X(), self.Y()) #cv.cvPoint(int(round(self.X())),int(round(self.Y())))
    
    def asTuple(self):
        return (self.X(),self.Y())
    
    def asArray3D(self):
        return np.array((self.X(),self.Y(),self.Z()))
    
    def asSpherical(self):
        ''' 
        Computes and returns a representation of this point in spherical coordinates: (r,phi,theta). 
        
        r = radius or distance of the point from the origin.
        phi = is the angle of the projection on the xy plain and the x axis
        theta = is the angle with the z axis.
        
        x = r*cos(phi)*sin(theta)
        y = r*sin(phi)*sin(theta)
        z = r*cos(theta)
        '''
        x,y,z,_ = self.asArray()
        
        r = np.sqrt(x**2+y**2+z**2)
        phi = np.arctan2(y,x)
        theta = np.arctan2(np.sqrt(x**2+y**2),z)
        
        return r,phi,theta

    def l2(self,point):
        dx = self.X()-point.X()
        dy = self.Y()-point.Y()
        dz = self.Z()-point.Z()
        
        return sqrt(dx*dx + dy*dy + dz*dz)
    
    def unit(self):
        '''
        Returns a vector in the same direction but of unit length.
        '''
        x = self.X()
        y = self.Y()
        z = self.Z()
        l = np.sqrt(x*x+y*y+z*z)
        if l < 0.000001:
            # Point is at 0,0,0
            return pv.Point(0,0,0)
        
        return (1.0/l)*self 
    
    def magnitude(self):
        x = self.X()
        y = self.Y()
        z = self.Z()
        
        return np.sqrt(x**2+y**2+z**2)
        
        
    def __sub__(self,point):
        return Point(self.X()-point.X(),self.Y()-point.Y(),self.Z()-point.Z())
    
    def __add__(self,point):
        return Point(self.X()+point.X(),self.Y()+point.Y(),self.Z()+point.Z())
    
    def __mul__(self,val):
        if isinstance(val,float) or isinstance(val,int): 
            return Point(self.X()*val,self.Y()*val,self.Z()*val)

    def __rmul__(self,val):
        if isinstance(val,float) or isinstance(val,int): 
            return Point(self.X()*val,self.Y()*val,self.Z()*val)
    
    def __str__(self):
        return "pv.Point(%f,%f,%f)"%(self.X(),self.Y(),self.Z())
    
    def __repr__(self):
        return "pv.Point(%f,%f,%f)"%(self.X(),self.Y(),self.Z())
    
    
def readPointsFile(filename):
    '''
    This function reads a points file that was created by the EyePicker 
    application. EyePicker produces a csv file where each line corresponds
    to a file and can contain a number of points.
    
    This function returns a dictionary where the key is the filename and
    each entry contains a list of points. 
    '''
    f = csv.reader(open(filename,'rb'))
    
    result = {}
    
    for row in f:
        fname = row[0]
        
        row = row[1:]
        
        # Make sure the number of values is even
        assert len(row) % 2 == 0
        
        points = []
        for i in range(0,len(row),2):
            x = float(row[i])
            y = float(row[i+1])
            points.append(Point(x,y))
        
        result[fname] = points
        
    return result
    

    
    