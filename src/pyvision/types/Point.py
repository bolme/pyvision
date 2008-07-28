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
    
    def asVector2H(self):
        ''' Return a 2D homogenious vector [x,y,w] '''
        return array([[self.x],[self.y],[self.w]])
    
    def asVector3H(self):
        ''' Return a 3D homogenious vector [x,y,z,w] '''
        return array([[self.x],[self.y],[self.z],[self.w]])
    
    def l2(self,point):
        dx = self.X()-point.X()
        dy = self.Y()-point.Y()
        dz = self.Z()-point.Z()
        
        return sqrt(dx*dx + dy*dy + dz*dz)
    
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
        return "Point(%f,%f,%f)"%(self.X(),self.Y(),self.Z())
    
    def __repr__(self):
        return "Point(%f,%f,%f)"%(self.X(),self.Y(),self.Z())
        
        return "Point(%f,%f,%f)"%(self.X(),self.Y(),self.Z())