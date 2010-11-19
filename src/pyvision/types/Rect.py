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

import Point as pt
#import cv
import pyvision as pv
import numpy as np

def BoundingRect(*points):
    '''
    Create a rectangle that includes all of the points.
    '''
    tmp = []
    for each in points:
        if type(each) == list or type(each) == tuple:
            tmp += each
        else:
            tmp.append(each)
    assert len(tmp) > 0
    minx = maxx = tmp[0].X()
    miny = maxy = tmp[0].Y()
    for each in tmp:
        minx = min(minx,each.X())
        maxx = max(maxx,each.X())
        miny = min(miny,each.Y())
        maxy = max(maxy,each.Y())
    return Rect(minx,miny,maxx-minx,maxy-miny)

def CenteredRect(cx,cy,w,h):
    return pv.Rect(cx-0.5*w,cy-0.5*h,w,h)


class Rect:
    '''
    This is a simple structure that represents a rectangle.
    '''
    
    def __init__(self,x=0.0,y=0.0,w=0.0,h=0.0):
        ''' 
        Initialize a rectangle instance.
        
        Arguments:
        @param x: top left x coordinate
        @param y: top left y coordinate
        @param w: width
        @param h: height
        '''
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        
    def intersect(self, rect):
        '''
        Compute the intersection of two rectangles.
        
        @returns: a rectangle representing the intersection.
        '''
        r1 = self
        r2 = rect
        
        r1_x1 = r1.x
        r1_x2 = r1.x + r1.w
        r1_y1 = r1.y
        r1_y2 = r1.y + r1.h

        r2_x1 = r2.x
        r2_x2 = r2.x + r2.w
        r2_y1 = r2.y
        r2_y2 = r2.y + r2.h
        
        #print r1_x1,r1_x2,r1_y1,r1_y2
        #print r2_x1,r2_x2,r2_y1,r2_y2
        
        r3_x1 = max(r1_x1,r2_x1)
        r3_x2 = min(r1_x2,r2_x2)
        r3_y1 = max(r1_y1,r2_y1)
        r3_y2 = min(r1_y2,r2_y2)

        #print r3_x1,r3_x2,r3_y1,r3_y2
        r3_w = r3_x2-r3_x1 
        r3_h = r3_y2-r3_y1
        
        if r3_w < 0.0 or r3_h < 0.0:
            return None
        
        return Rect(r3_x1,r3_y1,r3_w, r3_h)
    
    def containsRect(self,rect):
        '''
        Determines if rect is entirely within (contained by) this rectangle.
        @param rect: an object of type pv.Rect
        @return: True if the rect is entirely within this rectangle's boundaries.
        '''
        t1 = (self.x <= rect.x) 
        t2 = (self.y <= rect.y)
        t3 = ( (self.x+self.w) >= (rect.x+rect.w) )
        t4 = ( (self.y+self.h) >= (rect.y+rect.h) )
        if( t1 and t2 and t3 and t4):
            return True
        else:
            return False        
    
    def containsPoint(self,point):
        '''
        Determine if a point is within a rectangle.
        
        @param point: an object of type pv.Point.
        @returns: True if the point is withen the Rect.
        '''
        x = point.X()
        y = point.Y()
        
        return x >= self.x and x <= self.x+self.w and y >= self.y and y <= self.y + self.h
        
    def center(self):
        '''
        Compute and return a point at the center of the rectangle
        
        @returns: a pv.Point at the center.
        '''
        return pt.Point(self.x+0.5*self.w,self.y+0.5*self.h)
    
    def area(self):
        '''
        @returns: the area of the rect
        '''
        return self.w*self.h
    
    def overlap(self,rect2):
        '''
        Compute an overlap measure for two detection rectangles.
        '''
        i = self.intersect(rect2)   # Compute the intersection
        if i == None:
            return 0.0
        u = self.area() + rect2.area() - i.area() # Compute the union
        return i.area()/u

    
    def similarity(self,rect):
        '''
        Compute the similarity of the rectangles in terms of overlap.
        '''
        i = self.intersect(rect)
        if i == None:
            return 0.0
        return i.area() / (0.5*self.area() + 0.5*rect.area())
        
    
    def rescale(self,scale):
        ''' 
        Expand or contract the size of the rectangle by a "scale" while
        keeping the Rect centered at the same location.
        
        @param scale: the scale factor
        @returns: the rescaled rect
        '''
        center = self.center()
        cx,cy = center.X(),center.Y()
        w = scale*self.w
        h = scale*self.h
        return Rect(cx-0.5*w,cy-0.5*h,w,h)
    
    def asInt(self):
        '''
        Return a dictionary representing the rectangle with integer values
        '''
        x = int(np.floor(self.x))
        y = int(np.floor(self.y))
        w = int(np.floor(self.w))
        h = int(np.floor(self.h))
        return {'x':x,'y':y,'w':w,'h':h}
    
    def __str__(self):
        '''
        @returns: a string representing this rectangle
        '''
        return "pv.Rect(%f,%f,%f,%f)"%(self.x,self.y,self.w,self.h)
    
    def __repr__(self):
        '''
        @returns: a string representing this rectangle
        '''
        return "pv.Rect(%f,%f,%f,%f)"%(self.x,self.y,self.w,self.h)
    
    def box(self):
        '''
        Get this rectangle as a bounding box as expected by many PIL functions.
        
        @returns: tuple of (left,top,right,bottom)
        '''
        return int(round(self.x)), int(round(self.y)), int(round(self.x+self.w)), int(round(self.y+self.h))

    def asOpenCV(self):
        '''
        @returns a representation compatible with opencv.
        '''
        return (int(round(self.x)),int(round(self.y)),int(round(self.w)),int(round(self.h)))

    def asTuple(self):
        '''
        @returns a tuple (x,y,w,h).
        '''
        return (self.x,self.y,self.w,self.h)

    def asCenteredTuple(self):
        '''
        @returns a tuple (cx,cy,w,h).
        '''
        return (self.x+0.5*self.w,self.y+0.5*self.h,self.w,self.h)
    
    def asCorners(self):
        '''
        Returns the four corners.  Can be used to transform this rect 
        through an affine or perspective transformation.
        '''
        x,y,w,h = self.asTuple()
        pt1 = pv.Point(x,y)
        pt2 = pv.Point(x+w,y)
        pt3 = pv.Point(x+w,y+h)
        pt4 = pv.Point(x,y+h)
        return [pt1,pt2,pt3,pt4]

    def asPolygon(self):
        '''
        Returns the four corners with the upper left corner repeated twice.  
        Can be used to transform this rect through an affine or perspective 
        transformations. It can also be plotted with annotatePolygon.
        '''
        x,y,w,h = self.asTuple()
        pt1 = pv.Point(x,y)
        pt2 = pv.Point(x+w,y)
        pt3 = pv.Point(x+w,y+h)
        pt4 = pv.Point(x,y+h)
        return [pt1,pt2,pt3,pt4,pt1]

    def __mul__(self,val):
        '''
        Multiply the rectangle by a constant.
        '''
        if isinstance(val,float) or isinstance(val,int): 
            return Rect(self.x*val,self.y*val,self.w*val,self.h*val)

    def __rmul__(self,val):
        '''
        Multiply the rectangle by a constant.
        '''
        if isinstance(val,float) or isinstance(val,int): 
            return Rect(self.x*val,self.y*val,self.w*val,self.h*val)


def test():
    '''
    '''
    import Point
    p1 = Point.Point(1,1)
    p2 = Point.Point(4,4)
    p3 = Point.Point(5,4)
    p4 = Point.Point(6,8)

    r1 = BoundingRect(p1,p2)
    r2 = BoundingRect(p3,p4)
    r3 = Rect(3,3,3,3)
    print r1
    print r2
    print r1.intersect(r2)
    print r3.intersect(r2)
    
if __name__ == "__main__":
    test()


