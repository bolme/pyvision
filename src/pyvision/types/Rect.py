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


class Rect:
    def __init__(self,x=0.0,y=0.0,w=0.0,h=0.0):
        ''' 
        Create a point.
        
        Arguments:
        x: x coordinate
        y: y coordinate
        w: width
        h: height
        '''
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        
    def intersect(self, rect):
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
    
    def center(self):
        return pt.Point(self.x+0.5*self.w,self.y+0.5*self.h)
    
    def area(self):
        '''
        return the area of the rect
        '''
        return self.w*self.h
    
    def __str__(self):
        return "Rect(%f,%f,%f,%f)"%(self.x,self.y,self.w,self.h)
    
    def box(self):
        '''bounding box as expected by many PIL functions'''
        return int(round(self.x)), int(round(self.y)), int(round(self.x+self.w)), int(round(self.y+self.h))


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


