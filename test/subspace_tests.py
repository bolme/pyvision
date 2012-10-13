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


from pylab import *
from time import sleep
from numpy.random import normal
from numpy import *
from scipy.linalg import eig



def lda():
    ''' TODO: Simple LDA test code.  This should be changed into a nice class.'''
    n = 20

    data1 =  normal(size=(2,n/2))+array([[1.5],[0]])
    data2 =  normal(size=(2,n/2))+array([[-1.5],[0]])
    
    data = concatenate([data1,data2],axis=1)
    print data

    scale = [[0.5,0],[0,4]]
    theta = 13*3.14/8.0
    rotate = [[cos(theta),-sin(theta)],[sin(theta),cos(theta)]]
    X1 = dot(rotate,dot(scale,data1))
    X2 = dot(rotate,dot(scale,data2))
    X = dot(rotate,dot(scale,data))
    
    m1 = X1.mean(axis=1).reshape([2,1])
    m2 = X2.mean(axis=1).reshape([2,1])
    m = X.mean(axis=1).reshape([2,1])
    print "m1",m1
    print "m2",m2
    
    St = dot(X,X.transpose())
    S1 = dot((X1-m1),(X1-m1).transpose())
    S2 = dot((X2-m2),(X2-m2).transpose())
    Sw = S1+S2
    Sb = St-Sw
    
    print "S1,S2,Sw"
    print S1
    print S2
    print Sw
    print Sb
    
    print help(eig)
    val,pc = eig(Sb,Sw)
       
    val = real(val) 
    
    print 
    print "w,vr"
    print val
    print pc
    plot(X1[0],X1[1],'r+',X2[0],X2[1],'b+',[0,3*pc[0,0]],[0,3*pc[1,0]],'g-',[0,3*pc[0,1]],[0,3*pc[1,1]],'g-')
    text(3*pc[0,0],3*pc[1,0],"%0.2f"%val[0],color='g')
    text(3*pc[0,1],3*pc[1,1],"%0.2f"%val[1],color='g')
    
    axis([-10,10,-10,10])
    title("LDA Test")
    show()

if __name__ == '__main__':
    lda()