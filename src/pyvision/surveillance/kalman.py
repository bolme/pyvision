# PyVision License
#
# Copyright (c) 2010 David S. Bolme
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
Created on Sep 27, 2010

@author: bolme
'''

import numpy as np

class KalmanFilter:
    
    def __init__(self,x_0,P_0,F,Q,H,R,B=None):
        '''
        Based on wikipedia.  
        
        Warning: Controls have not been tested.
        
        @param x_0: initial state
        @param P_0: initial state covariance estimate
        @param F: the state transistion model
        @param B: control input model
        @param Q: process noise covariance
        @param z_k: measurement
        @param H: observation model - transform state to observation
        @param R: measurement noise covariance
        not needed
        @param w: process noise
        @param v_k: measurement noise
        '''
        
        self.F = np.array(F,dtype=np.float64)
        d = self.F.shape[0]
        assert self.F.shape == (d,d)
        
        self.P_k = np.array(P_0,dtype=np.float64)
        assert self.P_k.shape == (d,d)

        if B == None:
            B = np.zeros((d,1))
        self.B = np.array(B,dtype=np.float64)
        b = self.B.shape[1]
        assert self.B.shape == (d,b)

        self.Q = np.array(Q,dtype=np.float64)
        assert self.Q.shape == (d,d)

        self.H = np.array(H,dtype=np.float64)
        m = self.H.shape[0]
        assert self.H.shape == (m,d)

        self.R = np.array(R,dtype=np.float64)
        assert self.R.shape == (m,m)

        self.x_k = np.array(x_0,dtype=np.float64).reshape(d,1)
        
        self.d = d
        self.b = b
        self.m = m
        
        #self.history = []
        
        
    def predict(self, u=None):
        # Check and convert inputs
        m = self.m
        b = self.b
        d = self.d
                
        if u == None:
            u = np.zeros((b,))
        u = np.array(u,dtype=np.float64).reshape(b,1)

        x_k = np.dot(self.F,self.x_k) + np.dot(self.B,u)
        return x_k
        
                
    def update(self, z=None, u=None):
        '''
        @param z: measurement
        @param u: control
        '''
        # Check and convert inputs
        m = self.m
        b = self.b
        d = self.d
        
        
        if u == None:
            u = np.zeros((b,))
        u = np.array(u,dtype=np.float64).reshape(b,1)
        
        # Predict step
        x_k = np.dot(self.F,self.x_k) + np.dot(self.B,u)
        #print "Pred X:\n",x_k
        
        P_k = np.dot(np.dot(self.F,self.P_k),self.F.T) + self.Q
        #print "Pred P:\n",P_k

        if z == None:
            self.x_k = x_k
            self.P_k = P_k
            return self.x_k

        z = np.array(z,dtype=np.float64).reshape(m,1)
        
        # Update Step
        y_k = z - np.dot(self.H,x_k)
        #print "Innovation Resid:\n",y_k
        
        S_k = np.dot(np.dot(self.H,P_k),self.H.T) + self.R
        #print "Innonvation Cov:\n",S_k
        
        K = np.dot(np.dot(P_k,self.H.T),np.linalg.inv(S_k))
        #print "Kalman Gain:\n",K

        x_k = x_k + np.dot(K,y_k)
        #print "Postieriori state:\n",x_k
        
        P_k = np.dot(np.eye(d) - np.dot(K,self.H),P_k)
        #print "Posteriori Cov:",P_k
        
        self.x_k = x_k
        self.P_k = P_k
        
        return self.x_k
    
    def state(self):
        ''' Return the current estimate of state. '''
        return self.x_k
    
    def setState(self,x_k):
        self.x_k = np.array(x_k,np.float64)
    
    def __str__(self):
        return "KalmanFilter<%s>"%self.state().flatten()
        
    
        