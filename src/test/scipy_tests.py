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

import unittest

from scipy import array, dot, identity
from scipy.linalg import svd, eig, inv
from scipy.fftpack import fft2


class TestScipy(unittest.TestCase):
    
    def setUp(self):
        self.A = array([[1,3],[4,1]],'f')
        self.x = array([[2],[5]],'d')
        self.a = array([1,2,3,4],'f')
        self.b = array([5,6,7,8],'d')
    
    def test_scipy_create(self):
        tmp = self.a
        self.assertAlmostEqual( tmp[0], 1.0 )
        self.assertAlmostEqual( tmp[1], 2.0 )
        self.assertAlmostEqual( tmp[2], 3.0 )
        self.assertAlmostEqual( tmp[3], 4.0 )

    def test_scipy_add(self):
        tmp = self.a + self.b
        
        self.assertAlmostEqual( tmp[0], 6.0 )
        self.assertAlmostEqual( tmp[1], 8.0 )
        self.assertAlmostEqual( tmp[2], 10.0 )
        self.assertAlmostEqual( tmp[3], 12.0 )
        
    def test_scipy_sub(self):
        tmp = self.a - self.b
        
        self.assertAlmostEqual( tmp[0], -4.0 )
        self.assertAlmostEqual( tmp[1], -4.0 )
        self.assertAlmostEqual( tmp[2], -4.0 )
        self.assertAlmostEqual( tmp[3], -4.0 )

    def test_scipy_mul(self):
        tmp = self.a * self.b
        
        self.assertAlmostEqual( tmp[0], 5.0 )
        self.assertAlmostEqual( tmp[1], 12.0 )
        self.assertAlmostEqual( tmp[2], 21.0 )
        self.assertAlmostEqual( tmp[3], 32.0 )

    def test_scipy_div(self):
        tmp = self.a / self.b
        
        self.assertAlmostEqual( tmp[0], 1.0/5.0 )
        self.assertAlmostEqual( tmp[1], 2.0/6.0 )
        self.assertAlmostEqual( tmp[2], 3.0/7.0 )
        self.assertAlmostEqual( tmp[3], 4.0/8.0 )
        
    def test_scipy_pow(self):
        tmp = self.a ** self.b
        
        self.assertAlmostEqual( tmp[0], 1.0 )
        self.assertAlmostEqual( tmp[1], 64.0 )
        self.assertAlmostEqual( tmp[2], 2187.0 )
        self.assertAlmostEqual( tmp[3], 65536.0 )
        
    def test_scipy_dot(self):
        tmp = dot(self.A, self.x)
        
        self.assertAlmostEqual( tmp[0,0], 17.0 )
        self.assertAlmostEqual( tmp[1,0], 13.0 )

    def test_scipy_svd(self):
        U,D,Vt = svd(self.A)
        
        D = array([[D[0],0],[0,D[1]]],'d')
        
        self.assertAlmostEqual( sum(sum(abs(dot(U,U.transpose())- identity(2)))), 0.0, 5)
        self.assertAlmostEqual( sum(sum(abs(dot(Vt,Vt.transpose())- identity(2)))), 0.0, 5)
        self.assertAlmostEqual( sum(sum(abs(dot(U,dot(D,Vt)) - self.A))), 0.0, 5)

    def test_scipy_inv(self):
        invA = inv(self.A)
                
        self.assertAlmostEqual( sum(sum(abs(dot(invA,self.A)-identity(2)))), 0.0, 5)
        self.assertAlmostEqual( sum(sum(abs(dot(self.A,invA)-identity(2)))), 0.0, 5)
        
        



