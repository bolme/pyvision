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
This module contains a methods for finding eyes in a face image.
The images passed to these methods are cropped images that have
come from a face detector.
'''

from pyvision.types.Point import Point
from pyvision.vector import SVM
    
    
class SVMLocator:
    
    def __init__(self,type=SVM.TYPE_NU_SVR,**kwargs):
        self.x_svm = SVM.SVM(type=type,**kwargs)
        self.y_svm = SVM.SVM(type=type,**kwargs)
    
    def addTraining(self,image,location):
        '''
        Pass in an image that is roughly centered on the feature,
        and a true location of that feature in the image.
        '''
        self.x_svm.addTraining(location.X(),image)
        self.y_svm.addTraining(location.Y(),image)

        
    def train(self):
        # compute the mean location
        self.x_svm.train()
        self.y_svm.train()

        
    def predict(self,image):
        x = self.x_svm.predict(image)
        y = self.y_svm.predict(image)

        return Point(x,y)
        
    
    
