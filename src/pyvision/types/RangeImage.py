# PyVision License
#
# Copyright (c) 2009 David S. Bolme
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


import gzip
import numpy as np
import time
import pyvision as pv
import scipy.interpolate as it
import scipy.ndimage as nd

class RangeImage:
    '''
    This class is used to handle range images. Originally written to handle
    output from the Minolta Vivid sensors distributed with the Face Recognition
    Grand Challenge 2004
    
    This implementation currently can parse range images in ".abs" or ".abs.gz" format.
    
    Very little type checking is done during parsing so unexpected exception or
    unusual behavior may occur if the file is not formated properly.
    
    This is a sample for the .abs file format:
    480 rows
    640 columns
    pixels (flag X Y Z):
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ...
    -999999.000000 -999999.000000 -999999.000000 -9999 ...
    -999999.000000 -999999.000000 -999999.000000 -9999 ...
    -999999.000000 -999999.000000 -999999.000000 -9999 ...
    
    Author: David S. Bolme 2009
    '''
    
    def __init__(self,filename):
        '''
        Reads a file containing range data.
        '''

        if filename[-7:] == '.abs.gz':
            # Assume the date is a zlib compressed abs file 
            f = gzip.open(filename)
            
        if filename[-4:] == '.abs':
            # Assume uncompressed
            f = open(filename)
        
        #print buffer[:100]
        
        #lines = buffer.split(EOL)
        rows = int(f.next().split()[0])
        cols = int(f.next().split()[0])
        format = f.next()
        
        self.width = cols
        self.height = rows
        
        self.flags = np.array([int(v) for v in f.next().split()]).reshape(rows,cols).transpose()
        self.x = np.array([float(v) for v in f.next().split()]).reshape(rows,cols).transpose()
        self.y = np.array([float(v) for v in f.next().split()]).reshape(rows,cols).transpose()
        self.z = np.array([float(v) for v in f.next().split()]).reshape(rows,cols).transpose()
        
        
    def getRange(self):
        '''
        @returns: xmin,xmax,ymin,ymax,zmin,zmax
        '''
        
        flags = np.array(self.flags.flatten(),dtype=np.bool)

        X = self.x.flatten()[flags]
        Y = self.y.flatten()[flags]
        Z = self.z.flatten()[flags]
        return min(X),max(X),min(Y),max(Y),min(Z),max(Z)    
    
    
    def getXImage(self):
        '''
        @returns: the x coordinates.
        '''
        xmin,xmax,ymin,ymax,zmin,zmax = self.getRange()
        
        r,c = self.x.shape 
        
        flags = np.array(self.flags.flatten(),dtype=np.bool)
        X = self.x.flatten().copy()
        X[ flags != True ] = xmin
        
        X = X.reshape(r,c)
        
        return pv.Image(X)
        
        
    def getYImage(self):
        '''
        @returns: the y coordinates.
        '''
        xmin,xmax,ymin,ymax,zmin,zmax = self.getRange()
        
        r,c = self.x.shape 
        
        flags = np.array(self.flags.flatten(),dtype=np.bool)
        Y = self.y.flatten().copy()
        Y[ flags != True ] = ymin
        
        Y = Y.reshape(r,c)
        
        return pv.Image(Y)


    def getZImage(self):
        '''
        @returns: the z coordinates.
        '''
        xmin,xmax,ymin,ymax,zmin,zmax = self.getRange()
        
        r,c = self.x.shape 
        
        flags = np.array(self.flags.flatten(),dtype=np.bool)
        Z = self.z.flatten().copy()
        Z[ flags != True ] = zmin
        
        Z = Z.reshape(r,c)
        
        return pv.Image(Z)

    def getMaskImage(self):
        '''
        @returns: the missing value mask.
        '''
        xmin,xmax,ymin,ymax,zmin,zmax = self.getRange()
        
        r,c = self.x.shape 
        
        flags = np.array(self.flags.flatten(),dtype=np.bool)
        Z = self.z.flatten().copy()
        Z[ flags != True ] = zmin
        
        Z = Z.reshape(r,c)
        
        return pv.Image(Z)
    
    def populateMissingData(self,approach="Smooth",ilog=None):
        '''
        This function is used to interpolate missing data in the image.
        '''
        if approach == 'Smooth':
            # first run a median filter over the array, then smooth the result.
            xmin,xmax,ymin,ymax,zmin,zmax = self.getRange()
            mask = np.array(self.flags,dtype=np.bool)
            
            z = self.getZImage().asMatrix2D()
            median = nd.median_filter(z,size=(15,15))

            mask = mask.flatten()
            z = z.flatten()
            median = median.flatten()

            z[ mask==False ] = median[ mask==False ]
            
            if ilog != None:
                ilog.log(pv.Image(median.reshape(self.width,self.height)),label="Median")
                ilog.log(pv.Image(z.reshape(self.width,self.height)),label="ZMedian")
            
            mask = mask.flatten()
            z = z.flatten()
            median = median.flatten()
            
            for i in range(5):
                tmp = z.copy()
                smooth = nd.gaussian_filter(z.reshape(self.width,self.height),2.0).flatten()
                z[ mask==False ] = smooth[ mask==False ]
                print "Iteration:",i,(z-tmp).max(),(z-tmp).min()
                ilog.log(pv.Image(z.reshape(self.width,self.height)),label="ZSmooth%02d"%i)
                ilog.log(pv.Image((z-tmp).reshape(self.width,self.height)),label="ZSmooth%02d"%i)
                
                
        if approach == 'RBF':
            mask = np.array(self.flags,dtype=np.bool)
            mask = mask.flatten()
    
            x = np.arange(self.width).reshape(self.width,1)
            x = x*np.ones((1,self.height))
            x = x.flatten()
    
            y = np.arange(self.height).reshape(1,self.height)
            y = y*np.ones((self.width,1))
            y = y.flatten()
            
            z = self.z.copy()
            z = z.flatten()
            
            print "Coords:"
            print len(mask)
            print len(x[mask])
            print len(y[mask])
            print len(z[mask])
            
            # this produces an error.  Probably has too much data
            it.Rbf(x[mask],y[mask],z[mask])
            pass
        
        
                
            
            
        
if __name__ == "__main__":
    ilog = pv.ImageLog()
    filename = "02463d562.abs.gz"
    im = pv.Image("02463d563.ppm")
    
    t = time.time()
    ri = RangeImage(filename)
    t = time.time() - t
    print t
    
    print ri.getRange()
    ilog.log(ri.getXImage(),"X_Image")
    ilog.log(ri.getYImage(),"Y_Image")
    ilog.log(ri.getZImage(),"Z_Image")
    ilog.log(im,"Color")
    
    ri.populateMissingData(ilog=ilog)
    
    ilog.show()
    
    
    
    
    