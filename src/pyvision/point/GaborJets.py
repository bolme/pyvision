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

'''
This is implemented based on Wiskott et.al. "Elastic Bunch Graph Matching" PAMI 1997
'''


import numpy as np
import unittest
import os.path
import pyvision as pv
from pyvision.analysis.face import EyesFile
#import time

class GaborWavelet:
    def __init__(self, freq, oreint, sigma):
        '''
        From Wiskott 96
        '''
        self.freq = freq
        self.oreint = oreint
        self.sigma = sigma
        
        self.k = np.array([freq*np.cos(oreint),freq*np.sin(oreint)],'f')
        self.k2 = np.dot(self.k,self.k)
        self.sigma2 = sigma*sigma
        
    def mask(self,size):
        w,h = size

        #m = np.zeros(size,np.complex64)
        x = np.arange(-w/2,w/2).reshape(w,1)*np.ones(size)
        x = np.concatenate((x[w/2:,:],x[:w/2,:]),axis=0)
        y = np.arange(-h/2,h/2).reshape(1,h)*np.ones(size)
        y = np.concatenate((y[:,h/2:],y[:,:h/2]),axis=1)

        m2 = self(x,y)
        return m2
        
    
    def __call__(self,x,y):
        if isinstance(x,int) and isinstance(y,int):
            #print "ints"
            x = np.array([x,y],'d')
            
            x2 = np.dot(x,x)
            k = self.k
            k2 = self.k2
            sigma2 = self.sigma2
            
            dot_kx = np.dot(k,x)
            tmp = (k2/sigma2)*np.exp(-k2*x2/(2.0*sigma2))*(np.exp(1j*dot_kx) - np.exp(-sigma2/2.0))
            return tmp
        else:
            x = np.array([x,y],'d')
            
            x2 = np.array(x*x).sum(axis=0)
            k = self.k
            k2 = self.k2
            sigma2 = self.sigma2
            
            dot_kx = np.array(k.reshape(2,1,1)*x).sum(axis=0)
            tmp = (k2/sigma2)*np.exp(-k2*x2/(2.0*sigma2))*(np.exp(1j*dot_kx) - np.exp(-sigma2/2.0))
            return tmp


class FilterBank:
    '''
    This class uses the FFT to quickly compute corelations and convolutions
    for an image.  The algorithm precomputes the filters in frequency space
    and uses only one FFT to to transform the image into frequency space.  
    '''
    def __init__(self,tile_size=(128,128),window=None,preprocess=None):
        self.tile_size = tile_size
        self.filters = []
        self.window = None
        self.preprocess = preprocess
        if window != None:
            self.window = window(tile_size)
    
    
    def addFilter(self,f,recenter=False):
        ''' 
        f can be a function f(x,y) is defined over x = (-w/2, w/2] and 
        y = (-h/2,h/2] and should be centered on the coord 0,0.
        
        TODO: At some point this function should be expanded to take filters 
        represented by arrays.
        '''
        if recenter == True:
            raise NotImplementedError
        if isinstance(f,GaborWavelet):
            filt = np.fft.fft2(f.mask(self.tile_size))
            self.filters.append(filt)
        else:
            w,h = self.tile_size
            m = np.zeros((w,h),np.complex64)
            for x in range(-w/2,w/2):
                for y in range(-h/2,h/2):
                    m[x,y] = f(x,y)
            filt = np.fft.fft2(m)
            self.filters.append(filt.conj())


    def convolve(self,im,ilog=None):
        if isinstance(im,pv.Image):
            im = im.asMatrix2D()
        
        w,h = self.tile_size
        assert im.shape[0] == w
        assert im.shape[1] == h
        
        if self.preprocess != None:
            im = self.preprocess(im)
            
        if self.window != None:
            im = self.window*im
            
        if ilog != None:
            ilog.log(pv.Image(im))
        
        c = len(self.filters)
        
        result = np.zeros((w,h,c),np.complex64)
        
        fft_image = np.fft.fft2(im)

        for i in range(c):
            product = self.filters[i]*fft_image
            result[:,:,i] = np.fft.ifft2(product)

        return result
    

def createGaborKernels():
    '''Create gabor wavelets from Wiskott 1999'''
    kernels = []
    sigma = 2*np.pi
    for freq in [np.pi*2.**(-(i+2.)/2.) for i in range(5)]:
        for orient in [i*np.pi/8. for i in range(8)]:
            #print "Freq: %8.5f  Orient: %8.5f   Sigma: %8.5f"%(freq,orient,sigma)
            wavelet = GaborWavelet(freq,orient,sigma)
            kernels.append(wavelet)
    
    return kernels
            

class GaborFilters:
    
    def __init__(self, kernels=createGaborKernels(), tile_size=(128,128),window=None,preprocess=None):
        self.kernels = kernels
        self.bank = FilterBank(tile_size=tile_size,window=window,preprocess=preprocess)
        self.k = np.zeros((len(kernels),2),dtype=np.float64)
        for i in range(len(kernels)):
            wavelet = kernels[i]
            self.bank.addFilter(wavelet)
            self.k[i,0] = wavelet.k[0]
            self.k[i,1] = wavelet.k[1]
        #print "K",self.k
    

    def convolve(self,im, ilog=None):       
        data = self.bank.convolve(im,ilog=ilog)
        return GaborImage(data,self.kernels,self.k) 

    
class GaborImage:
    def __init__(self,data,kernels,k):
        self.data = data
        self.kernels = kernels
        self.k = k

    def extractJet(self,pt,subpixel=True):
        x = int(np.round(pt.X()))
        y = int(np.round(pt.Y()))
        x = max(min(x,self.data.shape[0]-1),0)
        y = max(min(y,self.data.shape[0]-1),0)
        jet = self.data[x,y,:]

        return GaborJet(jet,self.kernels,self.k,x,y,pt,subpixel)
    
    def locatePoint(self,jet,start_pt=None,method="Simple"):
        '''
        If start_pt == None perform a grid search with a spacing of one half 
        the longest Gabor wavelength. Otherwize start at start_pt and follow
        the Jacobian to the local minimum.
        
        @param jet: the an example jet.
        @param start_pt The point to start the search from.
        '''
        if start_pt == None:
            # Compute the gate to use in the search
            kx = self.k[:,0]
            ky = self.k[:,1]
            kt = np.sqrt(kx*kx + ky*ky).min()
            gate = int(round(0.5*np.pi/kt))
            
            # search for best similarity
            
            best_sim = -1.0
            best_pt = pv.Point(0,0)
            w,h,_ = self.data.shape[:]
            for y in range(gate,h,gate):
                for x in range(gate,w,gate):
                    pt = pv.Point(x,y)
                    novel = self.extractJet(pt,subpixel=False)
                    sim = novel.simDisplace(jet)
                    if sim > best_sim:
                        best_sim = sim
                        best_pt = pt
                        
            start_pt = best_pt
        
        pt = start_pt
        
        #print pt

        novel = self.extractJet(pt,subpixel=False)
        d = novel.displace(jet,method=method)
        pt = pv.Point(novel.x + d[0], novel.y+d[1])
        #print pt

        novel = self.extractJet(pt,subpixel=False)
        d = novel.displace(jet,method=method)
        pt = pv.Point(novel.x + d[0], novel.y+d[1])
        #print pt

        novel = self.extractJet(pt,subpixel=False)
        d = novel.displace(jet,method=method)
        pt = pv.Point(novel.x + d[0], novel.y+d[1])
        #print pt

        novel = self.extractJet(pt,subpixel=False)
        sim = novel.simPhase(jet)
        #pt = pv.Point(novel.x + d[0], novel.y+d[1])
        #print pt
        
        return pt,sim,novel
    
    def show(self,*args,**kwargs):
        print(self.data.shape)
        tiles = []
        w,h,n = self.data.shape
        for i in range(n):
            mat = self.data[:,:,i]
            tiles.append(pv.Image(mat.real))
            tiles.append(pv.Image(mat.imag))
        mont = pv.ImageMontage(tiles,layout=(8,10),tileSize=(w,h))
        mont.show(*args,**kwargs)
        
    
class GaborJet:
    def __init__(self,jet,kernels,k,x,y,pt,subpixel):
        self.jet = jet
        self.kernels = kernels
        self.k = k
        self.x = x
        self.y = y
        
        re = np.real(jet)
        im = np.imag(jet)
        
        self.mag = np.sqrt(re*re + im*im)
        self.phase = np.arctan2(re,im)
        
        if subpixel:
            d = np.array([[pt.X()-x],[pt.Y()-y]])
            comp = np.dot(self.k,d)
            self.phase -= comp.flatten()
            self.jet = self.mag*np.exp(1.0j*self.phase)
            
        
    def displace(self,gj,method="Blocked",**kwargs):
        '''
        @param method: can be one of "Blocked", "Iter", "Simple"
        '''
        if method=="Blocked":
            return self.displaceBlocked(gj,**kwargs)
        elif method=="Iter":
            return self.displaceIter(gj,**kwargs)
        elif method=="Simple":
            return self.displaceSimple(gj,**kwargs)
        else:
            raise ValueError("Unknown displacement estimation method: %s"%method)

    def displaceSimple(self,gj):
        m1 = self.mag
        m2 = gj.mag
        p1 = self.phase
        p2 = gj.phase
        kx = self.k[:,0]
        ky = self.k[:,1]
        
        p = p1 - p2
        
        mask = p > np.pi
        p -= 2.0*np.pi * mask
        
        mask = p < -np.pi
        p += 2.0*np.pi * mask
        
         
        phi_x = (m1*m2*kx*p).sum()
        phi_y = (m1*m2*ky*p).sum()
        gam_xx = (m1*m2*kx*kx).sum()
        gam_xy = (m1*m2*kx*ky).sum()
        gam_yy = (m1*m2*ky*ky).sum()
        
        denom = gam_xx*gam_yy - gam_xy*gam_xy
        
        if denom == 0:
            print("Warning: divide by zero error in gabor displacement. returning (0.0,0.0)")
            return 0.,0.

        else:
            tmp1 = np.array([[gam_yy, -gam_xy],[-gam_xy,gam_xx]])
            tmp2 = np.array([[phi_x],[phi_y]])
            tmp = np.dot(tmp1,tmp2)/denom

            return tmp[0,0],tmp[1,0]


    def displaceBlocked(self,gj,block_size=8):
        m1 = self.mag
        m2 = gj.mag
        p1 = self.phase
        p2 = gj.phase
        kx = self.k[:,0]
        ky = self.k[:,1]
        k = self.k      
        d = np.array([[0.],[0.]])
        
        s = len(m1)
        bs = block_size
        nb = s/bs
        assert len(m1)%bs == 0
        
        for b in range(nb):
            correction = np.dot(k,d).flatten()   
            low,high = s-(b+1)*bs,s

            p = p1 - p2 - correction
            
            mask = p > np.pi
            while mask.max() == True:
                p -= 2.0*np.pi * mask
                mask = p > np.pi
            
            mask = p < -np.pi
            while mask.max() == True:
                p += 2.0*np.pi * mask
                mask = p < -np.pi
    
            phi_x = (m1*m2*kx*p)[low:high].sum()
            phi_y = (m1*m2*ky*p)[low:high].sum()
            gam_xx = (m1*m2*kx*kx)[low:high].sum()
            gam_xy = (m1*m2*kx*ky)[low:high].sum()
            gam_yy = (m1*m2*ky*ky)[low:high].sum()
            
            denom = gam_xx*gam_yy - gam_xy*gam_xy
            
            if denom == 0:
                print("Warning: divide by zero error in gabor displacement. returning (0.0,0.0)")
                return 0.,0.
    
            else:
                tmp1 = np.array([[gam_yy, -gam_xy],[-gam_xy,gam_xx]])
                tmp2 = np.array([[phi_x],[phi_y]])
                tmp = np.dot(tmp1,tmp2)/denom
                d += tmp
    
        return d[0,0],d[1,0]


    def displaceIter(self,gj,N=8):
        m1 = self.mag
        m2 = gj.mag
        p1 = self.phase
        p2 = gj.phase
        kx = self.k[:,0]
        ky = self.k[:,1]
        k = self.k      
        d = np.array([[0.],[0.]])
            
        for _ in range(N):
            correction = np.dot(k,d).flatten()   

            p = p1 - p2 - correction
            
            mask = p > np.pi
            while mask.max() == True:
                p -= 2.0*np.pi * mask
                mask = p > np.pi
            
            mask = p < -np.pi
            while mask.max() == True:
                p += 2.0*np.pi * mask
                mask = p < -np.pi
    
            phi_x = (m1*m2*kx*p).sum()
            phi_y = (m1*m2*ky*p).sum()
            gam_xx = (m1*m2*kx*kx).sum()
            gam_xy = (m1*m2*kx*ky).sum()
            gam_yy = (m1*m2*ky*ky).sum()
            
            denom = gam_xx*gam_yy - gam_xy*gam_xy
            
            if denom == 0:
                print("Warning: divide by zero error in gabor displacement. returning (0.0,0.0)")
                return 0.,0.
    
            else:
                tmp1 = np.array([[gam_yy, -gam_xy],[-gam_xy,gam_xx]])
                tmp2 = np.array([[phi_x],[phi_y]])
                tmp = np.dot(tmp1,tmp2)/denom
                d += tmp
    
        return d[0,0],d[1,0]


    def simMag(self,gj):
        '''
        Magnitude similarity measure.
        '''
        m1 = self.mag
        m2 = gj.mag
        
        tmp1 = (m1*m2).sum()
        tmp2 = (m1*m1).sum()
        tmp3 = (m2*m2).sum()
        
        return tmp1/np.sqrt(tmp2*tmp3)


    def simPhase(self,gj):
        '''
        Magnitude similarity measure.
        '''
        m1 = self.mag
        m2 = gj.mag
        p1 = self.phase
        p2 = gj.phase
        
        tmp1 = (m1*m2*np.cos(p1 - p2)).sum()
        tmp2 = (m1*m1).sum()
        tmp3 = (m2*m2).sum()
        
        return tmp1/np.sqrt(tmp2*tmp3)


    def simDisplace(self,gj,d=None):
        '''
        Displacement similarity measure.
        '''        
        m1 = self.mag
        m2 = gj.mag
        p1 = self.phase
        p2 = gj.phase
        k = self.k     
        
        if d:
            pass
        else:
            d = np.array(self.displace(gj)).reshape(2,1)
             
        correction = np.dot(k,d).flatten()   
        
        tmp1 = (m1*m2*np.cos(p1 - p2 - correction)).sum()
        tmp2 = (m1*m1).sum()
        tmp3 = (m2*m2).sum()
        
        return tmp1/np.sqrt(tmp2*tmp3)


class _FastFilterTest(unittest.TestCase):
    def setUp(self):
        SCRAPS_FACE_DATA = os.path.join(pv.__path__[0],"data","csuScrapShots")
        self.test_images = []
        self.eyes = EyesFile(os.path.join(SCRAPS_FACE_DATA,"coords.txt"))
        for filename in self.eyes.files()[0:10]:
            im = pv.Image(os.path.join(SCRAPS_FACE_DATA, filename + ".pgm"))
            eyes = self.eyes.getEyes(filename)
            #print eyes
            affine = pv.AffineFromPoints(eyes[0][0],eyes[0][1],pv.Point(40,40),pv.Point(88,40),(128,128))
            im = affine.transformImage(im)
            
            self.test_images.append(im)

    
    def test_gabor1(self):
        ilog = None # pv.ImageLog(name="GaborTest1")
        
        bank = FilterBank(tile_size=(128,128))
        kernels = createGaborKernels()
        for wavelet in kernels:
            bank.addFilter(wavelet)
            
        for i in range(len(bank.filters)):
            corr_filter = np.fft.ifft2(bank.filters[i])
            if ilog:
                ilog.log(pv.Image(np.fft.fftshift(corr_filter.real)),label="Filter_RE_%d"%i)
                ilog.log(pv.Image(np.fft.fftshift(corr_filter.imag)),label="Filter_IM_%d"%i)
            
        
        for im in self.test_images[:1]:
            if ilog:
                ilog.log(im,label="ORIG")
            results = bank.convolve(im)
            #print "RShape",results.shape[2]
            if ilog:
                for i in range(results.shape[2]):
                    ilog.log(pv.Image(results[:,:,i].real),label="CONV_RE_%d"%i)
                    ilog.log(pv.Image(results[:,:,i].imag),label="CONV_IM_%d"%i)
        if ilog:
            ilog.show()

        
    def test_gabor2(self):
        #print
        bank = FilterBank(tile_size=(128,128))
        #print "generating filters"
        freq = 5
        oreint = 3
        test_values = [[0.39182228, 0.39265844, 5.4866541e-07, 7.4505806e-08, -0.00035403497, -0.049227916], [0.39180198, 0.3926785, -1.0430813e-07, 4.4703484e-08, 0.017195048, -0.045993667], [0.39180198, 0.39267856, -5.9604645e-08, 7.4505806e-09, 0.045639627, 0.017549083], [0.1959009, 0.19633935, -1.0430813e-07, -1.4901161e-08, -0.054863166, -0.010506729], [0.19590084, 0.19633923, -1.0337681e-07, -3.7252903e-08, -0.050385918, -0.024042567], [0.19590083, 0.19633923, -1.8626451e-07, -2.6077032e-08, 0.053237237, 0.014138865], [0.097950377, 0.098169744, 7.5995922e-07, 4.0978193e-08, -0.029739097, 0.029439665], [0.097950377, 0.098169573, 8.5681677e-08, 7.4505806e-09, -0.034587309, 0.023616293], [0.097950369, 0.098169573, 3.7252903e-09, 2.7939677e-08, 0.040645007, 0.0075459499], [0.048975188, 0.049084734, -1.0859221e-06, 1.4528632e-07, -0.0026100441, 0.025389811], [0.048975196, 0.049084827, -9.3132257e-09, 3.3527613e-08, -0.0058529032, 0.02486741], [0.048975192, 0.049084831, -7.6368451e-08, -4.703179e-08, 0.025110571, 0.0032778508], [0.024487557, 0.024542304, -0.00027022697, 1.1050142e-06, 0.0053004427, 0.013041494], [0.024487549, 0.024542348, -4.4152141e-05, 3.0389987e-05, 0.0040912526, 0.013478963], [0.024487551, 0.024542345, -4.4191256e-05, 3.0397903e-05, 0.013955923, 0.001284558]]
        i = 0
        for f in range(freq):
            #f=f+2
            for o in range(oreint):
                #print "F:%10.5f O:%10.5f"%(f,o)
                w1 = GaborWavelet(np.pi*2.0**(-(f+2.0)/2.0),o*np.pi/oreint,np.pi)
                mask = w1.mask((64,64))
                ssr = (np.real(mask)**2).sum()
                ssi = (np.imag(mask)**2).sum()
                sr = np.real(mask).sum()
                si = np.imag(mask).sum()
                
                self.assertAlmostEqual(test_values[i][0],ssr,places=5)
                self.assertAlmostEqual(test_values[i][1],ssi,places=5)
                self.assertAlmostEqual(test_values[i][2],sr,places=5)
                self.assertAlmostEqual(test_values[i][3],si,places=5)
                self.assertAlmostEqual(test_values[i][4],np.real(mask)[3,2],places=5)
                self.assertAlmostEqual(test_values[i][5],np.imag(mask)[3,2],places=5)

                bank.addFilter(w1)
                i+=1
                #mask = w1.mask((128,128))
                #Image(real(mask)).show()
                #Image(imag(mask)).show()
        #print "Filter Time:",stop-start
                
            


    def test_GaborFilters(self):
        
        #bank = FilterBank(tile_size=(128,128))
        kernels = createGaborKernels()
        filters = GaborFilters(kernels)
        
        gim = filters.convolve(self.test_images[0])
        
        template = gim.extractJet(pv.Point(64,64))
        
        table = pv.Table()
        
        for i in range(0,128):
            table[i,'disp'] = i - 64
            novel = gim.extractJet(pv.Point(i,64))
            table[i,'simMag'] = template.simMag(novel)
            table[i,'simPhase'] = template.simPhase(novel)
            table[i,'simDisplace'] = template.simDisplace(novel)
            dx,dy = template.displace(novel,method="Simple")
            table[i,'displace_dx'] = dx
            table[i,'displace_dy'] = dy
            dx,dy = template.displace(novel,method="Blocked")
            table[i,'blocked_dx'] = dx
            table[i,'blocked_dy'] = dy
            dx,dy = template.displace(novel,method="Iter")
            table[i,'iter_dx'] = dx
            table[i,'iter_dy'] = dy
        
        #print
        #print table
        #table.save("../../gabor_plot.csv")
      
    def test_GaborImage(self):
        
        kernels = createGaborKernels()
        filters = GaborFilters(kernels)
        
        gim = filters.convolve(self.test_images[0])
        
        test_point = pv.Point(62.6,64.8)
        template = gim.extractJet(test_point)
        
        new_point,_,_ = gim.locatePoint(template,pv.Point(60,70))
        self.assertAlmostEqual(new_point.l2(test_point),0.0)

        new_point,_,_ = gim.locatePoint(template,pv.Point(30,49))        
        self.assertTrue(new_point.l2(test_point) > 1.0)
        
        new_point,_,_ = gim.locatePoint(template)        
        self.assertAlmostEqual(new_point.l2(test_point),0.0)
        
        
      
                
                
        
