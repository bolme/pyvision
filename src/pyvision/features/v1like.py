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
This module contains code from Nicolas Pinto.  Added to PyVision by David Bolme

Pinto N, Cox DD, DiCarlo JJ (2008) Why is Real-World Visual Object Recognition Hard?
PLoS Computational Biology 4(1): e27 doi:10.1371/journal.pcbi.0040027

Created on May 29, 2011

@author: bolme
'''

#import scipy as sp
import numpy as np
import scipy.signal
conv = scipy.signal.convolve
import pyvision as pv

class V1Processing:
    
    def __init__(self,params,featsel):
        '''
        @param params: representation parameters
        @type params: dict
        @param featsel: features to include in the vector
        @type featsel: dict
        '''
        self.params = params
        self.featsel = featsel
        
    def extractFromImage(self,im):
        '''
        Extract V1 like features from an image.
        
        @param im: the image
        @type im: pv.Image
        '''
        #image should have channels as third axis
        mat = im.asMatrix3D()
        mat = mat.transpose((1,2,0))
        out = v1like_fromarray(mat,self.params)
        return out

#============================================================================
# Pinto's code below here
#============================================================================
class MinMaxError(Exception): pass

V1LIKE_PARAMS_A = {
    'preproc': {
        'max_edge': 150,
        'lsum_ksize': 3,
        'resize_method': 'bicubic',            
        },
    'normin': {
        'kshape': (3,3),
        'threshold': 1.0,
        },
    'filter': {
        'kshape': (43,43),
        'orients': [ o*np.pi/16 for o in xrange(16) ],
        'freqs': [ 1./n for n in [2, 3, 4, 6, 11, 18] ],
        'phases': [0],
        },
    'activ': {
        'minout': 0,
        'maxout': 1,
        },
    'normout': {
        'kshape': (3,3),
        'threshold': 1.0,
        },
    'pool': {
        'lsum_ksize': 17,
        'outshape': (30,30),
        },
}

V1LIKE_FEATURES_A = {
    'output': True,
    'input_gray': None,
    'input_colorhists': None,
    'normin_hists': None,
    'filter_hists': None,
    'activ_hists': None,
    'normout_hists': None,
    'pool_hists': None,
    }

# ------------------------------------------------------------------------------
def gray_convert(arr):
    #assert(arr.min()>=0 and arr.max()<=1)
    # grayscale conversion
    out = 0.2989*arr[:,:,0] + \
        0.5870*arr[:,:,1] + \
        0.1141*arr[:,:,2]
    #out.shape = out.shape + (1,)    
    return out


# ------------------------------------------------------------------------------
def opp_convert(arr):
    #assert(arr.min()>=0 and arr.max()<=1)
    out = np.empty_like(arr)

    # red-green
    out[:,:,0] = arr[:,:,0] - arr[:,:,1]
    # blue-yellow
    out[:,:,1] = arr[:,:,2] - arr[:,:,[0,1]].min(2)
    # intensity
    out[:,:,2] = arr.max(2)

    return out

# ------------------------------------------------------------------------------
def oppnorm_convert(arr, threshold=0.1):
    #assert(arr.min()>=0 and arr.max()<=1)
    #out = sp.empty_like(arr)
    arr = arr.astype('float32')
    out = np.empty(arr.shape[:2]+(2,), dtype='float32')

    print out.shape

    # red-green
    out[:,:,0] = arr[:,:,0] - arr[:,:,1]
    # blue-yellow
    out[:,:,1] = arr[:,:,2] - arr[:,:,[0,1]].min(2)
    # intensity
    denom = arr.max(2)

    mask = denom < threshold#*denom[:,:,2].mean()
    
    out[:,:,0] /= denom    
    out[:,:,1] /= denom

    np.putmask(out[:,:,0], mask, 0)
    np.putmask(out[:,:,1], mask, 0)

    return out

# ------------------------------------------------------------------------------
def chrom_convert(arr):
    #assert(arr.min()>=0 and arr.max()<=1)

    opp = opp_convert(arr)
    out = np.empty_like(opp[:,:,[0,1]])

    rg = opp[:,:,0]
    by = opp[:,:,1]
    intensity = opp[:,:,2]

    lowi = intensity < 0.1*intensity.max()
    rg[lowi] = 0
    by[lowi] = 0

    denom = intensity
    denom[denom==0] = 1
    out[:,:,0] = rg / denom
    out[:,:,1] = by / denom

    return out

# ------------------------------------------------------------------------------
def rg2_convert(arr):
    #assert(arr.min()>=0 and arr.max()<=1)

    out = np.empty_like(arr[:,:,[0,1]])

    red = arr[:,:,0]
    green = arr[:,:,1]
    #blue = arr[:,:,2]
    intensity = arr.mean(2)

    lowi = intensity < 0.1*intensity.max()
    arr[lowi] = 0

    denom = arr.sum(2)
    denom[denom==0] = 1
    out[:,:,0] = red / denom
    out[:,:,1] = green / denom
    
    return out

# ------------------------------------------------------------------------------
def hsv_convert(arr):
    """ fast rgb_to_hsv using numpy array """
 
    # adapted from Arnar Flatberg
    # http://www.mail-archive.com/numpy-discussion@scipy.org/msg06147.html
    # it now handles NaN properly and mimics colorsys.rgb_to_hsv output

    #assert(arr.min()>=0 and arr.max()<=1)

    #arr = arr/255.
    arr = arr.astype("float32")
    out = np.empty_like(arr)

    arr_max = arr.max(-1)
    delta = arr.ptp(-1)
    s = delta / arr_max
    
    s[delta==0] = 0

    # red is max
    idx = (arr[:,:,0] == arr_max) 
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]

    # green is max
    idx = (arr[:,:,1] == arr_max) 
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0] ) / delta[idx]

    # blue is max
    idx = (arr[:,:,2] == arr_max) 
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1] ) / delta[idx]

    out[:,:,0] = (out[:,:,0]/6.0) % 1.0
    out[:,:,1] = s
    out[:,:,2] = arr_max

    # rescale back to [0, 255]
    #out *= 255.

    # remove NaN
    out[np.isnan(out)] = 0

    return out

def rgb_convert(arr):
    #assert(arr.min()>=0 and arr.max()<=1)

    # force 3 dims
    if arr.ndim == 2 or arr.shape[2] == 1:
        arr_new = np.empty(arr.shape[:2] + (3,), dtype="float32")
        arr_new[:,:,0] = arr.copy()
        arr_new[:,:,1] = arr.copy()
        arr_new[:,:,2] = arr.copy()
        arr = arr_new    
    
    return arr

#def rephists2(hin, division, nfeatures):
#    """ Compute local feature histograms from a given 3d (width X height X
#    n_channels) image.
#
#    These histograms are intended to serve as easy-to-compute additional
#    features that can be concatenated onto the V1-like output vector to
#    increase performance with little additional complexity. These additional
#    features are only used in the V1LIKE+ (i.e. + 'easy tricks') version of
#    the model.
#
#    Inputs:
#      hin -- 3d image (width X height X n_channels)
#      division -- granularity of the local histograms (e.g. 2 corresponds
#                  to computing feature histograms in each quadrant)
#      nfeatures -- desired number of resulting features
#
#    Outputs:
#      fvector -- feature vector
#
#    """
#
#    hin_h, hin_w, hin_d = hin.shape
#    nzones = hin_d * division**2
#    nbins = nfeatures / nzones
#    sx = (hin_w-1.)/division
#    sy = (hin_h-1.)/division
#    fvector = sp.zeros((nfeatures), 'f')
#    hists = []
#    for d in xrange(hin_d):
#        h = [sp.histogram(hin[j*sy:(j+1)*sy,i*sx:(i+1)*sx,d], bins=nbins)[0].ravel()
#             for i in xrange(division)
#             for j in xrange(division)
#             ]
#        hists += [h]
#
#    hists = sp.array(hists, 'f').ravel()
#    fvector[:hists.size] = hists
#    return fvector


def fastnorm(x):
    """ Fast Euclidean Norm (L2)

    This version should be faster than numpy.linalg.norm if 
    the dot function uses blas.

    Inputs:
      x -- numpy array

    Output:
      L2 norm from 1d representation of x
    
    """    
    xv = x.ravel()
    return np.dot(xv, xv)**(1/2.)



def gabor2d(gw, gh, gx0, gy0, wfreq, worient, wphase, shape):
    """ Generate a gabor 2d array
    
    Inputs:
      gw -- width of the gaussian envelope
      gh -- height of the gaussian envelope
      gx0 -- x indice of center of the gaussian envelope
      gy0 -- y indice of center of the gaussian envelope
      wfreq -- frequency of the 2d wave
      worient -- orientation of the 2d wave
      wphase -- phase of the 2d wave
      shape -- shape tuple (height, width)

    Outputs:
      gabor -- 2d gabor with zero-mean and unit-variance

    """
    
    height, width = shape
    y, x = np.mgrid[0:height, 0:width]
    
    X = x * np.cos(worient) * wfreq
    Y = y * np.sin(worient) * wfreq
    
    env = np.exp( -np.pi * ( ((x-gx0)**2./gw**2.) + ((y-gy0)**2./gh**2.) ) )
    wave = np.exp( 1j*(2*np.pi*(X+Y) + wphase) )
    gabor = np.real(env * wave)
    
    gabor -= gabor.mean()
    gabor /= fastnorm(gabor)
    
    return gabor


def sresample(src, outshape):
    """ Simple 3d array resampling

    Inputs:
      src -- a ndimensional array (dim>2)
      outshape -- fixed output shape for the first 2 dimensions

    Outputs:
       hout -- resulting n-dimensional array

    """

    inh, inw = src.shape[:2]
    outh, outw = outshape
    hslice = (np.arange(outh) * (inh-1.)/(outh-1.)).round().astype(int)
    wslice = (np.arange(outw) * (inw-1.)/(outw-1.)).round().astype(int)
    hout = src[hslice, :][:, wslice]
    return hout.copy()



def v1like_pool(hin, conv_mode, lsum_ksize, outshape=None, order=1):
    """ V1LIKE Pooling
    Boxcar Low-pass filter featuremap-wise

    Inputs:
      hin -- a 3-dimensional array (width X height X n_channels)
      lsum_ksize -- kernel size of the local sum ex: 17
      outshape -- fixed output shape (2d slices)
      order -- XXX

    Outputs:
       hout -- resulting 3-dimensional array

    """

    order = float(order)
    assert(order >= 1)

    # -- local sum
    if lsum_ksize is not None:
        hin_h, hin_w, hin_d = hin.shape
        dtype = hin.dtype
        if conv_mode == "valid":
            aux_shape = hin_h-lsum_ksize+1, hin_w-lsum_ksize+1, hin_d
            aux = np.empty(aux_shape, dtype)
        else:
            aux = np.empty(hin.shape, dtype)
        k1d = np.ones((lsum_ksize), 'f')
        #k2d = np.ones((lsum_ksize, lsum_ksize), 'f')
        krow = k1d[None,:]
        kcol = k1d[:,None]
        for d in xrange(aux.shape[2]):
            if order == 1:
                aux[:,:,d] = conv(conv(hin[:,:,d], krow, conv_mode,old_behavior=True), kcol, conv_mode,old_behavior=True)
            else:
                aux[:,:,d] = conv(conv(hin[:,:,d]**order, krow, conv_mode,old_behavior=True), kcol, conv_mode,old_behavior=True)**(1./order)

    else:
        aux = hin

    # -- resample output
    if outshape is None or outshape == aux.shape:
        hout = aux
    else:
        hout = sresample(aux, outshape)

    return hout



fft_cache = {}

def v1like_filter(hin, conv_mode, filterbank, use_cache=False):
    """ V1LIKE linear filtering
    Perform separable convolutions on an image with a set of filters

    Inputs:
      hin -- input image (a 2-dimensional array)
      filterbank -- TODO list of tuples with 1d filters (row, col)
                    used to perform separable convolution
      use_cache -- Boolean, use internal fft_cache (works _well_ if the input
      shapes don't vary much, otherwise you'll blow away the memory)

    Outputs:
      hout -- a 3-dimensional array with outputs of the filters
              (width X height X n_filters)

    """

    nfilters = len(filterbank)

    filt0 = filterbank[0]
    fft_shape = np.array(hin.shape) + np.array(filt0.shape) - 1
    hin_fft = scipy.signal.fftn(hin, fft_shape)

    if conv_mode == "valid":
        hout_shape = list( np.array(hin.shape[:2]) - np.array(filt0.shape[:2]) + 1 ) + [nfilters]
        hout_new = np.empty(hout_shape, 'f')
        begy = filt0.shape[0]
        endy = begy + hout_shape[0]
        begx = filt0.shape[1]
        endx = begx + hout_shape[1]
    elif conv_mode == "same":
        hout_shape = hin.shape[:2] + (nfilters,)
        hout_new = np.empty(hout_shape, 'f')
        begy = filt0.shape[0] / 2
        endy = begy + hout_shape[0]
        begx = filt0.shape[1] / 2
        endx = begx + hout_shape[1]
    else:
        raise NotImplementedError

    for i in xrange(nfilters):
        filt = filterbank[i]

        if use_cache:
            key = (filt.tostring(), tuple(fft_shape))
            if key in fft_cache:
                filt_fft = fft_cache[key]
            else:
                filt_fft = scipy.signal.fftn(filt, fft_shape)
                fft_cache[key] = filt_fft
        else:
            filt_fft = scipy.signal.fftn(filt, fft_shape)

        res_fft = scipy.signal.ifftn(hin_fft*filt_fft)
        res_fft = res_fft[begy:endy, begx:endx]
        hout_new[:,:,i] = np.real(res_fft)

    hout = hout_new

    return hout

# TODO: make this not a global variable
filt_l = None

def get_gabor_filters(params):
    """ Return a Gabor filterbank (generate it if needed)

    Inputs:
    params -- filters parameters (dict)

    Outputs:
    filt_l -- filterbank (list)

    """

    global filt_l

    if filt_l is not None:
        return filt_l

    # -- get parameters
    fh, fw = params['kshape']
    orients = params['orients']
    freqs = params['freqs']
    phases = params['phases']
    #nf =  len(orients) * len(freqs) * len(phases)
    #fbshape = nf, fh, fw
    xc = fw/2
    yc = fh/2
    filt_l = []

    # -- build the filterbank
    for freq in freqs:
        for orient in orients:
            for phase in phases:
                # create 2d gabor
                filt = gabor2d(xc,yc,xc,yc,
                               freq,orient,phase,
                               (fw,fh))
                filt_l += [filt]

    return filt_l



def v1like_norm(hin, conv_mode, kshape, threshold):
    """ V1LIKE local normalization

    Each pixel in the input image is divisively normalized by the L2 norm
    of the pixels in a local neighborhood around it, and the result of this
    division is placed in the output image.

    Inputs:
      hin -- a 3-dimensional array (width X height X rgb)
      kshape -- kernel shape (tuple) ex: (3,3) for a 3x3 normalization
                neighborhood
      threshold -- magnitude threshold, if the vector's length is below
                   it doesn't get resized ex: 1.

    Outputs:
      hout -- a normalized 3-dimensional array (width X height X rgb)

    """
    eps = 1e-5
    kh, kw = kshape
    dtype = hin.dtype
    hsrc = hin[:].copy()

    # -- prepare hout
    hin_h, hin_w, hin_d = hin.shape
    hout_h = hin_h# - kh + 1
    hout_w = hin_w# - kw + 1

    if conv_mode != "same":
        hout_h = hout_h - kh + 1
        hout_w = hout_w - kw + 1

    hout_d = hin_d
    hout = np.empty((hout_h, hout_w, hout_d), 'float32')

    # -- compute numerator (hnum) and divisor (hdiv)
    # sum kernel
    hin_d = hin.shape[-1]
    kshape3d = list(kshape) + [hin_d]
    ker = np.ones(kshape3d, dtype=dtype)
    size = ker.size

    # compute sum-of-square
    hsq = hsrc ** 2.
    #hssq = conv(hsq, ker, conv_mode).astype(dtype)
    kerH = ker[:,0,0][:, None]#, None]
    kerW = ker[0,:,0][None, :]#, None]
    kerD = ker[0,0,:][None, None, :]

    hssq = conv(
                conv(
                     conv(hsq, kerD, 'valid')[:,:,0].astype(dtype),
                     kerW,
                     conv_mode,old_behavior=True),
                kerH,
                conv_mode,old_behavior=True).astype(dtype)
    hssq = hssq[:,:,None]

    # compute hnum and hdiv
    ys = kh / 2
    xs = kw / 2
    hout_h, hout_w, hout_d = hout.shape[-3:]
    hs = hout_h
    ws = hout_w

    hsum = conv(
                conv(
                     conv(hsrc,
                          kerD, 'valid')[:,:,0].astype(dtype),
                     kerW,
                     conv_mode,old_behavior=True),
                kerH,
                conv_mode,old_behavior=True).astype(dtype)

    hsum = hsum[:,:,None]
    if conv_mode == 'same':
        hnum = hsrc - (hsum/size)
    else:
        hnum = hsrc[ys:ys+hs, xs:xs+ws] - (hsum/size)
    val = (hssq - (hsum**2.)/size)
    val[val<0] = 0
    hdiv = val ** (1./2) + eps

    # -- apply normalization
    # 'volume' threshold
    np.putmask(hdiv, hdiv < (threshold+eps), 1.)
    result = (hnum / hdiv)

    #print result.shape
    hout[:] = result
    #print hout.shape, hout.dtype
    return hout




def v1like_fromarray(arr, params):
    """ Applies a simple V1-like model and generates a feature vector from
    its outputs.

    Inputs:
      arr -- image's array
      params -- representation parameters (dict)
      featsel -- features to include to the vector (dict)

    Outputs:
      fvector -- corresponding feature vector

    """

    if 'conv_mode' not in params:
        params['conv_mode'] = 'same'
    if 'color_space' not in params:
        params['color_space'] = 'gray'

    arr = np.atleast_3d(arr)

    smallest_edge = min(arr.shape[:2])

    rep = params

    preproc_lsum = rep['preproc']['lsum_ksize']
    if preproc_lsum is None:
        preproc_lsum = 1
    smallest_edge -= (preproc_lsum-1)

    normin_kshape = rep['normin']['kshape']
    smallest_edge -= (normin_kshape[0]-1)

    filter_kshape = rep['filter']['kshape']
    smallest_edge -= (filter_kshape[0]-1)

    normout_kshape = rep['normout']['kshape']
    smallest_edge -= (normout_kshape[0]-1)

    pool_lsum = rep['pool']['lsum_ksize']
    smallest_edge -= (pool_lsum-1)

    arrh, arrw, _ = arr.shape

    if smallest_edge <= 0 and rep['conv_mode'] == 'valid':
        if arrh > arrw:
            new_w = arrw - smallest_edge + 1
            new_h =  int(np.round(1.*new_w  * arrh/arrw))
            print new_w, new_h
            raise
        elif arrh < arrw:
            new_h = arrh - smallest_edge + 1
            new_w =  int(np.round(1.*new_h  * arrw/arrh))
            print new_w, new_h
            raise
        else:
            pass

    # TODO: finish image size adjustment
    assert min(arr.shape[:2]) > 0

    # use the first 3 channels only
    orig_imga = arr.astype("float32")[:,:,:3]
    
    # make sure that we don't have a 3-channel (pseudo) gray image
    if orig_imga.shape[2] == 3 \
            and (orig_imga[:,:,0]-orig_imga.mean(2) < 0.1*orig_imga.max()).all() \
            and (orig_imga[:,:,1]-orig_imga.mean(2) < 0.1*orig_imga.max()).all() \
            and (orig_imga[:,:,2]-orig_imga.mean(2) < 0.1*orig_imga.max()).all():
        orig_imga = np.atleast_3d(orig_imga[:,:,0])

    # rescale to [0,1]
    #print orig_imga.min(), orig_imga.max()
    if orig_imga.min() == orig_imga.max():
        raise MinMaxError("[ERROR] orig_imga.min() == orig_imga.max() "
                          "orig_imga.min() = %f, orig_imga.max() = %f"
                          % (orig_imga.min(), orig_imga.max())
                          )

    orig_imga -= orig_imga.min()
    orig_imga /= orig_imga.max()

    # -- color conversion
    # insure 3 dims
    #print orig_imga.shape
    if orig_imga.ndim == 2 or orig_imga.shape[2] == 1:
        orig_imga_new = np.empty(orig_imga.shape[:2] + (3,), dtype="float32")
        orig_imga.shape = orig_imga_new[:,:,0].shape
        orig_imga_new[:,:,0] = 0.2989*orig_imga
        orig_imga_new[:,:,1] = 0.5870*orig_imga
        orig_imga_new[:,:,2] = 0.1141*orig_imga
        orig_imga = orig_imga_new

    # -
    if params['color_space'] == 'rgb':
        orig_imga_conv = orig_imga
#     elif params['color_space'] == 'rg':
#         orig_imga_conv = colorconv.rg_convert(orig_imga)
    elif params['color_space'] == 'rg2':
        orig_imga_conv = rg2_convert(orig_imga)
    elif params['color_space'] == 'gray':
        orig_imga_conv = gray_convert(orig_imga)
        orig_imga_conv.shape = orig_imga_conv.shape + (1,)
    elif params['color_space'] == 'opp':
        orig_imga_conv = opp_convert(orig_imga)
    elif params['color_space'] == 'oppnorm':
        orig_imga_conv = oppnorm_convert(orig_imga)
    elif params['color_space'] == 'chrom':
        orig_imga_conv = chrom_convert(orig_imga)
#     elif params['color_space'] == 'opponent':
#         orig_imga_conv = colorconv.opponent_convert(orig_imga)
#     elif params['color_space'] == 'W':
#         orig_imga_conv = colorconv.W_convert(orig_imga)
    elif params['color_space'] == 'hsv':
        orig_imga_conv = hsv_convert(orig_imga)
    else:
        raise ValueError, "params['color_space'] not understood"

    # -- process each map

    for cidx in xrange(orig_imga_conv.shape[2]):
        imga0 = orig_imga_conv[:,:,cidx]

        assert(imga0.min() != imga0.max())

        # -- 0. preprocessing
        #imga0 = imga0 / 255.0

        # flip image ?
        if 'flip_lr' in params['preproc'] and params['preproc']['flip_lr']:
            imga0 = imga0[:,::-1]

        if 'flip_ud' in params['preproc'] and params['preproc']['flip_ud']:
            imga0 = imga0[::-1,:]

        # smoothing
        lsum_ksize = params['preproc']['lsum_ksize']
        conv_mode = params['conv_mode']
        if lsum_ksize is not None:
            k = np.ones((lsum_ksize), 'f') / lsum_ksize
            imga0 = conv(conv(imga0, k[np.newaxis,:], conv_mode,old_behavior=True),
                          k[:,np.newaxis], conv_mode,old_behavior=True)

        # whiten full image (assume True)
        if 'whiten' not in params['preproc'] or params['preproc']['whiten']:
            imga0 -= imga0.mean()
            if imga0.std() != 0:
                imga0 /= imga0.std()

        # -- 1. input normalization
        imga1 = v1like_norm(imga0[:,:,np.newaxis], conv_mode, **params['normin'])
        
        # -- 2. linear filtering
        filt_l = get_gabor_filters(params['filter'])
        imga2 = v1like_filter(imga1[:,:,0], conv_mode, filt_l)

        # -- 3. simple non-linear activation (clamping)
        minout = params['activ']['minout'] # sustain activity
        maxout = params['activ']['maxout'] # saturation
        imga3 = imga2.clip(minout, maxout)

        # -- 4. output normalization
        imga4 = v1like_norm(imga3, conv_mode, **params['normout'])

        # -- 5. sparsify ?
        if "sparsify" in params and params["sparsify"]:
            imga4 = (imga4.max(2)[:,:,None] == imga4)
            #raise

        # -- 6. volume dimension reduction
        imga5 = v1like_pool(imga4, conv_mode, **params['pool'])
        output = imga5

    return output

