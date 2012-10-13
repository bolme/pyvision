'''
Created on Oct 7, 2011

@author: bolme
'''
import pyvision as pv
import scipy as sp

if __name__ == '__main__':
    im = pv.Image("baboon.jpg")
    mat = im.asMatrix2D()
    
    U,D,Vt = sp.linalg.svd(mat)
    D = sp.diag(D)
    
    for dim in [256,128,64,32,16,8,4]:
        U = U[:,:dim]
        D = D[:dim,:dim]
        Vt = Vt[:dim,:]
        mat = sp.dot(sp.dot(U,D),Vt)
        pv.Image(mat).show(delay=0)