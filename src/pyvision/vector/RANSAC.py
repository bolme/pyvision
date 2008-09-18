from numpy import nonzero, dot, arange, abs
from numpy.linalg import lstsq
import random

##
# Uses the RANSAC algorithm to solve Ax=b
#
# M.A. Fischler and R.C. Bolles.  Random sample consensus: A paradigm for 
# model fitting with applications to image analysis and automated cartography.  
# Communication of Association for Computing Machinery, 24(6): 381--395, 1981.
def RANSAC(A,b,count=None,tol=5.0,verbose=True):
    #n = len(y.flatten())
    n,k = A.shape
    if count == None:
        count = n
    tmp = arange(n)
    bestx = None
    bestcount = 0
    for i in range(n):
        sample = random.sample(tmp,3)

        ty = b[sample,:]
        tX = A[sample,:]

        x = lstsq(tX,ty)[0]

        count = (abs(b - dot(x,A.transpose())) < tol).sum()
        
        if bestcount < count:
            bestcount = count
            bestx = x
            
        #print x, count, bestcount
                
    x = bestx    
    
    #refine the estimate
    inliers = nonzero(abs(b - dot(x,A.transpose())) < tol)[0]
    for i in range(10):
        ty = b[inliers,:]
        tX = A[inliers,:]
        x = lstsq(tX,ty)[0]
        new_inliers = nonzero(abs(b - dot(x,A.transpose())) < tol)[0]
        if list(new_inliers) == list(inliers):
            break
        inliers = new_inliers

    return x

   
