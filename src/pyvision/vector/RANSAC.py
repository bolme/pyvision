#from numpy import np.dot, arange, abs
#from numpy.linalg import lstsq
import numpy as np
import random


def computeErrorAndCount(A,b,x,group,tol):
    error = np.abs(b - np.dot(A,x))
    tot_error = 0.0
    count = 0
    inliers = np.zeros(b.shape,dtype=np.bool)
    for j in range(0,len(error),group):
        e = 0 
        for l in range(group):
            e += error[j+l]**2
        e = np.sqrt(e)
        if e < tol:
            count += 1
            tot_error += e
            for l in range(group):
                inliers[j+l] = True
                
    if count == 0:
        return tot_error,count,inliers
    
    return tot_error/count,count,inliers

##
# Uses the RANSAC algorithm to solve Ax=b
#
# M.A. Fischler and R.C. Bolles.  Random sample consensus: A paradigm for 
# model fitting with applications to image analysis and automated cartography.  
# Communication of Association for Computing Machinery, 24(6): 381--395, 1981.
def RANSAC(A,b,count=None,tol=1.0,niter=None,group=1,verbose=False,full_output=False):
    #n = len(y.flatten())
    n,k = A.shape
    group = int(group)
    assert group > 0
    assert n % group == 0
    assert n >= k
    tmp = np.arange(n/group)
    
    if niter == None:
        niter = n/group
    
    bestx = np.linalg.lstsq(A,b)[0]
    besterror,bestcount,bestinliers = computeErrorAndCount(A,b,bestx,group,tol)
    if verbose: print "New Best (LS):",bestcount,besterror,float(bestcount*group)/n
    
    if bestcount == n/group:
        if full_output:
            return bestx,bestcount,besterror,bestinliers
        
        return bestx
        
    #bestcount = 0
    #besterror = 0.0
    for _ in xrange(niter):
        sample = random.sample(tmp,k/group)
        
        new_sample = []
        for j in sample:
            for l in range(group):
                new_sample.append(group*j+l)

        sample = new_sample

        ty = b[sample,:]
        tX = A[sample,:]
        
        #print tX.shape,ty.shape

        try:
            x = np.linalg.lstsq(tX,ty)[0]
        except:
            continue
        
        error,count,inliers = computeErrorAndCount(A,b,x,group,tol)
        
        if bestcount < count or (bestcount == count and error < besterror):
            bestcount = count
            besterror = error 
            bestx = x
            bestinliers = inliers
            if verbose: print "    New Best:",bestcount,besterror,float(bestcount*group)/n
            
        #print x, count, bestcount
                
    x = bestx    
    
    #refine the estimate
    #error,count,inliers = computeErrorAndCount(A,b,bestx,group,tol)
    inliers = bestinliers
    for _ in xrange(10):
        ty = b[inliers.flatten(),:]
        tX = A[inliers.flatten(),:]
        try:
            x = np.linalg.lstsq(tX,ty)[0]        
        except:
            continue
        error,count,inliers = computeErrorAndCount(A,b,x,group,tol)
        #if verbose: print "    ",error,count,x
        if bestcount < count or (bestcount == count and error < besterror):
            bestcount = count
            besterror = error 
            bestx = x
            bestinliers = inliers
            if verbose: print "Improved Best:",bestcount,besterror,float(bestcount*group)/n
        
        #new_inliers = nonzero(abs(b - np.dot(A,x)) < tol)[0]
        #if list(new_inliers) == list(inliers):
        #    break
        #inliers = new_inliers

    if full_output:
        return bestx,bestcount,besterror,bestinliers
    
    return bestx


def _quantile(errors,quantile):
    
    errors = errors.copy()
    i = int(quantile*errors.shape[0])
    errors.sort()
    return errors[i]
    
   
##
# Uses the LMeDs algorithm to solve Ax=b
#
# M.A. Fischler and R.C. Bolles.  Random sample consensus: A paradigm for 
# model fitting with applications to image analysis and automated cartography.  
# Communication of Association for Computing Machinery, 24(6): 381--395, 1981.
def LMeDs(A,b,quantile=0.75,N = None,verbose=True):
    #n = len(y.flatten())
    n,k = A.shape
    tmp = np.arange(n)

    best_sample = tmp
    x = bestx = np.linalg.lstsq(A,b)[0]
    best_error = _quantile(np.abs(b - np.dot(A,x)),quantile)
    #print "LMeDs Error:",best_error
    if N == None:
        N = n
        
    for i in range(N):
        sample = random.sample(tmp,k)

        ty = b[sample,:]
        tX = A[sample,:]

        try:
            x = np.linalg.lstsq(tX,ty)[0]
        except:
            continue
        
        med_error = _quantile(np.abs(b - np.dot(A,x)),quantile)
        
        if med_error < best_error:
            #print "      Error:",best_error
            best_sample = sample
            best_error = med_error
            bestx = x
            
        #print x, count, bestcount
                
    x = bestx    
    
    #refine the estimate using local search
    #print "    Local Search"
    sample = np.zeros([n],dtype=np.bool)
    sample[best_sample] = True
    best_sample = sample
    random.shuffle(tmp)
    
    keep_going = True
    while keep_going:
        #print "     Iter"
        keep_going = False
        for i in tmp:
            sample = best_sample.copy()
            sample[i] = not sample[i]
            
            ty = b[sample,:]
            tX = A[sample,:]
            
            try:
                x = np.linalg.lstsq(tX,ty)[0]
            except:
                continue
            
            med_error = _quantile(np.abs(b - np.dot(A,x)),quantile)
            
            if med_error < best_error or (med_error == best_error and best_sample.sum() < sample.sum()):
                #print "      Error:",best_error
                keep_going = True
                best_sample = sample
                best_error = med_error
                bestx = x
    
    #inliers = nonzero(np.abs(b - np.dot(A,x)) < tol)[0]
    #for i in range(10):
    #    ty = b[inliers,:]
    #    tX = A[inliers,:]
    #    x = np.linalg.lstsq(tX,ty)[0]
    #    new_inliers = nonzero(np.abs(b - np.dot(A,x)) < tol)[0]
    #    if list(new_inliers) == list(inliers):
    #        break
    #    inliers = new_inliers

    return x

   
if __name__ == '__main__':
    A = []
    b = []
    #print dir(random)
    
    for x in range(40):
        b.append( 10*x + 5 + random.normalvariate(0.0,2.0))
        A.append([x,1])
        
    A = np.array(A)
    b = np.array(b)
    b[0] = -20
    
    print np.linalg.lstsq(A,b)[0]
    print RANSAC(A,b,tol=6.0)
    

    A = []
    b = []
    #print dir(random)
    
    for y in range(-10,10):
        for x in range(-10,10):
            b.append( 15*y + 10*x + 5 + random.normalvariate(0.0,2.0))
            A.append([x,y,1])
        
    A = np.array(A)
    b = np.array(b)
    b[0] = -200000.
    
    print np.linalg.lstsq(A,b)[0]
    print RANSAC(A,b,group=2,tol=6,full_output = True,verbose=True)
    


