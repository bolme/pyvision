from numpy import *

def hammingWindow(size):
    '''
    Windowing function from:
    http://en.wikipedia.org/wiki/Window_function
    '''
    w,h = size
    X = arange(w).reshape(w,1)
    Y = arange(h).reshape(1,h)
    X = X*ones((1,h),'d')
    Y = Y*ones((w,1),'d')
    
    window = (5.3836-0.46164*cos(2*pi*X/(w-1.0)))*(5.3836-0.46164*cos(2*pi*Y/(h-1.0)))
    return window


def hannWindow(size):
    '''
    Windowing function from:
    http://en.wikipedia.org/wiki/Window_function
    '''
    w,h = size
    X = arange(w).reshape(w,1)
    Y = arange(h).reshape(1,h)
    X = X*ones((1,h),'d')
    Y = Y*ones((w,1),'d')
    
    window = (0.5*(1-cos(2*pi*X/(w-1.0))))*(0.5*(1-cos(2*pi*Y/(h-1.0))))
    return window

def cosineWindow(size):
    '''
    Windowing function from:
    http://en.wikipedia.org/wiki/Window_function
    '''
    w,h = size
    X = arange(w).reshape(w,1)
    Y = arange(h).reshape(1,h)
    X = X*ones((1,h),'d')
    Y = Y*ones((w,1),'d')
    
    window = (sin(pi*X/(w-1.0)))*(sin(pi*Y/(h-1.0)))
    return window
