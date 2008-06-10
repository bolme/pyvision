from math import *
from random import *


def synth_sin():
    f = open("synth1_sin.txt",'w')
    for i in range(1000):
        x = uniform(-2*pi,2*pi)
        truth = sin(x)
        measured = normalvariate(truth,0.2)
        f.write("%f %f %f\n"%(x,measured,truth))
    
    
def synth_cos():
    f = open("synth1_cos.txt",'w')
    for i in range(1000):
        x = uniform(-2*pi,2*pi)
        truth = cos(x)
        measured = normalvariate(truth,0.2)
        f.write("%f %f %f\n"%(x,measured,truth))
    
    
def synth_mix():
    f = open("synth1_mix.txt",'w')
    for i in range(1000):
        x = uniform(-2*pi,2*pi)
        truth = cos(2*x)+cos(0.7*x+1)+4
        measured = normalvariate(truth,0.2)
        f.write("%f %f %f\n"%(x,measured,truth))
    
    
def synth_quad():
    f = open("synth1_quad.txt",'w')
    for i in range(1000):
        x = uniform(-2*pi,2*pi)
        truth = x*x
        measured = normalvariate(truth,0.2)
        f.write("%f %f %f\n"%(x,measured,truth))
    
    
def synth_lin():
    f = open("synth1_lin.txt",'w')
    for i in range(1000):
        x = uniform(-2*pi,2*pi)
        truth = 0.5*x
        measured = normalvariate(truth,0.2)
        f.write("%f %f %f\n"%(x,measured,truth))
    
    
def synth_cube():
    f = open("synth1_cube.txt",'w')
    for i in range(1000):
        x = uniform(-2*pi,2*pi)
        truth = x*x*x
        measured = normalvariate(truth,0.2)
        f.write("%f %f %f\n"%(x,measured,truth))
    
    

if __name__ == "__main__":
    #synth_sin()
    #synth_cos()
    #synth_mix()
    #synth_quad()
    #synth_cube()
    #synth_lin()
    pass
    
    
    
    