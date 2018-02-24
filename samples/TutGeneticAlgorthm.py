# This is a sample script for using the genetic algorithm.

import pyvision as pv
import pyvision.optimize.genetic as ga
import numpy as np
import pickle as pkl

def callback(population):
    plot = pv.Plot(x_range=[0,10],y_range=[0,10],title='Search Space')
    
    plot.point(pv.Point(3,4),size=20,shape=16,color='gray')
    plot.point(pv.Point(8,2),size=10,shape=16,color='gray')
    plot.point(pv.Point(5,9),size=40,shape=16,color='gray')
    
    pts = [ [each[1][0].value,each[1][1].value] for each in population ]
    
    #pts = [ pv.Point(each[1][0],each[1][1]) for each in population ]
    #print pts
    plot.points(pts,color='red')
    plot.show(delay=10,window='Search Space')
    #for each in population:
    #    plot.point(each[1])


def myFitness(*args,**kwargs):
    '''
    This function has three local minimum.
    
    Min values is approx -0.63 @ [8,2]
    '''
    val = 0.0
    x,y = args
    cx,cy,sigma=3.0,4.0,1.0
    dist2 = (x-cx)**2 + (y-cy)**2
    scale = 1.0/(2*np.pi*sigma**2)
    val += scale*np.exp(-dist2/(2*sigma**2))
    
    cx,cy,sigma=8.0,2.0,0.5
    dist2 = (x-cx)**2 + (y-cy)**2
    scale = 1.0/(2*np.pi*sigma**2)
    val += scale*np.exp(-dist2/(2*sigma**2))
    
    cx,cy,sigma=5.0,9.0,2.0
    dist2 = (x-cx)**2 + (y-cy)**2
    scale = 1.0/(2*np.pi*sigma**2)
    val += scale*np.exp(-dist2/(2*sigma**2))
    
    return -val

# use True and False to toggle between running GA and loading archived results
if True:
    # Check the fitness function
    print(myFitness(8,2))
    assert abs(myFitness(8,2) + 0.63) < 0.01
    
    # Here are the three arguments
    args = [ga.GAFloat(0,10),
            ga.GAFloat(0,10)]
    
    # no keyword arguments this time
    kwargs = {}
    
    ilog = pv.ImageLog()
    ga_test = ga.GeneticAlgorithm(myFitness,args,kwargs,n_processes='AUTO')
    
    print('running ga')
    
    result = ga_test.optimize(max_iter=2000,callback=callback,display=True)
    print("Best Score =",result[0])
    print("Best args =",result[1])
    print("Best kwargs =",result[2])
    

else:
    # load the archived result
    
    # look in the "/tmp" directory for a log of the run in a *_pyvis directory
    # pkl files document the result
    # this directory can also be used to restart a GA run if there was a problem
    
    result = pkl.load(open('/tmp/20150501_140701_pyvis_log/003701_Fitness_22.93110867.pkl','rb'))
    
    print(result)
    
    # to get useable args and kwargs you need to do this
    ga.list_generate(result)
    score = result[0]
    args = result[1]
    kwargs = result[2]
    
    print(score)
    print(args)
    print(kwargs)
    print(myFitness(*args,**kwargs))

    
