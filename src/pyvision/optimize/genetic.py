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
This module contains a basic framework for a genetic algorithm. This specific
implentation specifically keeps high selective pressure on the population by
always keeping the best individuals found.  The algorithm checks to make sure
that no individual is tested twice.

Chromozomes are constructed from "Variable" classes that govern rondom 
generation and recombination of each element in the chromozome. For 
example if your fitness function requires a number from one to ten you
could uses a ChoiceVariable with choices = [1,2,3,4,5,6,7,8,9,10].

TODO: Add a tutorial.

This class also has some examples in the unit tests.
'''

import random
#import unittest
import math
import multiprocessing as mp
#import sys
import traceback
import numpy as np
import copy
#import time
import pyvision as pv
import os
import cPickle as pkl

# Genetic algorithm message types
_GA_EVALUATE="GA_EVALUATE"
_GA_STOP_WORKER="GA_STOP_WORKER"
_GA_SCORE = "GA_SCORE"

def _clipRange(val,minval,maxval):
    minval,maxval = min(minval,maxval),max(minval,maxval)
    val = min(val,maxval)
    val = max(val,minval)
    return val


def _circularRange(val,minval,maxval):
    minval,maxval = min(minval,maxval),max(minval,maxval)
    dist = maxval - minval
    if val > maxval:
        t1 = (val - maxval)/dist
        t2 = t1 - math.floor(t1)
        t3 = minval + dist*t2
        return t3
    elif val < minval:
        t1 = (val - minval)/dist
        t2 = t1 - math.ceil(t1)
        t3 = maxval + dist*t2
        return t3

    return val


class GAVariable:
    ''' 
    This is a superclass for a variable that is optimized by the GA. It 
    has three methods that need to be overridden by subclasses: combine,
    mutate, and generate. 
    
    
    '''
    def __init__(self,mutation_rate=0.025):
        self.mutation_rate = mutation_rate
    
    def random(self):
        ''' Initialize this variable randomly '''
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def combine(self,other):
        '''combine this variable with other.'''
        raise NotImplementedError("This method should be overridden by subclasses")

    def mutate(self):
        '''introduce mutations into the variable.'''
        raise NotImplementedError("This method should be overridden by subclasses")

    def generate(self):
        '''generate the actual value that will be populated in the arguments'''
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def flatValue(self):
        return self.generate()
        
    def __repr__(self):
        return str(self.value)

    
    
class GAFloat(GAVariable):
    
    def __init__(self,minval,maxval,**kwargs):
        GAVariable.__init__(self, **kwargs)
        self.minval,self.maxval = min(minval,maxval),max(minval,maxval)
        self.random()
        
    def clipRange(self):
        self.value = _clipRange(self.value, self.minval, self.maxval)

    def random(self):
        ''' Initialize this variable randomly '''
        self.value = self.minval + (self.maxval - self.minval)*random.random()
        self.clipRange()
    
    def combine(self,other):
        '''combine this variable with other.'''
        dist = np.abs(self.value - other.value)+0.000001
        if random.randint(0,1) == 0:
            self.value = other.value
        self.value += np.random.normal(0,dist/3.0)
        self.clipRange()

    def mutate(self):
        '''introduce mutations into the variable.'''
        if random.random() < self.mutation_rate:
            dist = np.abs(self.maxval-self.minval)
            self.value += np.random.normal(0,dist/50.0)
        self.clipRange()

    def generate(self):
        '''generate the actual value that will be populated in the arguments'''
        return self.value
    
    def __repr__(self):
        return str(self.value)
        

class GALogFloat(GAVariable):
    '''
    A float on a log scale.
    '''
    
    def __init__(self,minval,maxval,**kwargs):
        GAVariable.__init__(self, **kwargs)
        minval = np.log(minval)
        maxval = np.log(maxval)
        self.minval,self.maxval = min(minval,maxval),max(minval,maxval)
        self.random()
        
    def clipRange(self):
        self.value = _clipRange(self.value, self.minval, self.maxval)

    def random(self):
        ''' Initialize this variable randomly '''
        self.value = self.minval + (self.maxval - self.minval)*random.random()
        self.clipRange()
    
    def combine(self,other):
        '''combine this variable with other.'''
        dist = np.abs(self.value - other.value)+0.000001
        if random.randint(0,1) == 0:
            self.value = other.value
        self.value += np.random.normal(0,dist/3.0)
        self.clipRange()

    def mutate(self):
        '''introduce mutations into the variable.'''
        if random.random() < self.mutation_rate:
            dist = np.abs(self.maxval-self.minval)
            self.value += np.random.normal(0,dist/50.0)
        self.clipRange()

    def generate(self):
        '''generate the actual value that will be populated in the arguments'''
        return np.exp(self.value)
    
    def __repr__(self):
        return str(np.exp(self.value))
    
    
class GAAngle(GAVariable):
    '''
    Maps to an angle in the range -pi to pi.
    '''
    
    def __init__(self,minval=-np.pi,maxval=np.pi,**kwargs):
        GAVariable.__init__(self, **kwargs)
        self.minval,self.maxval = minval,maxval
        self.random()
        
    def clipRange(self):
        self.value = _circularRange(self.value, self.minval, self.maxval)

    def random(self):
        ''' Initialize this variable randomly '''
        self.value = self.minval + (self.maxval - self.minval)*random.random()
        self.clipRange()
    
    def combine(self,other):
        '''combine this variable with other.'''
        t1 = 0.5*(self.minval + self.maxval) # find the center of the range
        t2 = t1 - self.value # adjustment to center self.value
        t3 = _circularRange( other.value + t2, self.minval, self.maxval) # adjust other.value
        dist = np.abs(t3 - t1) # compute dist

        # select one value
        if random.randint(0,1) == 0:
            self.value = other.value
        
        # adjust it
        self.value += np.random.normal(0,dist/3.0)
        
        # clip
        self.clipRange()

    def mutate(self):
        '''introduce mutations into the variable.'''
        if random.random() < self.mutation_rate:
            dist = np.abs(self.maxval-self.minval)
            self.value += np.random.normal(0,dist/50.0)
        self.clipRange()

    def generate(self):
        '''generate the actual value that will be populated in the arguments'''
        return self.value
    
    def __repr__(self):
        return str(self.value)
        
class GAUnitRect(GAVariable):
    
    def __init__(self,min_width=0.2,max_width=1.0,min_height=0.2,max_height=1.0,**kwargs):
        GAVariable.__init__(self,**kwargs)
        
        assert min_width >= 0
        assert min_width <= 1
        assert max_width >= 0
        assert max_width <= 1
        assert min_height >= 0
        assert min_height <= 1
        assert max_height >= 0
        assert max_height <= 1
        assert min_width <= max_width
        assert min_height <= max_height
        
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        
        self.random()
        
    def clipRange(self):
        self.width  = _clipRange(self.width, self.min_width, self.max_width)
        self.height = _clipRange(self.height, self.min_height, self.max_height)
        self.cx     = _clipRange(self.cx, 0.5*self.width, 1.0 - 0.5*self.width)
        self.cy     = _clipRange(self.cy, 0.5*self.height, 1.0 - 0.5*self.height)

        
    def random(self):
        ''' Initialize this variable randomly '''
        self.cx = random.random()
        self.cy = random.random()
        
        diff = self.max_width - self.min_width
        self.width = self.min_width + random.random() * diff
        
        diff = self.max_height - self.min_height
        self.height = self.min_height + random.random() * diff
        self.clipRange()

    
    def combine(self,other):
        '''combine this variable with other.'''

        # select one value
        cx_dist = np.abs(self.cx - other.cx) + 1e-7
        cy_dist = np.abs(self.cy - other.cy) + 1e-7
        w_dist = np.abs(self.width - other.width) + 1e-7
        h_dist = np.abs(self.height - other.height) + 1e-7
        
        if random.randint(0,1) == 0:
            self.cx = other.cx
            self.cy = other.cy
            self.width = other.width
            self.height = other.height
            
        if cx_dist <= 0 or cy_dist <= 0 or w_dist <= 0 or h_dist <= 0  :
            print "Combining:",self,other,cx_dist,cy_dist,w_dist,h_dist
        self.cx += np.random.normal(0,cx_dist/3.0)
        self.cy += np.random.normal(0,cy_dist/3.0)
        self.width += np.random.normal(0,w_dist/3.0)
        self.height += np.random.normal(0,h_dist/3.0)

        # clip
        self.clipRange()

    def mutate(self):
        '''introduce mutations into the variable.'''
        if random.random() < self.mutation_rate:
            dist = np.abs(1.0)
            self.cx += np.random.normal(0,dist/50.0)

            dist = np.abs(1.0)
            self.cy += np.random.normal(0,dist/50.0)

            dist = np.abs(1.0)
            self.cy += np.random.normal(0,dist/50.0)

            dist = np.abs(1.0)
            self.cy += np.random.normal(0,dist/50.0)

            self.clipRange()

    def generate(self):
        '''generate the actual value that will be populated in the arguments'''
        return pv.CenteredRect(self.cx, self.cy, self.width, self.height)
    
    def flatValue(self):
        return self.value.asCenteredTuple()
    
    def __repr__(self):
        return str(self.generate())

        
        
class GAUnitRect2(GAVariable):
    
    def __init__(self,min_width=0.2,max_width=1.0,min_height=0.2,max_height=1.0,**kwargs):
        GAVariable.__init__(self,**kwargs)
        
        assert min_width >= 0
        assert min_width <= 1
        assert max_width >= 0
        assert max_width <= 1
        assert min_height >= 0
        assert min_height <= 1
        assert max_height >= 0
        assert max_height <= 1
        assert min_width <= max_width
        assert min_height <= max_height
        
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        
        self.random()
        
    def clipRange(self):
        self.left   = _clipRange(0, self.left, 1)
        self.right  = _clipRange(0, self.right, 1)
        self.top    = _clipRange(0, self.top, 1)
        self.bottom = _clipRange(0, self.bottom,1)
        if self.left > self.right:
            self.left,self.right = self.right,self.left
        if self.bottom < self.top:
            self.bottom,self.top = self.top,self.bottom

        
    def random(self):
        ''' Initialize this variable randomly '''
        self.left = random.random()
        self.right = random.random()
        self.top = random.random()
        self.bottom = random.random()
        self.clipRange()

    
    def combine(self,other):
        '''combine this variable with other.'''

        # select one value
        l_dist = np.abs(self.left - other.left) + 1e-7
        r_dist = np.abs(self.right - other.right) + 1e-7
        t_dist = np.abs(self.top - other.top) + 1e-7
        b_dist = np.abs(self.bottom - other.bottom) + 1e-7
        
        if random.randint(0,1) == 0:
            self.left   = other.left
            self.right  = other.right
            self.top    = other.top
            self.bottom = other.bottom
            
        if l_dist <= 0 or r_dist <= 0 or t_dist <= 0 or b_dist <= 0  :
            print "Combining:",self,other,l_dist,r_dist,t_dist,b_dist
        self.left   += np.random.normal(0,l_dist/3.0)
        self.right  += np.random.normal(0,r_dist/3.0)
        self.top    += np.random.normal(0,t_dist/3.0)
        self.bottom += np.random.normal(0,b_dist/3.0)

        # clip
        self.clipRange()

    def mutate(self):
        '''introduce mutations into the variable.'''
        if random.random() < self.mutation_rate:
            dist = np.abs(1.0)
            self.left += np.random.normal(0,dist/50.0)

            dist = np.abs(1.0)
            self.right += np.random.normal(0,dist/50.0)

            dist = np.abs(1.0)
            self.top += np.random.normal(0,dist/50.0)

            dist = np.abs(1.0)
            self.bottom += np.random.normal(0,dist/50.0)

            self.clipRange()

    def generate(self):
        '''generate the actual value that will be populated in the arguments'''
        return pv.BoundingRect([pv.Point(self.left,self.top),pv.Point(self.right,self.bottom)])
    
    def flatValue(self):
        return self.value.asCenteredTuple()
    
    def __repr__(self):
        return str(self.generate())

        
        

class GAInteger(GAVariable):
    
    def __init__(self,minval,maxval,**kwargs):
        GAVariable.__init__(self, **kwargs)
        self.minval,self.maxval = min(minval,maxval),max(minval,maxval)
        self.random()
        
    def clipRange(self):
        self.value = _clipRange(self.value, self.minval, self.maxval)

    def random(self):
        ''' Initialize this variable randomly '''
        self.value = random.randint(self.minval,self.maxval)
        self.clipRange()
    
    def combine(self,other):
        '''combine this variable with other.'''
        dist = np.abs(self.value - other.value)
        if random.randint(0,1) == 0:
            self.value = other.value
        self.value += random.randint(-dist-1,dist+1)
        self.clipRange()

    def mutate(self):
        '''introduce mutations into the variable.'''
        if random.random() < self.mutation_rate:
            dist = np.abs(self.maxval-self.minval)
            self.value += random.randint(-dist-1,dist+1)
        self.clipRange()

    def generate(self):
        '''generate the actual value that will be populated in the arguments'''
        return self.value
    
    def __repr__(self):
        return str(self.value)
        

class GABoolean(GAVariable):
    
    def __init__(self,**kwargs):
        GAVariable.__init__(self, **kwargs)
        self.random()
        

    def random(self):
        ''' Initialize this variable randomly '''
        self.value = random.randint(0,1) == 1
    
    def combine(self,other):
        '''combine this variable with other.'''
        if random.randint(0,1) == 0:
            self.value = other.value

    def mutate(self):
        '''introduce mutations into the variable.'''
        if random.random() < self.mutation_rate:
            self.random()

    def generate(self):
        '''generate the actual value that will be populated in the arguments'''
        return self.value
    
    def __repr__(self):
        return str(self.value)
        

class GAChoice(GAVariable):
    
    def __init__(self,*choices, **kwargs):
        GAVariable.__init__(self, **kwargs)
        self.choices = choices
        self.random()
        

    def random(self):
        ''' Initialize this variable randomly '''
        self.value = random.sample(self.choices,1)[0]
    
    def combine(self,other):
        '''combine this variable with other.'''
        if random.randint(0,1) == 0:
            self.value = other.value

    def mutate(self):
        '''introduce mutations into the variable.'''
        if random.random() < self.mutation_rate:
            self.value = random.sample(self.choices,1)[0]

    def generate(self):
        '''generate the actual value that will be populated in the arguments'''
        return self.value
    
    def __repr__(self):
        return str(self.value)
        

class GARanking(GAVariable):
    
    def __init__(self,n_elements,**kwargs):
        GAVariable.__init__(self, **kwargs)
        self.n_elements = n_elements
        self.ranking = list(range(n_elements))
        self.random()
        
    def random(self):
        ''' Initialize this variable randomly '''
        random.shuffle(self.ranking)
    
    def combine(self,other):
        '''combine this variable with other.'''
        a = [ (i,self.ranking[i]) for i in range(len(self.ranking))]
        b = [ (i,other.ranking[i]) for i in range(len(self.ranking))]
        
        a.sort(lambda x,y: cmp(x[1],y[1]))
        b.sort(lambda x,y: cmp(x[1],y[1]))
        
        c = []
        for i in range(len(a)):
            if random.randint(0,1) == 0:
                c.append(a[i])
            else:
                c.append(b[i])

        random.shuffle(c)
        
        c.sort(lambda x,y: cmp(x[0],y[0]))
        
        self.ranking = [r for _,r in c]
        
        
    def mutate(self):
        '''introduce mutations into the variable.'''
        n = int(self.mutation_rate * self.n_elements) + 1
        for _ in range(n):
            i1 = random.randint(0,self.n_elements-1)
            i2 = random.randint(0,self.n_elements-1)
            x = self.ranking[i1]
            del self.ranking[i1]
            self.ranking.insert(i2,x)
        
    
    def generate(self):
        '''generate the actual value that will be populated in the arguments'''
        return self.ranking
    
    def flatValue(self):
        return self.ranking
    
    def __repr__(self):
        return str(self.ranking)
    
    
class GAUnitVector(GAVariable):
    ''' A vector constrained to length 1.0. '''
    def __init__(self,n_elements,**kwargs):
        GAVariable.__init__(self, **kwargs)
        self.n_elements = n_elements
        self.random()
        
    def clipRange(self):
        self.value = pv.unit(self.value)
        
    def random(self):
        ''' Initialize this variable randomly '''
        self.value = np.random.normal(size=[self.n_elements])
        self.clipRange()
    
    def combine(self,other):
        '''combine this variable with other.'''
        for i in range(len(self.value)):
            dist = np.abs(self.value[i] - other.value[i])+0.000001
            if random.randint(0,1) == 0:
                self.value[i] = other.value[i]
            self.value[i] += np.random.normal(0,dist/3.0)
        self.clipRange()

        
        
    def mutate(self):
        '''introduce mutations into the variable.'''
        if random.random() < self.mutation_rate:
            for i in range(len(self.value)):
                self.value[i] += np.random.normal(0,0.02)
        self.clipRange()
        
    
    def generate(self):
        '''generate the actual value that will be populated in the arguments'''
        return self.value
    
    def flatValue(self):
        return list(self.value.flatten())
    
    def __repr__(self):
        return str(self.value)

class GAWeighting(GAVariable):
    ''' A positive vector that sums to 1.0. '''
    
    def __init__(self,n_elements,**kwargs):
        GAVariable.__init__(self, **kwargs)
        self.n_elements = n_elements
        self.random()
        
    def clipRange(self):
        self.value = self.value*(self.value > 0)
        weight = self.value.sum()
        
        # prevent divide by zero
        weight = max(weight,0.0001)
        
        self.value = self.value/weight
        
        
    def random(self):
        ''' Initialize this variable randomly '''
        self.value = np.random.random(size=[self.n_elements])
        self.clipRange()
    
    def combine(self,other):
        '''combine this variable with other.'''
        for i in range(len(self.value)):
            dist = np.abs(self.value[i] - other.value[i])+0.000001
            if random.randint(0,1) == 0:
                self.value[i] = other.value[i]
            self.value[i] += np.random.normal(0,dist/3.0)
        self.clipRange()
        
    def mutate(self):
        '''introduce mutations into the variable.'''
        if random.random() < self.mutation_rate:
            for i in range(len(self.value)):
                self.value[i] += np.random.normal(0,1)/(50.0*self.n_elements)
        self.clipRange()
        
    
    def generate(self):
        '''generate the actual value that will be populated in the arguments'''
        return self.value
    
    def flatValue(self):
        return list(self.value.flatten())
    
    def __repr__(self):
        return str(self.value)
        

class GASequence:
    # TODO: Create this class
    pass

class GASet:
    # TODO: Create this class
    pass
    

        
def list_generate(args):
    for i in range(len(args)):
        #print args[i]
        if isinstance(args[i],GAVariable):
            args[i] = args[i].generate()
        elif isinstance(args[i],(list,tuple)):
            list_generate(args[i])
        elif isinstance(args[i],dict):
            dict_generate(args[i])
            
def dict_generate(args): 
    for i in args.keys():
        if isinstance(args[i],GAVariable):
            args[i] = args[i].generate()
        elif isinstance(args[i],(list,tuple)):
            list_generate(args[i])
        elif isinstance(args[i],dict):
            dict_generate(args[i])
       

def _gaWork(data):
    '''
    This is a work function that gets called on child processes 
    to evaluate a fitness function.
    '''
    try:
        fitness,args,kwargs = data
        assert isinstance(args, (list,tuple))
        assert isinstance(kwargs, dict)
        args = copy.deepcopy(args)
        kwargs = copy.deepcopy(kwargs)
        list_generate(args)
        dict_generate(kwargs)
        score = fitness(*args,**kwargs)
    except:
        print "Error in work queue."
        traceback.print_exc()
        score = np.inf
    return score

        
class GeneticAlgorithm:
    
    def __init__(self,fitness,args=[],kwargs={},initializer=None,initargs=[],population_size=100,n_processes="AUTO"):
        self.fitness = fitness
        self.args = args
        self.kwargs = kwargs
        self.initializer = initializer
        self.initargs=initargs
        self.population_size = population_size
        self.n_processes = n_processes
        self.pool = None
        if self.n_processes == "AUTO":
            self.n_processes = mp.cpu_count()
            
        self.run_data = None
        
        self.running_workers = 0
        
        self.best_score = np.inf
        self.population = []
        self.bests = []
        self.worsts = []
        self.history = []
        self.iter = 0
        
        
    def list_random(self,args):
        for i in range(len(args)):
            if isinstance(args[i],GAVariable):
                args[i].random()
            elif isinstance(args[i],(list,tuple)):
                self.list_random(args[i])
            elif isinstance(args[i],dict):
                self.dict_random(args[i])

    def dict_random(self,args):
        for i in args.keys():
            if isinstance(args[i],GAVariable):
                args[i].random()
            elif isinstance(args[i],(list,tuple)):
                self.list_random(args[i])
            elif isinstance(args[i],dict):
                self.dict_random(args[i])

        
    def random(self):
        '''
        Randomly generate an individual for initialization
        '''
        args = copy.deepcopy(self.args)
        kwargs = copy.deepcopy(self.kwargs)
        
        self.list_random(args)
        self.dict_random(kwargs)
        
        return args,kwargs
        
        
    def list_mutate(self,args):
        for i in range(len(args)):
            if isinstance(args[i],GAVariable):
                args[i].mutate()
            elif isinstance(args[i],(list,tuple)):
                self.list_mutate(args[i])
            elif isinstance(args[i],dict):
                self.dict_mutate(args[i])

    def dict_mutate(self,args):
        for i in args.keys():
            if isinstance(args[i],GAVariable):
                args[i].mutate()
            elif isinstance(args[i],(list,tuple)):
                self.list_mutate(args[i])
            elif isinstance(args[i],dict):
                self.dict_mutate(args[i])

        
    def mutate(self,args):
        '''
        Randomly generate an individual for initialization
        '''
        args = copy.deepcopy(args)
        
        if isinstance(args, (list,tuple)):
            self.list_mutate(args)
        if isinstance(args, dict):
            self.dict_mutate(args)
        
        return args
        
        
    def list_combine(self,args,other):
        assert len(args) == len(other)

        for i in range(len(args)):
            if isinstance(args[i],GAVariable):
                args[i].combine(other[i])
            elif isinstance(args[i],(list,tuple)):
                self.list_combine(args[i], other[i])
            elif isinstance(args[i],dict):
                self.dict_combine(args[i], other[i])
            else:
                pass
    
    def dict_combine(self,args,other):
        assert len(args) == len(other)

        for key in args.keys():
            if isinstance(args[key],GAVariable):
                args[key].combine(other[key])
            elif isinstance(args[key],(list,tuple)):
                self.list_combine(args[key], other[key])
            elif isinstance(args[key],dict):
                self.dict_combine(args[key], other[key])
            else:
                pass
    
    def combine(self,args1,args2):
        # Make deep copies to preserve data
        args = copy.deepcopy(args1)
        other = copy.deepcopy(args2)
        
        if isinstance(args,list):
            self.list_combine(args, other)
        else:
            self.dict_combine(args, other)
        
        return args
    
    
    def addIndividual(self,score,args,kwargs,ilog=None,display=False,verbose=False):
        # This try block allows multiple values to be returned by the fitness function. 
        # Everything except the score will be stored in extra and should be picklible.
        extras = []
        try:
            extras = score[1:]
            score = score[0]
        except:
            # this means score must be a single value
            pass
        
        if not np.isfinite(score):
            return
        
        if score < self.best_score:
            self.best_score = score
            # Print data
            if verbose:
                print "New Best Score:",score
                
                for i in range(len(args)):
                    print "    arg%02d"%i,str(args[i])[:70]
                keys = list(kwargs.keys())
                keys.sort()
                for key in keys:
                    print "    %10s:"%key,str(kwargs[key])[:70]
            
        self.population.append([score,args,kwargs])
        self.population.sort(lambda x,y: cmp(x[0],y[0]))
        self.population = self.population[:self.population_size]

        self.history.append(score)
        self.bests.append(self.population[0][0])
        self.worsts.append(self.population[-1][0])
        
        self.iter += 1

        if verbose:
            self.printPopulation()

        if ilog != None:    
            ilog.pickle([score,args,kwargs],"Fitness_%0.8f"%score)
            for i in xrange(len(extras)):
                extra = extras[i]
                if isinstance(extra,pv.Image):
                    ilog(extra,"extra_%02d_%0.8f"%(i,score))
                else:
                    ilog.pickle(extra,"extra_%02d_%0.8f"%(i,score))
                
        if self.iter % 64 == 0 and (display or ilog is not None):
            plot = self.plotConvergence()
            #pv.Plot(title="Population Statistics",xlabel="Iteration",ylabel="Score")
            #data = [ [i,self.bests[i]] for i in range(len(self.bests)) ]
            #plot.lines(data,width=3,color='green')
            #data = [ [i,self.history[i]] for i in range(len(self.bests)) ]
            #plot.points(data,shape=16,color='blue',size=2)
            #data = [ [i,self.worsts[i]] for i in range(len(self.bests)) ]
            #plot.lines(data,width=3,color='red')
            if ilog is not None:
                ilog(plot,"PopulationData")
            if display:
                plot.show(delay=1,window="Convergence")
    

    def printPopulation(self):
        print "GA Population (Iteration %d):"%self.iter,
        for i in range(len(self.population)):
            if i % 10 == 0:
                print
                print "   ",
            print "%8.3f"%self.population[i][0],
        print
        print
        
    def plotConvergence(self):
        plot = pv.Plot(title="Population Statistics",xlabel="Iteration",ylabel="Score")
        data = [ [i,self.bests[i]] for i in range(len(self.bests)) ]
        plot.lines(data,width=3,color='green')
        data = [ [i,self.history[i]] for i in range(len(self.bests)) ]
        plot.points(data,shape=16,color='blue',size=2)
        data = [ [i,self.worsts[i]] for i in range(len(self.bests)) ]
        plot.lines(data,width=3,color='red')
        return plot
        
        
    
    def optimize(self,max_iter=1000,callback=None,ilog=None,restart_dir=None,display=False, verbose=False):
        '''
        @returns: best_score, args, kwargs
        '''
        
        # Create worker process pool
        if self.n_processes > 1: 
            # print "Init Params (%d cores):"%self.n_processes,str(self.initializer)[:20], str(self.initargs)[:20]
            self.pool = mp.Pool(self.n_processes,initializer=self.initializer,initargs=self.initargs)    
            
        # Initialize the population with random members
        work = []
        for i in range(max(self.population_size-len(self.population),0)):
            args,kwargs = self.random()
            work.append((self.fitness,args,kwargs))

        if restart_dir is not None:
            # Scan restart dir
            files = os.listdir(restart_dir)
            for filename in files:
                if verbose:
                    print filename
                if "_Fitness_" not in filename or not filename.endswith('.pkl'):
                    continue
                path = os.path.join(restart_dir,filename)
                data = pkl.load(open(path,'rb'))
                if verbose:
                    print 'Reloading:',path
                    for each in data:
                        if len(str(each)) > 70:
                            print "    %s..."%str(each)[:70]
                        else:
                            print "    %s"%str(each)
                    
                # Call addIninvidiual
                self.addIndividual(data[0], data[1], data[2], ilog=ilog, display=display,verbose=verbose)
        
        elif self.n_processes > 1:
            scores = self.pool.map(_gaWork, work)
            for i in range(len(scores)):
                score = scores[i]
                _,args,kwargs = work[i]
                self.addIndividual(score,args,kwargs,ilog=ilog,display=display)
        else:
            for each in work:
                score = _gaWork(each)
                _,args,kwargs = each
                self.addIndividual(score,args,kwargs,ilog=ilog,display=display)
                
        if len(self.population) < 2:
            raise ValueError("Could not initialize population.")
        
        
        while self.iter < max_iter:
            
            # Generate the next round of work
            n_work = max(1,self.n_processes)
            work = []
            while len(work) < n_work:
                i1 = random.randint(0,len(self.population)-1)
                i2 = random.randint(0,len(self.population)-1)
                if i1 != i2:
                    _,args1,kwargs1 = self.population[i1]
                    _,args2,kwargs2 = self.population[i2]
                    args = self.combine(args1, args2)
                    kwargs = self.combine(kwargs1, kwargs2)
                    args = self.mutate(args)
                    kwargs = self.mutate(kwargs)
                    work.append((self.fitness,args,kwargs))

            if self.n_processes > 1:
                scores = self.pool.map(_gaWork, work)
                for i in range(len(scores)):
                    score = scores[i]
                    _,args,kwargs = work[i]
                    self.addIndividual(score,args,kwargs,ilog=ilog,display=display)
            else:
                for each in work:
                    score = _gaWork(each)
                    _,args,kwargs = each
                    self.addIndividual(score,args,kwargs,ilog=ilog,display=display)

            if callback != None:
                callback(self.population)
                
        args   = copy.deepcopy(self.population[0][1])
        kwargs = copy.deepcopy(self.population[0][2])
        list_generate(args)
        dict_generate(kwargs)
        
        self.pool = None
        
        return self.population[0][0],args,kwargs


            
            



