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
import unittest
import math
import multiprocessing as mp
import sys
import traceback
import numpy as np
import copy
import time

# Genetic algorithm message types
_GA_EVALUATE="GA_EVALUATE"
_GA_STOP_WORKER="GA_STOP_WORKER"
_GA_SCORE = "GA_SCORE"

def _clipRange(val,minval,maxval):
    minval,maxval = min(minval,maxval),max(minval,maxval)
    val = min(val,maxval)
    val = max(val,minval)
    return val


class GAVariable:
    ''' 
    This is a superclass for a variable that is optimized by the GA. It 
    has three methods that need to be overridden by subclasses: combine,
    mutate, and generate. 
    
    
    '''
    def __init__(self,mutation_rate=0.05):
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
            self.value += np.random.normal(0,dist/3.0)
        self.clipRange()

    def generate(self):
        '''generate the actual value that will be populated in the arguments'''
        return self.value
    
    def __repr__(self):
        return str(self.value)
        

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
    
    def __init__(self,minval,maxval,**kwargs):
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
            self.value = ~self.value

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
    
    def __repr__(self):
        return str(self.ranking)
        

class GASequence:
    # TODO: Create this class
    pass

class GASet:
    # TODO: Create this class
    pass
    

        
def list_generate(args):
    for i in range(len(args)):
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
       

def _gaEvaluate(fitness,args,kwargs):
    
    assert isinstance(args, (list,tuple))
    assert isinstance(kwargs, dict)
    args = copy.deepcopy(args)
    kwargs = copy.deepcopy(kwargs)
    list_generate(args)
    dict_generate(kwargs)
    return fitness(*args,**kwargs)

def _gaWorker(work_queue,results_queue):
    while True:
        command = None
        try:
            command = work_queue.get()
            #print command
            if command[0] == _GA_EVALUATE:
                _,fitness,args,kwargs = command
                score = _gaEvaluate(fitness, args, kwargs)
                #print "worker:",score,args,kwargs
                results_queue.put([_GA_SCORE,score,args,kwargs])
            elif command[0] == _GA_STOP_WORKER:
                #print "Stopping Worker"
                break
            else:
                sys.stderr.write("Worker encountered unknown command of type: %s\n"%(command[0]))
        except:    
            print "Error in work queue."
            traceback.print_exc()
    print "Worker Complete."
    sys.exit()


def _gaWork(data):
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
    
    def __init__(self,fitness,args=[],kwargs={},population_size=100,n_processes="AUTO"):
        self.fitness = fitness
        self.args = args
        self.kwargs = kwargs
        self.population_size = population_size
        self.n_processes = n_processes
        if self.n_processes == "AUTO":
            self.n_processes = mp.cpu_count()
            
        self.run_data = None
        
        self.running_workers = 0
        
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
    

    def printPopulation(self):
        print "GA Population:",
        for i in range(len(self.population)):
            if i % 10 == 0:
                print
                print "   ",
            print "%8.3f"%self.population[i][0],
        print
        print


        
    
    def optimize(self,max_iter=1000,callback=None,ilog=None):
        best_score = 0.0
        best_alg = None
        history = []
        bests = []
        worsts = []
        iter = 0
        self.population = []
        
        # Create worker process pool
        if self.n_processes > 1: 
            pool = mp.Pool(self.n_processes)    
            
        # Initialize the population with random members
        work = []
        for i in range(max(self.population_size-len(self.population),0)):
            args,kwargs = self.random()
            work.append((self.fitness,args,kwargs))

        if self.n_processes > 1:
            scores = pool.map(_gaWork, work)
            for i in range(len(scores)):
                score = scores[i]
                if score == np.inf:
                    continue
                _,args,kwargs = work[i]
                self.population.append([score,args,kwargs])
                iter += 1
        else:
            for each in work:
                score = _gaWork(each)
                if score == np.inf:
                    continue
                _,args,kwargs = each
                self.population.append([score,args,kwargs])
                iter += 1
                
        if len(self.population) < 2:
            raise ValueError("Could not initialize population.")

        self.population.sort(lambda x,y: cmp(x[0],y[0]))
        
        if ilog != None:
            self.printPopulation()
        
        while iter < max_iter:
            
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
                scores = pool.map(_gaWork, work)
                for i in range(len(scores)):
                    score = scores[i]
                    if score == np.inf:
                        continue
                    _,args,kwargs = work[i]
                    self.population.append([score,args,kwargs])
                    iter += 1
            else:
                for each in work:
                    score = _gaWork(each)
                    if score == np.inf:
                        continue
                    _,args,kwargs = each
                    self.population.append([score,args,kwargs])
                    iter += 1

            self.population.sort(lambda x,y: cmp(x[0],y[0]))
            
            self.population = self.population[:self.population_size]

            if ilog != None:
                self.printPopulation()

            if callback != None:
                callback(self.population)
                
        return self.population[0]

            
            



