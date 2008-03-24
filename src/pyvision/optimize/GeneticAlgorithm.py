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

class RangeVariable:
    def __init__(self,values):
        self.values = values
        
    
    def random(self):
        return random.choice(self.values)
        
    def combine(self,c1,c2,mutation_rate = 0.05):
        #find c1, find c2
        fc1 = self.values.index(c1)
        fc2 = self.values.index(c2)
        diff = abs(fc1-fc2)
        #expand = int(mutation_rate*len(self.valuens))
        if random.random() < mutation_rate:
            return self.random()
        
        lb = int(min(fc1,fc2) - 0.25*diff - 1)
        ub = int(max(fc1,fc2) + 0.25*diff + 1)
        lb = max(0,lb)
        ub = min(ub,len(self.values)-1)
        
        return self.values[random.randint(lb,ub)]
                                          
class ChoiceVariable:
    ''' 
    This class represents a variable that can take the value of a fixed number of choices.
    '''
    
    def __init__(self,choices,mutation_rate = 0.05):
        '''
        Initialize the variable.
        
        Parameters:
         - choices - a list containing the possible values this variable can take.
         * mutation_rate - determines how often a random value is returned from the combine method.
        '''
        self.choices = choices
        self.mutation_rate = mutation_rate
        
    def random(self):
        '''
        Randomly selects a value from the list.
        
        Returns: A random value for this variable
        '''
        return random.choice(self.choices)
    
    def combine(self,c1,c2):
        '''
        Combines two values into one.
        
        Typically chooses one of the two value.  If a mutation is selected a random value is returned.
        
        Parameters:
         - c1,c2 - values of the two 'parents'  the child is typically one of these.
        
        Returns: A value form the choices list.
        '''
        if random.random() < self.mutation_rate:
            return self.random()
        else:
            assert c1 in self.choices
            assert c2 in self.choices
            return random.choice([c1,c2])

class SetVariable:
    '''
    Represents a variable that can contain a set of objects.
    '''
    def __init__(self,items,mutation_rate=0.05,mutation_size=None):
        '''
        Initialize the variable.
        
        Parameters:
         - items - The set of all posible objects.
         * mutation_rate - how often mutations are added to the set.
         * mutation_size - how many objects could be added as the result of the mutation.
        '''
        self.items = items
        self.mutation_rate = mutation_rate
        self.mutation_size = mutation_size
        if self.mutation_size == None:
            self.mutation_size = min(1+len(self.items)/20,len(self.items))
        
    def random(self):
        '''
        Randomly selects a subset.  The subset can have a size of 0 to 
        len(items) and will contain random members.
        
        Returns: subset
        '''
        n = random.randint(0,len(self.items))
        
        return set(random.sample(self.items,n))
              
    def combine(self,s1,s2):
        '''
        Combines the two sets into one.  Typically the result will contain 
        only items from the two sets and will have roughly simillar 
        lengths.  If there is a mutation more items may be randomly added
        to the new set.
        
        Parameters:
         - s1,s2 - Sets to combine.
         
         Returns: a random set composed mostly of the inputs.
        '''
        ls1 = len(s1)
        ls2 = len(s2)
        #lb = min(ls1,ls2)
        #ub = min(ls1,ls2)
        sa = s1.union(s2)
        if random.random() < self.mutation_rate:
            new_items = random.sample(self.items,self.mutation_size)
            sa = s1.union(s2).union(new_items)
        ub = len(sa)
        me = (ls1+ls2)/2
        lb = max(0,me - (ub-me))
        n = random.randint(lb,ub)
        return set( random.sample(sa,n) )

        


class GeneticAlgorithm:
    def __init__(self, fitness, search_space, n=50, minimum = True, ilog=None):
        ''' Configure the genetic algorithm '''
        self.n = n
        self.population = []
        self.fitness = fitness
        self.search_space = search_space
        self.minimum = minimum
        self.tested = set()
        
    def optimize(self, generations=10000):
        ''' Find the optimum value '''
        self.generatePopulation()
        best = self.population[0][1]
        
        for i in range(generations):
            self.iteration()
            if self.minimum:
                if self.population[0][1] < best:
                    best = self.population[0][1]
                    #print "Iteration:",i,"Best:",best,"Worst:",self.population[-1][1]
            else:
                if self.population[0][1] > best:
                    best = self.population[0][1]
            print "Iteration:",i,"Best:",best,"Worst:",self.population[-1][1]
        
    def iteration(self):
        parents = [self.population[0]]
        rank = 2
        while True:
            if rank > len(self.population): break
            parents.append(random.choice(self.population[:rank]))
            rank *= 2
        parents.append(random.choice(self.population))
        
        idv1 = random.choice(parents)[0]
        idv2 = random.choice(parents)[0]
        for i in range(25):
            idv3 = []
            for j in range(len(self.search_space)):
                v1 = idv1[j]
                v2 = idv2[j]
                v3 = self.search_space[j].combine(v1,v2)
                idv3.append(v3)
            if repr(idv3) not in self.tested:
                #self.tested.add(repr(idv3))
                break
        if repr(idv3) not in self.tested: 
            self.tested.add(repr(idv3))
        else:
            print "No new combination found."
            return
        fit = self.fitness(idv3)
        self.population.append([idv3,fit])
        if self.minimum:
            self.population.sort(lambda x,y: cmp(x[1],y[1]))
        else:
            self.population.sort(lambda x,y: cmp(y[1],x[1]))
        self.population = self.population[:self.n]        
        #print "Best:", self.population[0][1], "Worst:",self.population[-1][1]
            
        
    def generatePopulation(self):
        for i in range(self.n):
            idv = self.random()
            fit = self.fitness(idv)
            self.population.append([idv,fit])
            
        if self.minimum:
            self.population.sort(lambda x,y: cmp(x[1],y[1]))
        else:
            self.population.sort(lambda x,y: cmp(y[1],x[1]))
        print "Best: ",self.population[0][1]
        
    def random(self):
        idv = []
        for var in self.search_space:
            idv.append(var.random())
        return idv
        
    def combine(self,id1,id2, mutation_rate):
        ''' Combine two individuals '''
           
# cost, weight 
#napsack = {}
#for i in range(100):
#    item = i
#    cost = 10*random.random()
#    weight = 10*random.random()
#    napsack[item] = [cost,weight]
#print repr(napsack)

class _TestGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        self.napsack = {0: [3.8498804596455516, 2.6158205192194539], 1: [4.5189125797608938, 8.8903995320853131], 
                           2: [6.1173469407134675, 4.3700978933395547], 3: [6.9007385880574246, 3.8692730664014396], 
                           4: [8.914416746484628, 4.7836252543936624], 5: [0.46930045470625292, 2.8221296183042233], 
                           6: [2.40151324155915, 8.8948724188649706], 7: [7.2119610986308178, 8.2189050746603627], 
                           8: [6.8742463976743275, 0.12906664768659093], 9: [6.1975443218025514, 9.6352852208891058], 
                           10: [0.32923788840079182, 7.6920571985448998], 11: [6.7891716791357046, 4.8137531832815919], 
                           12: [8.5514192788966188, 5.2561657811810658], 13: [5.9180556988418376, 1.3006205399658544], 
                           14: [8.0671560149320385, 7.7731218970116878], 15: [6.0386660285670946, 1.7900969964148827], 
                           16: [3.3174935387979518, 3.6922312089695852], 17: [6.5380617593654211, 1.1739197783343658], 
                           18: [7.2019215829854657, 4.866573885535427], 19: [0.041011517330962199, 9.5962099162512047], 
                           20: [0.51605857190434534, 2.7326870371658032], 21: [1.8277846017416532, 6.3544813031122178], 
                           22: [9.8842564999823956, 3.0262295865225219], 23: [7.4860989752014362, 0.30898770713354562], 
                           24: [8.0834584531638498, 1.6854024634582421], 25: [0.54253774936371735, 6.5310787408262119], 
                           26: [3.4843324863054215, 2.4315902552567992], 27: [3.5779710897236816, 8.2106929764375973], 
                           28: [5.7742315723856166, 1.2463078836786967], 29: [6.0318777890316246, 0.36708708907350673], 
                           30: [2.8774075634643435, 6.173836479789311], 31: [9.1450725522551473, 7.791401839782262], 
                           32: [3.2047077180287742, 9.459858861282207], 33: [1.1167027704978094, 5.1292430796049526], 
                           34: [7.5335325192597722, 4.2008849118851144], 35: [8.0060003984360364, 0.34537450345452858], 
                           36: [4.1572532576362811, 5.582675997059412], 37: [4.829591636968158, 6.0661961083185245], 
                           38: [0.96351421647702828, 1.1439694238709974], 39: [0.5460964518464928, 6.8024967241254943], 
                           40: [5.4975251734276585, 4.5029749519689997], 41: [5.0668911404556907, 6.2606269283238518], 
                           42: [2.5506607296936989, 2.0198909909102314], 43: [9.4149331889825678, 7.5664649402992747], 
                           44: [7.8120957703705578, 2.5173457559511805], 45: [2.4158740472850893, 3.6750720594921651], 
                           46: [4.7301946764109806, 6.7643300136764939], 47: [1.9817641689609866, 7.2181879836120642], 
                           48: [5.0366321352514918, 2.5843130510050702], 49: [2.5304869867849886, 9.7949844854539752], 
                           50: [2.940309919221419, 9.2996118833243067], 51: [6.903698914615247, 7.8085029359642233], 
                           52: [2.4789462548245877, 2.1278890919908844], 53: [6.1657324514331684, 9.394895003579002], 
                           54: [5.9704063048726823, 8.4484636770335069], 55: [6.1647182439418247, 4.9416598764541302], 
                           56: [1.3433171070152927, 0.94367294420129033], 57: [3.0503134091567716, 8.5014086442476948], 
                           58: [8.9857262873424233, 4.4395019567438601], 59: [6.7615506265458354, 3.1237402173999982], 
                           60: [6.5496041740782642, 5.3398880866520937], 61: [8.827948237342536, 4.6869911466232654], 
                           62: [5.6171331059380067, 1.7649945424378233], 63: [4.5323876725646244, 0.80708196659166287], 
                           64: [6.7986694347766043, 3.8074853591525404], 65: [9.3562536405189149, 9.0672137648276987], 
                           66: [4.3728896028925277, 2.2430957497192483], 67: [2.2243966554075776, 5.8479695623273065], 
                           68: [2.2920587007505979, 2.2437759237758925], 69: [7.2566755756751231, 5.0412626439851183], 
                           70: [6.7249111241940165, 1.5536858497584638], 71: [8.0579178682464629, 1.2343910851579643], 
                           72: [6.8754141358879455, 6.0317671757976203], 73: [1.1863969223776127, 1.5399740016422581], 
                           74: [1.7084302568288123, 6.6386825337328892], 75: [3.6323577569600474, 6.9916456275588459], 
                           76: [8.8824506098079468, 4.0975086517123795], 77: [8.4492722663078563, 9.5384267527149156], 
                           78: [7.5004514342733666, 9.5470395401946533], 79: [3.8623626673491032, 1.3184566778556683], 
                           80: [5.2176965749255499, 1.6969605910220198], 81: [7.6971622834725126, 9.3186430484212899], 
                           82: [6.4352319531434858, 2.6571361421874986], 83: [9.4682994732626895, 5.0475691909212648], 
                           84: [3.8968677899083382, 7.7746635634368637], 85: [0.72528744066885809, 7.2339157247591448],
                           86: [7.4042465769195678, 4.6402243401612644], 87: [2.0952606356353831, 0.2403152717438195], 
                           88: [5.9367836289158511, 0.80621729681358167], 89: [1.0589966668958084, 9.4899899882108389], 
                           90: [8.1105586869235786, 8.7516004555149731], 91: [4.8618052284517219, 0.57213665933171343], 
                           92: [1.584372012138856, 1.6309946195183456], 93: [8.2770272808876406, 2.2543326689249712], 
                           94: [2.2764817157991581, 2.9509378599705984], 95: [0.71035686840701828, 6.941413618504634], 
                           96: [2.6785942780931373, 2.5005917823412362], 97: [0.72776728475540176, 4.5486326841546845], 
                           98: [7.6984307141459416, 7.0418998319943871], 99: [8.4203479639455541, 7.2403257387387328]}

    def test_napsack_choice(self):
        class NapsackFitness:
            def __init__(self,napsack,max_weight):
                self.napsack = napsack
                self.keys = napsack.keys()
                self.keys.sort()
                self.max_weight = max_weight
            
            def __call__(self,idv):
                profit = 0.0
                weight = 0.0
                for i in range(len(idv)):
                    if idv[i] == 1:
                        profit += self.napsack[self.keys[i]][0]
                        weight += self.napsack[self.keys[i]][1]
                if weight <= self.max_weight:
                    return profit
                else:
                    # Profit drops exponetailly if weight is over
                    return profit/(2*math.exp(weight - self.max_weight))
        
        # Can take about half the items        
        fitness = NapsackFitness(self.napsack,250.0)
        
        search_space = []
        keys = self.napsack.keys()
        keys.sort()
        for key in keys:
            search_space.append(ChoiceVariable([0,1]))
                    
        ga = GeneticAlgorithm(fitness,search_space,n=200,minimum=False)
        ga.optimize()

    def test_napsack_set(self):
        class NapsackFitness:
            def __init__(self,napsack,max_weight):
                self.napsack = napsack
                self.max_weight = max_weight
            
            def __call__(self,idv):
                profit = 0.0
                weight = 0.0
                for item in idv[0]:
                    profit += self.napsack[item][0]
                    weight += self.napsack[item][1]
                if weight <= self.max_weight:
                    return profit
                else:
                    # Profit drops exponetailly if weight is over
                    return profit/(2*math.exp(weight - self.max_weight))
        
        # Can take about half the items        
        fitness = NapsackFitness(self.napsack,250.0)
        
        search_space = [SetVariable(self.napsack.keys())]
        
        ga = GeneticAlgorithm(fitness,search_space,n = 200,minimum=False)
        ga.optimize()



