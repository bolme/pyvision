'''
Copyright David S. Bolme

Created on Feb 28, 2011

@author: bolme
'''
import unittest

import pyvision as pv
import pyvision.optimize.genetic as ga
#import scipy as sp
import numpy as np
import random

def callback(population):
    plot = pv.Plot(x_range=[0,10],y_range=[0,10])
    pts = [ [each[1][0].value,each[1][1].value] for each in population ]
    #pts = [ pv.Point(each[1][0],each[1][1]) for each in population ]
    #print pts
    plot.points(pts)
    plot.show(delay=10)
    #for each in population:
    #    plot.point(each[1])

def unitRectCallback(population):
    if random.random() < 0.05:
        return
    #print "Score:",population[0][0]
    im = pv.Image(np.zeros((1000,1000),dtype=np.float32))
    for each in population:
        rect = each[1][0].generate()
        im.annotateRect(1000*rect,color='gray')
    rect = population[0][1][0].generate()
    im.annotatePolygon((1000*rect).asPolygon(),color='green',width=3)
    target_rect = pv.CenteredRect(.33385,.69348,.3482,.55283)
    im.annotatePolygon((1000*target_rect).asPolygon(),color='white',width=3)
    im.show(delay=1)

class Fitness:
    '''
    Min values is approx -0.63 @ [8,2]
    '''

    def __call__(self,*args,**kwargs):
        
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

class FitnessInt:
    '''
    Min values is approx -0.63 @ [8,2]
    '''

    def __call__(self,*args,**kwargs):
        
        val = 0.0
        x,y = args
        x = 0.01*x
        y = 0.01*y
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

class FitnessNapsack:
    '''
    Greedy:     2122.02170961
    GreedySoft: 2154.98878966
    Best:       2151.52428896    
    '''
    def __init__(self):
        # Napsack items: [weight, value]
        self.napsack = [ [3.8498804596455516, 2.6158205192194539],[4.5189125797608938, 8.8903995320853131], 
                         [6.1173469407134675, 4.3700978933395547],[6.9007385880574246, 3.8692730664014396], 
                         [8.9144167464846280, 4.7836252543936624],[0.4693004547062529, 2.8221296183042233], 
                         [2.4015132415591500, 8.8948724188649706],[7.2119610986308178, 8.2189050746603627], 
                         [6.8742463976743275, 0.1290666476865909],[6.1975443218025514, 9.6352852208891058], 
                         [0.3292378884007918, 7.6920571985448998],[6.7891716791357046, 4.8137531832815919], 
                         [8.5514192788966188, 5.2561657811810658],[5.9180556988418376, 1.3006205399658544], 
                         [8.0671560149320385, 7.7731218970116878],[6.0386660285670946, 1.7900969964148827], 
                         [3.3174935387979518, 3.6922312089695852],[6.5380617593654211, 1.1739197783343658], 
                         [7.2019215829854657, 4.8665738855354270],[0.0410115173309621, 9.5962099162512047], 
                         [0.5160585719043453, 2.7326870371658032],[1.8277846017416532, 6.3544813031122178], 
                         [9.8842564999823956, 3.0262295865225219],[7.4860989752014362, 0.3089877071335456], 
                         [8.0834584531638498, 1.6854024634582421],[0.5425377493637173, 6.5310787408262119], 
                         [3.4843324863054215, 2.4315902552567992],[3.5779710897236816, 8.2106929764375973], 
                         [5.7742315723856166, 1.2463078836786967],[6.0318777890316246, 0.3670870890735067], 
                         [2.8774075634643435, 6.1738364797893110],[9.1450725522551473, 7.7914018397822620], 
                         [3.2047077180287742, 9.4598588612822070],[1.1167027704978094, 5.1292430796049526], 
                         [7.5335325192597722, 4.2008849118851144],[8.0060003984360364, 0.3453745034545285], 
                         [4.1572532576362811, 5.5826759970594120],[4.8295916369681580, 6.0661961083185245], 
                         [0.9635142164770282, 1.1439694238709974],[0.5460964518464928, 6.8024967241254943], 
                         [5.4975251734276585, 4.5029749519689997],[5.0668911404556907, 6.2606269283238518], 
                         [2.5506607296936989, 2.0198909909102314],[9.4149331889825678, 7.5664649402992747], 
                         [7.8120957703705578, 2.5173457559511805],[2.4158740472850893, 3.6750720594921651], 
                         [4.7301946764109806, 6.7643300136764939],[1.9817641689609866, 7.2181879836120642], 
                         [5.0366321352514918, 2.5843130510050702],[2.5304869867849886, 9.7949844854539752], 
                         [2.9403099192214190, 9.2996118833243067],[6.9036989146152470, 7.8085029359642233], 
                         [2.4789462548245877, 2.1278890919908844],[6.1657324514331684, 9.3948950035790020], 
                         [5.9704063048726823, 8.4484636770335069],[6.1647182439418247, 4.9416598764541302], 
                         [1.3433171070152927, 0.9436729442012903],[3.0503134091567716, 8.5014086442476948], 
                         [8.9857262873424233, 4.4395019567438601],[6.7615506265458354, 3.1237402173999982], 
                         [6.5496041740782642, 5.3398880866520937],[8.8279482373425360, 4.6869911466232654], 
                         [5.6171331059380067, 1.7649945424378233],[4.5323876725646244, 0.8070819665916628], 
                         [6.7986694347766043, 3.8074853591525404],[9.3562536405189149, 9.0672137648276987], 
                         [4.3728896028925277, 2.2430957497192483],[2.2243966554075776, 5.8479695623273065], 
                         [2.2920587007505979, 2.2437759237758925],[7.2566755756751231, 5.0412626439851183], 
                         [6.7249111241940165, 1.5536858497584638],[8.0579178682464629, 1.2343910851579643], 
                         [6.8754141358879455, 6.0317671757976203],[1.1863969223776127, 1.5399740016422581], 
                         [1.7084302568288123, 6.6386825337328892],[3.6323577569600474, 6.9916456275588459], 
                         [8.8824506098079468, 4.0975086517123795],[8.4492722663078563, 9.5384267527149156], 
                         [7.5004514342733666, 9.5470395401946533],[3.8623626673491032, 1.3184566778556683], 
                         [5.2176965749255499, 1.6969605910220198],[7.6971622834725126, 9.3186430484212899], 
                         [6.4352319531434858, 2.6571361421874986],[9.4682994732626895, 5.0475691909212648], 
                         [3.8968677899083382, 7.7746635634368637],[0.7252874406688580, 7.2339157247591448],
                         [7.4042465769195678, 4.6402243401612644],[2.0952606356353831, 0.2403152717438195], 
                         [5.9367836289158511, 0.8062172968135816],[1.0589966668958084, 9.4899899882108389], 
                         [8.1105586869235786, 8.7516004555149731],[4.8618052284517219, 0.5721366593317134], 
                         [1.5843720121388560, 1.6309946195183456],[8.2770272808876406, 2.2543326689249712], 
                         [2.2764817157991581, 2.9509378599705984],[0.7103568684070182, 6.9414136185046340], 
                         [2.6785942780931373, 2.5005917823412362],[0.7277672847554017, 4.5486326841546845], 
                         [7.6984307141459416, 7.0418998319943871],[8.4203479639455541, 7.2403257387387328]]
        self.napsack = 50+np.array(self.napsack)
        
        self.max_weight = 2000.0

    def solve(self):
        # split into seperate lists
        weights = self.napsack[:,0].flatten()
        values  = self.napsack[:,1].flatten()

        # compute density        
        density = values/weights

        # Reorder by density
        order = density.argsort()[::-1]
        weights = weights[order]
        values = values[order]
        density = density[order]
        
        _ = self.greedy(weights,values,self.max_weight)
        #score = self.greedy_soft(weights,values,self.max_weight)
        #print "Greedy",score,remaining,solution
        #print "GreedySoft",self.greedy_soft(weights, values, self.max_weight)
        #print "Search",self.search(weights, values, self.max_weight,0.0,0.0,[])
            
    def greedy(self,weights,values,remaining):
        if len(weights) == 0:
            return [],0.0,remaining
        
        if weights[0] <= remaining:
            solution,score,remaining = self.greedy(weights[1:],values[1:],remaining-weights[0])
            return [True]+solution,values[0]+score,remaining
        else: 
            solution,score,remaining = self.greedy(weights[1:],values[1:],remaining)
            return [False]+solution,score,remaining

    def greedy_soft(self,weights,values,remaining):
        if len(weights) == 0:
            return 0.0
        
        if weights[0] <= remaining:
            score = self.greedy_soft(weights[1:],values[1:],remaining-weights[0])
            return values[0]+score
        else: 
            frac = float(remaining) / weights[0]
            score = frac*values[0]
            return score
    
    def search(self,weights,values,remaining,best,score,solution):
        if len(weights) == 0 and score > best:
            #print "New Best:",score,solution
            #assert 0 
            return score

        soft = self.greedy_soft(weights, values, remaining)
        
        if score + soft > best:
            if weights[0] <= remaining:
                new_score = self.search(weights[1:],values[1:],remaining-weights[0],best,score+values[0],solution+[True])
                if new_score > best:
                    best = new_score

            new_score = self.search(weights[1:],values[1:],remaining,best,score,solution+[False])
            if new_score > best:
                best = new_score
            
        return best

    def __call__(self,*args,**kwargs):
        ranking = args[0]
        weight = 0.0
        value = 0.0
        for i in ranking:
            weight += self.napsack[i,0]
            if weight < self.max_weight:
                value += self.napsack[i,1]
            else:
                break
        return -value
            
    
def fitnessUnitRect(rect,**kwargs):
    target_rect = pv.CenteredRect(.33385,.69348,.3482,.55283)
    return -target_rect.overlap(rect)


def fitnessUnitVector(vec,**kwargs):
    target = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    sum_sq = 0.0
    for i in range(len(vec)):
        sum_sq += (target[i] - vec[i])**2
    score =  np.sqrt(sum_sq)
    #print "Score:",score
    return score


class GeneticAlgorithmTest(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test2DSurfaceFloatMP(self):
        alg = ga.GeneticAlgorithm(Fitness(),[ga.GAFloat(0.0,10.0),ga.GAFloat(0.0,10.0)],n_processes=4)
        _ = alg.optimize(max_iter=1000)
        #print "test2DSurfaceFloatMP",score,args,kwargs
        
    def test2DSurfaceFloatSP(self):
        alg = ga.GeneticAlgorithm(Fitness(),[ga.GAFloat(0.0,10.0),ga.GAFloat(0.0,10.0)],n_processes=1)
        _ = alg.optimize(max_iter=5000,callback=callback)
        #print "test2DSurfaceFloatSP",score,args,kwargs
        
    def test2DSurfaceIntSP(self):
        alg = ga.GeneticAlgorithm(FitnessInt(),[ga.GAInteger(0,1000),ga.GAInteger(0,1000)],n_processes=1)
        _ = alg.optimize(max_iter=1000)
        #print "test2DSurfaceIntSP",score,args,kwargs
        
    def testNapsack(self):
        fitness = FitnessNapsack()
        #fitness.solve()
        alg = ga.GeneticAlgorithm(fitness,[ga.GARanking(100)],n_processes=1)
        result = alg.optimize(max_iter=1000)
        #print "testNapsack",result

    def testGAUnitVector(self):
        #print "Running unitrect test"
        alg = ga.GeneticAlgorithm(fitnessUnitVector,[ga.GAUnitVector(9)],population_size=20,n_processes=1)
        _ = alg.optimize(max_iter=10000)
        #print "testGAUnitVector",score,args,kwargs
        
    
    def testGAUnitRect(self):
        #print "Running unitrect test"
        alg = ga.GeneticAlgorithm(fitnessUnitRect,[ga.GAUnitRect2(min_width=0.05,min_height=0.05,max_height=1.0,max_width=1.0)],population_size=20,n_processes=4)
        _ = alg.optimize(max_iter=1000,callback=unitRectCallback)
        #print "testGAUnitRect",score,args,kwargs
        
        
    def testCircularRange(self):
        self.assertAlmostEqual(ga._circularRange(10, -np.pi, np.pi),10-4*np.pi)
        self.assertAlmostEqual(ga._circularRange(50.77, -np.pi, np.pi),50.77-16*np.pi)
        self.assertAlmostEqual(ga._circularRange(-10, -np.pi, np.pi),-10+4*np.pi)
        self.assertAlmostEqual(ga._circularRange(-50.77, -np.pi, np.pi),-50.77+16*np.pi)
        self.assertAlmostEqual(ga._circularRange(-1.54, -np.pi, np.pi),-1.54)
        self.assertAlmostEqual(ga._circularRange(0.5, -np.pi, np.pi),0.5)
        
    def testGAAngle(self):
        # Test random
        for _ in range(100):
            ang = ga.GAAngle()
            #print ang
        # Test mutate
        for _ in range(100):
            ang = ga.GAAngle(mutation_rate=0.5)
            #print ang,
            ang.mutate()
            #print ang
            
        # Test Combine
        for _ in range(100):
            ang1 = ga.GAAngle()
            ang2 = ga.GAAngle()
            ang2.value = ang1.value + 0.1
            ang2.clipRange()
            #print ang1,ang2,
            ang1.combine(ang2)
            #print ang1
        
        # Test Combine
        for _ in range(100):
            ang1 = ga.GAAngle()
            ang2 = ga.GAAngle()
            ang2.value = ang1.value + 1
            ang2.clipRange()
            #print ang1,ang2,
            ang1.combine(ang2)
            #print ang1
        
        
        
        


if __name__ == "__main__":
    print "Running GA Test Suite"
    #import sys;sys.argv = ['', 'Test.test2DSurface']
    unittest.main()