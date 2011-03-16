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

import copy
import numpy as np
import pyvision as pv
import unittest
    
def buildPositiveNegativeLists(names,matrix,class_equal):
    positive = []
    negative = []
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            if class_equal(names[i],names[j]):
                positive.append(matrix[i][j])
            else:
                negative.append(matrix[i][j])
    return positive,negative
    
def readCsuDistanceMatrix(directory):
    from os import listdir
    from os.path import join
    
    filenames = []
    for each in listdir(directory):
        if each[-4:] == ".sfi":
            filenames.append(each)
        
    filenames.sort()
    matrix = []
    for filename in filenames:
        f = open(join(directory,filename),'r')
        row = []
        count = 0
        for line in f:
            fname,sim = line.split()
            assert fname == filenames[count]
            sim = -float(sim)
            row.append(sim)
            count += 1
        f.close()
        assert len(row) == len(filenames)
        matrix.append(row)
    assert len(matrix) == len(filenames)
        
    return filenames,matrix

class ROCPoint:
    def __init__(self,nscore,nidx,n,far,mscore,midx,m,frr):
        self.nscore,self.nidx,self.n,self.far,self.mscore,self.midx,self.m,self.frr = nscore,nidx,n,far,mscore,midx,m,frr
        self.tar = 1.0 - self.frr
        self.trr = 1.0 - self.far
    
    def __str__(self):
        return "ROCPoint %f FRR at %f FAR"%(self.frr,self.far)

ROC_LOG_SAMPLED = 1
ROC_MATCH_SAMPLED = 2
ROC_PRECISE_SAMPLED = 3
ROC_PRECISE_ALL = 4

class ROC:
    
    # TODO: add options for area under curve (AUC) and equal error rate (EER)
    
    def __init__(self,match,nonmatch,is_distance=True):
        self.match = np.array(match).copy()
        self.nonmatch = np.array(nonmatch).copy()
        self.is_distance = is_distance
        
        if not is_distance:
            self.match    = -self.match
            self.nonmatch = -self.nonmatch
        
        
        self.match.sort()
        self.nonmatch.sort()
        
        
    def getCurve(self,method=ROC_LOG_SAMPLED):
        """
        returns header,rows
        """
        header = ["score","frr","far","trr","tar"]
        rows = []
        
        if method == ROC_LOG_SAMPLED:
            for far in 10**np.arange(-6,0.0000001,0.01):
                point = self.getFAR(far)
                
                row = [point.nscore,point.frr,point.far,point.trr,point.tar]
                rows.append(row)

        if method == ROC_MATCH_SAMPLED:
            for score in self.match:
                if self.is_distance:
                    point = self.getMatch(score)
                else:
                    point = self.getMatch(-score)
                row = [point.nscore,point.frr,point.far,point.trr,point.tar]
                rows.append(row)
            
        if method == ROC_PRECISE_SAMPLED:
            m = len(self.match)
            n = len(self.nonmatch)
            both = np.concatenate([self.match,self.nonmatch])
            matches = np.array(len(self.match)*[1]+len(self.nonmatch)*[0])
            nonmatches = np.array(len(self.match)*[0]+len(self.nonmatch)*[1])
            order = both.argsort()
            scores = both[order]
            matches = matches[order]
            nonmatches = nonmatches[order]
            tar = matches.cumsum()/float(m)
            far = nonmatches.cumsum()/float(n)
            keep = np.ones(len(tar),dtype=np.bool)
            keep[1:-1][(far[:-2] == far[1:-1]) & (far[2:] == far[1:-1])] = False
            keep[1:-1][(tar[:-2] == tar[1:-1]) & (tar[2:] == tar[1:-1])] = False
            scores = scores[keep]
            tar = tar[keep]
            far = far[keep]
            rows = np.array([scores,1.0-tar,far,1.0-far,tar]).T
            
        if method == ROC_PRECISE_ALL:
            m = len(self.match)
            n = len(self.nonmatch)
            both = np.concatenate([self.match,self.nonmatch])
            matches = np.array(len(self.match)*[1]+len(self.nonmatch)*[0])
            nonmatches = np.array(len(self.match)*[0]+len(self.nonmatch)*[1])
            order = both.argsort()
            scores = both[order]
            matches = matches[order]
            nonmatches = nonmatches[order]
            tar = matches.cumsum()/float(m)
            far = nonmatches.cumsum()/float(n)
            rows = np.array([scores,1.0-tar,far,1.0-far,tar]).T
            
        return header,rows
        
            
    def getFAR(self,far):
        match = self.match
        nonmatch = self.nonmatch
        orig_far = far
        
        m = len(match)
        n = len(nonmatch)
        
        nidx = int(round(far*n))
        far = float(nidx)/n
        if nidx >= len(nonmatch):
            nscore = None
        #elif nidx == 0:
        #    nscore = nonmatch[nidx]
        else:
            nscore = nonmatch[nidx]
        
        if nscore != None:
            midx = np.searchsorted(match,nscore,side='left')    
        else:
            midx = m
             
        frr = 1.0-float(midx)/m
        if midx >= len(match):
            mscore = None
        else:
            mscore = match[midx]

        #assert mscore == None or mscore <= nscore
        
        #if nidx == 0:
        #print "Zero:",orig_far,nscore,nidx,n,far,mscore,midx,m,frr
        #print nonmatch
        #print match
        if self.is_distance:
            return ROCPoint(nscore,nidx,n,far,mscore,midx,m,frr)
        else:
            if nscore != None:
                nscore = -nscore
            if mscore != None:
                mscore = -mscore
            return ROCPoint(nscore,nidx,n,far,mscore,midx,m,frr)

    def getFRR(self,frr):
        match = self.match
        nonmatch = self.nonmatch
        
        m = len(match)
        n = len(nonmatch)
        
        midx = int(round((1.0-frr)*m))
        frr = 1.0 - float(midx)/m
        if midx >= len(match):
            mscore = None
        else:
            mscore = match[midx-1]
    
        nidx = np.searchsorted(nonmatch,mscore)
        far = float(nidx)/n
        if nidx-1 < 0:
            nscore = None
        else:
            nscore = nonmatch[nidx-1]
                
        assert nscore == None or mscore >= nscore
        
        if self.is_distance:
            return ROCPoint(nscore,nidx,n,far,mscore,midx,m,frr)
        else:
            if nscore != None:
                nscore = -nscore
            if mscore != None:
                mscore = -mscore
            return ROCPoint(nscore,nidx,n,far,mscore,midx,m,frr)

                
    def getEER(self):
        _,curve = self.getCurve(method=ROC_PRECISE_SAMPLED)
        
        for score,frr,far,trr,tar in curve:
            if far > frr:
                break
        
        return far

                
    def getMatch(self,mscore):
        if not self.is_distance:
            mscore = -mscore
        match = self.match
        nonmatch = self.nonmatch
        
        m = len(match)
        n = len(nonmatch)
        
        midx = np.searchsorted(match,mscore)
        #midx = int(round((1.0-frr)*m))
        frr = 1.0 - float(midx)/m
    
        nidx = np.searchsorted(nonmatch,mscore)
        far = float(nidx)/n
        if nidx-1 < 0:
            nscore = None
        else:
            nscore = nonmatch[nidx-1]
                
        assert nscore == None or mscore >= nscore
        
        if self.is_distance:
            return ROCPoint(nscore,nidx,n,far,mscore,midx,m,frr)
        else:
            if nscore != None:
                nscore = -nscore
            if mscore != None:
                mscore = -mscore
            return ROCPoint(nscore,nidx,n,far,mscore,midx,m,frr)
                
                
    def results(self):
        table = pv.Table(default_value="")

        pt = self.getFAR(0.001)
        table[0,'FAR'] = 0.001
        table[0,'TAR'] = pt.tar
        
        pt = self.getFAR(0.01)
        table[1,'FAR'] = 0.01
        table[1,'TAR'] = pt.tar

        pt = self.getFAR(0.1)
        table[2,'FAR'] = 0.1
        table[2,'TAR'] = pt.tar
        
        table[3,'EER'] = self.getEER()
        table[4,'AUC'] = self.getAUC()
        
        return table
    
    def plot(self,plot,method=ROC_PRECISE_SAMPLED,**kwargs):
        _,curve = self.getCurve(method=method)
        points = [[0.0,0.0]] + [ [x,y] for _,_,x,_,y in curve ]+[[1.0,1.0]]
        plot.lines(points,**kwargs)

    
    def getAUC(self,**kwargs):
        _,curve = self.getCurve(method=ROC_PRECISE_SAMPLED)
        points = [[0.0,0.0]] + [ [x,y] for _,_,x,_,y in curve ]+[[1.0,1.0]]
        auc = 0.0
        for i in range(len(points)-1):
            y = 0.5*(points[i][1] + points[i+1][1])
            dx = points[i+1][0] - points[i][0]
            auc += y*dx
        return auc
    
        
class ROCTest(unittest.TestCase):
    
    def setUp(self):
        self.match = [-0.3126333774819825, 1.0777130777174635, 1.1045667643589598, 1.022042510130833, -0.58552060836929942, 
                      -0.59682041549981257, -1.4873074501595509, -0.49958344415133116, 0.36814022366653204, 0.9292572191289511, 
                      0.56740023418734642, -1.3117888037744228, 1.7695340517922449, 0.4098641799520919, 0.43642273019233646, 
                      -0.14893755966202349, -1.3490978540595631, 0.18192684849996424, 1.4547096287864199, 1.1698331636208563, 
                      0.40439133210485323, -1.2333503530027063, -0.1765228044654879, 0.070450455376130774, -0.85038212096409027, 
                      1.6679580794589872, 1.1589669301436729, 1.1756719870079611, -1.0799654160891785, -0.11025751625199756, 
                      0.098294009710337069, -0.49832134960232527, -1.4626964355118197, 1.1064208531539006, -0.4251178714268497, 
                      1.297279496554774, -1.9318553699779215, -1.2787762925010133, 0.92426958166955997, 0.38501300779378478, 
                      -1.7823019361063408, -0.43568112010605503, 0.65785964631537774, -0.63359960475947019, -0.02194247979690072, 
                      -0.55595471945130093, -0.8043184500851891, 0.13759846217215868, 0.12524112107182517, 0.48665310853849575, 
                      -1.2285460272311253, -1.7721136485547013, 1.4552123210449597, -0.38319646950962838, 0.96456771860484702, 
                      0.24739740122504011, -0.38962322566309304, -0.49974207901118639, -1.4515801398271369, 1.0736452978649289, 
                      0.55985898085565033, -0.43789279416506094, 0.48021091037667496, 1.8414133735020126, 1.8695789066643793, 
                      0.56021842531028732, -0.678323243576336, -0.94407219986362523, -0.33987307773274095, -0.71991668517746144, 
                      1.0625139713435376, -1.8026944722350828, 1.8903853852837578, 0.2475468598692494, -0.70834534737086463, 
                      -0.62816536381195498, 0.37297277354517611, 0.034474071621219016, 0.47274333081191594, -2.3662542473841786, 
                      1.8813720711684221, -0.29916037509951754, -0.57712027528715559, 0.27431335749394231, 0.46414272602323764, 
                      -0.61367838919068374, -0.48441048748772131, 0.7807315137448595, 0.5057878952931828, -0.33232362411214894, 
                      -0.77896199497583019, 0.81373804337730904, -1.9957402084896527, 1.7976405059518497, 1.2302892852847949, 
                      0.67699419193473098, -0.51325483082725243, 0.857942641750577, 1.4295866533235857, -0.76819949833721834]
 
        self.nonmatch = [1.417052501689489, -0.043563190732366364, 1.6036891630756054, 0.66248145163751671, 0.052384028443254405, 
                         0.59629061593353161, 0.82947993373378237, 1.115113519426044, 0.67551158941676637, 1.8422107418890203, 
                         0.84941135662024392, 1.1996391852657751, 0.94030154845981673, 2.3269103026602771, -0.030603020790364033, 
                         1.258988565904706, 2.9637747860603456, 1.8173999730963109, 0.71892491243068934, 0.81740037138666277, 
                         1.7601258039962009, 3.1707523951166898, 0.66982205389142613, 1.6097271105344255, 1.189734646321116, 
                         0.22708332837080747, 0.84698202914050347, 1.7635878414797439, 2.3830213681725447, 2.5497367162352944, 
                         2.635862209152271, -0.21290078686666103, 1.4048627271264558, 0.72941226308255924, 0.85692961327062467, 
                         0.97820944194897774, -0.15500601865255503, 0.58435763771835081, 2.5992330339800831, -0.87305656967588074, 
                         0.69311232136547551, 1.1302262899327531, 0.71334154902008384, 0.35695476951005345, -0.5187124559973717, 
                         2.024435812626129, 0.26963199371831936, -0.46510024343728285, -0.19970133295471326, 2.0355468834785726, 
                         0.82313200923780616, 0.30440704254838935, 0.93632925544825862, 1.9575547911114448, 1.2245628328633855, 
                         1.0878755116923233, 2.1602536867629665, 0.04070893565830036, 2.3369676117570961, 1.9724448182299648, 
                         1.9850705023975075, 1.015833476781514, 2.4223167168334743, 0.061707944792565028, 0.94626273945251693, 
                         1.210865335077099, 1.1145727311637936, 2.8519712553054348, 0.93533306721111675, -0.0060786748305075022, 
                         1.9322277720024843, 0.65603343285714444, 1.194849545457592, 0.27772775162736463, 0.078490050192145722, 
                         -1.4721630242727111, 1.854285772101625, 1.6112593328478453, 1.8560106579121847, 2.540591694748537, 
                         1.7772416902829931, -0.20781473501608816, 1.5221307283377219, 0.1579604472392464, -0.30160614059311297, 
                         0.80127729857699337, 1.269704867230514, 2.0490141432761941, 2.0273848755661028, 1.0147875805479856, 
                         -0.06676206771791926, 2.1662293957716994, 2.1413537986988493, 0.9046180315857989, -1.0291168800124986, 
                         1.0301894509766261, 1.1930459134315883, 0.66868219673238327, 0.43346537494032156, -1.0433576612271738]

        
    def testFAR(self):
        roc = pv.ROC(self.match,self.nonmatch,is_distance=True)

        result = roc.getFAR(0.1)
        self.assertAlmostEqual(result.far,0.1)
        self.assertAlmostEqual(result.tar,0.45)

        result = roc.getFAR(0.01)
        self.assertAlmostEqual(result.far,0.01)
        self.assertAlmostEqual(result.tar,0.15)
        
    def testFRR(self):
        roc = pv.ROC(self.match,self.nonmatch,is_distance=True)

        result = roc.getFRR(0.5)
        self.assertAlmostEqual(result.far,0.18)
        self.assertAlmostEqual(result.tar,0.50)

        result = roc.getFRR(0.80)
        self.assertAlmostEqual(result.far,0.04)
        self.assertAlmostEqual(result.tar,0.20)
        
    def testEER(self):
        roc = pv.ROC(self.match,self.nonmatch,is_distance=True)

        eer = roc.getEER()
        self.assertAlmostEqual(eer,0.29)

    def testAUC(self):
        roc = pv.ROC(self.match,self.nonmatch,is_distance=True)
        auc = roc.getAUC()
        self.assertAlmostEqual(auc,0.7608)


    def testPlot(self):
        roc = pv.ROC(self.match,self.nonmatch,is_distance=True)
        plot = pv.Plot()
        roc.plot(plot,method=ROC_PRECISE_SAMPLED,color="red",width=5)
        roc.plot(plot,method=ROC_PRECISE_ALL,color='black')
        plot.lines([[0,1],[1,0]])
        eer = roc.getEER()
        plot.point([eer,1-eer])
        plot.show(delay=100000)
        
        print
        print roc.results()


        
if __name__ == '__main__':
    filenames,matrix = readCsuDistanceMatrix("/Users/bolme/vision/data/celeb_db/distances/PCA_EU")
    print len(filenames),len(matrix),len(matrix[0])
    positive,negative = buildPositiveNegativeLists(filenames,matrix,lambda x,y: x[:10] == y[:10])
    
    print len(positive), len(negative)    
    curve = computeRoc(positive,negative)
    
    positive.sort()
    negative.sort()
    tmp1 = positive[::len(positive)/100]
    tmp2 = negative[::len(negative)/100]
    
    writeRocFile(curve,"/Users/bolme/pca_eu.txt")
    #print curve   
    
    
    
    
    
    
    