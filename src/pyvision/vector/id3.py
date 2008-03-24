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

# TODO: Needs some work.
import math


def lg(x):
    return math.log(x)/math.log(2)

def entropy(labels):
    label_set = set(labels)
    
    # setup variables needed for statistics
    sums = {}
    count = 0.0
    for each in label_set:
        sums[each] = 0.0
        
    for each in labels:
        sums[each] += 1.0
        count += 1.0
        
    ent = 0.0
    for each in sums.values():
        p_i = each/count
        ent -= p_i * lg (p_i)
    return ent

        
        
def maxValue(labels):
    label_set = set(labels)
    
    # setup variables needed for statistics
    sums = {}
    count = 0.0
    for each in label_set:
        sums[each] = 0.0
        
    for each in labels:
        sums[each] += 1.0
        count += 1.0
        
    highVal = 0.0
    highLab = labels[0]
    for key,value in sums.iteritems():
        if value > highVal:
            highVal = value
            highLab = key
    return highLab

def getLabels(features):
    labels = [ each[0] for each in features ]
    return labels
    
    
def splitFeatures(feature,features):
    split = {}
    for label,values in features:
        key = values[feature]
        if not split.has_key(key):
            split[key] = []
        split[key].append([label,values])
        
    return split
  
class ID3:
    
    def __init__(self):
        
        self.training_data = []
        self.testing_data = []
        self.labels = set()
        self.top = None

    def addTraining(self,label,feature):
        '''Training Data'''
        self.training_data.append((label,feature))
        self.labels |= self.labels | set([label])
        
    def addTesting(self,label,feature):
        '''Training Data'''
        self.testing_data.append((label,feature))
        #self.labels |= self.labels | set([label])
        
    def train(self):
        '''Train the classifier on the current data'''
        self.top = Node(self.training_data)
        
    def classify(self,feature):
        '''Classify the feature vector'''
        return self.top.classify(feature)
    
    def test(self, data = None):
        if data == None:
            data = self.testing_data
        #_logger.info("Running test.")
        correct = 0
        wrong = 0
        for label,feature in data:
            c,w = self.classify(feature)
            if c == label:
                correct += 1
            else:
                wrong += 1
        print "Test: %d/%d"%(correct,correct+wrong)   
        return float(correct)/float(correct+wrong)
        


class Node:
    def __init__(self,features):
        
        self.cutoff = 2
        self.min_entropy = 0.2
        
        self.feature = None
        self.entropy = None
        self.label = None #
        self.children = None
        
        self.train(features)
        
    def train(self,features):
        labels = getLabels(features)
        print "Ent:",entropy(labels)
        print "Max:",maxValue(labels)
        
        self.label = maxValue(labels)
        self.entropy = entropy(labels)
        
        if len(features) < self.cutoff or self.entropy < self.min_entropy:
            return
        
        no_feature = len(features[0][1])
        
        max_gain = 0.0
        max_feature = 0
        max_children = {}
        for i in range(no_feature):
            gain = self.entropy
            s = splitFeatures(i,features)
            for key,vals in s.iteritems():
                scale = float(len(vals))/float(len(features))
                e = entropy(getLabels(vals))
                #print "Split %3d:"%i,key,len(vals), e
                gain -= scale*e    
            if max_gain < gain:
                max_gain = gain
                max_feature = i
                max_children = s
        print "Gain: ",max_gain,max_feature
        self.feature = max_feature
        self.gain = max_gain
        
        self.children = {}
        for label,features in max_children.iteritems():
            self.children[label] = Node(features)
              
        #for i in range(features):
    def classify(self,feature):
        '''Classify the feature vector'''
        
        if self.feature:
            val = feature[self.feature]
            if self.children.has_key(val):
                return self.children[val].classify(feature)
        return self.label,None
    
            
        
            
def toBits(val,bits = 4):
    result = []
    for i in range(bits):
        result.append(val&1)
        val = val >> 1
    
    result.reverse()
    
    return result

   
        
if __name__ == "__main__":
    f = open("/Users/bolme/workspace/pyIntel/sample_data/optdigits/optdigits.tra")
    clfy = ID3()
    c = 0
    for line in f:
        vals = line.split(',')

        label =  int(vals[-1])
        
        vals = vals[0:-1] # strip label from back
        feature = []
        for each in vals:
            feature += toBits(int(each))
            
        if c < 3000:
            clfy.addTraining(label,feature)
        else:
            clfy.addTesting(label,feature)
        
        
        #print label, feature
        c += 1
    print clfy.labels
    print len(clfy.training_data)
    print c
    clfy.train()
    clfy.test()
    