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
This module contains functions for reading and writing files
for the Biometrics Evaluation Environment (BEE) including distance
matricies and sigsets.

@authors  David S. Bolme (CSU) and C.J. Carey (NIST)

<p>see: <a href="http://www.bee-biometrics.org">http://www.bee-biometrics.org</a>
'''

import xml.etree.cElementTree as ET
import os.path
import struct
import numpy as np
import pyvision as pv
import pyvision.analysis.roc

BIOMETRIC_SIGNATURE = '{http://www.bee-biometrics.org/schemas/sigset/0.1}biometric-signature'
PRESENTATION = '{http://www.bee-biometrics.org/schemas/sigset/0.1}presentation'

COMPLEX_BIOMETRIC_SIGNATURE = '{http://www.bee-biometrics.org/schemas/sigset/0.1}complex-biometric-signature'
COMPLEX_PRESENTATION = '{http://www.bee-biometrics.org/schemas/sigset/0.1}complex-presentation'
COMPLEX_COMPONENT = '{http://www.bee-biometrics.org/schemas/sigset/0.1}presentation-component'
COMPLEX_DATA = '{http://www.bee-biometrics.org/schemas/sigset/0.1}data'

##
# Parse a BEE sigset.
def parseSigSet(source):
    '''
    the format of a sigset is:
        sigset = [ 
                    ("subject_id", #biometric-signature
                        [ # multiple presentations
                            {'name':"recording_id", 'modality':"...", 'file-name':"...", 'file-format':"..."},
                            {'name':"recording_id", 'modality':"...", 'file-name':"...", 'file-format':"..."},
                            {'name':"recording_id", 'modality':"...", 'file-name':"...", 'file-format':"..."}
                        ]
                    ),
                    ("subject_id",#biometric-signature
                        [ # multiple presentations
                            {'name':"recording_id", 'modality':"...", 'file-name':"...", 'file-format':"..."},
                            {'name':"recording_id", 'modality':"...", 'file-name':"...", 'file-format':"..."},
                            {'name':"recording_id", 'modality':"...", 'file-name':"...", 'file-format':"..."}
                        ]
                    )
                ]   
                        
    
    '''

    sigset = ET.parse(source)
    result = []
    
    # Parse standard biometric signatures without namespaces
    for sig in sigset.findall('biometric-signature'):
        name = sig.get('name')
        signature = []
        result.append( (name,signature) )
        for pres in sig.findall('presentation'):
            presentation = {}
            for key in pres.keys():
                presentation[key] = pres.get(key)
            signature.append(presentation)

    # Parse standard biometric signatures.
    for sig in sigset.findall(BIOMETRIC_SIGNATURE):
        name = sig.get('name')
        signature = []
        result.append( (name, signature ) )
        for pres in sig.findall(PRESENTATION):
            presentation = {}
            for key in pres.keys():
                presentation[key] = pres.get(key)
            signature.append(presentation)

    # Parse complex biometric signatures.
    for sig in sigset.findall(COMPLEX_BIOMETRIC_SIGNATURE):
        name = sig.get('name')
        signature = []
        result.append( (name, signature) )
        for pres in sig.findall(COMPLEX_PRESENTATION):
            presentation = {}
            for key in pres.keys():
                presentation[key] = pres.get(key)
            for comp in pres.findall(COMPLEX_COMPONENT):
                for data in comp.findall(COMPLEX_DATA):
                    for key in data.keys():
                        presentation[key] = data.get(key)
            
            signature.append(presentation)
                
    return result

def sigset2xml(ss):
    root = ET.Element("biometric-signature-set")
    root.text="\n    "
    for signature in ss:
        sig = ET.SubElement(root,"biometric-signature")
        sig.set('name',signature[0])
        sig.text="\n        "
        sig.tail="\n    "
        for presentation in signature[1]:
            pres = ET.SubElement(sig,'presentation')
            for key,value in presentation.iteritems():
                pres.set(key,value)
            pres.tail="\n    "
    tree = ET.ElementTree(root)
    return tree
    
def sigset2array(ss):
    result = []
    for signature in ss:
        sub_id = signature[0]
        if len(signature[1]) != 1:
            raise TypeError("This function only handles simple sigsets.")
        #print signature[1][0]
        
        mode = signature[1][0]['modality']
        format = signature[1][0]['file-format']
        rec_id = signature[1][0]['name']
        filename = signature[1][0]['file-name']
        result.append([sub_id,mode,format,rec_id,filename])
    return result
        
    
def formatSigset(ss,n=None):
    c = 0
    for name,data in ss:
        if c == n:
            break
        print "Name: %s"%name
        for i in range(len(data)):
            print "    Presentation %d" %i
            pres = data[i]
            for key,value in pres.iteritems():
                print "        %-15s : %s"%(key,value) 
        c += 1
               

def fastROC(sorted_positives, sorted_negatives):
    '''
    '''
    
    positives = sorted_positives
    negatives = sorted_negatives
    
    n_pos = len(positives)
    n_neg = len(negatives)
    
    assert len(positives) < len(negatives)
    
    #timer.mark("Starting search sorted")
    indexes = np.searchsorted(negatives,positives)
    #timer.mark("Search time")
    #print "Searched:", len(indexes)
    
    
    tp = (1.0/n_pos) * np.arange(n_pos)
    fn = (1.0/n_neg) * indexes
    
    #timer.mark("ROC computed")
    
    curve = np.array([tp,fn]).transpose()
    
    #print "Curve:",curve.shape
    #print curve

    return curve
               
class BEEDistanceMatrix:

    def __init__(self, *args, **kwargs):
        '''
        Creates a BEE distance matrix
        '''
        if isinstance(args[0],str):
            self.loadFile(*args,**kwargs)
            
        elif isinstance(args[0],np.ndarray):
            self.loadMatrix(*args,**kwargs)
            
        else:
            raise TypeError("Cannot create a BEEDistanceMatrix from an object of type: %s"%type(args[0]))
        
    def loadFile(self,filename,sigset_dir=None):
        '''
        Loads a BEE matrix from a file.
        '''
        self.filename = filename
        self.shortname = os.path.basename(filename)
        
        # open the file for reading
        f = open(filename,'rb')
        
        #read the distance matrix header (first four lines of the file)
        
        #read and process line 1
        line = f.readline().split()
        assert line[0][0] in ['D','S','M']
        assert line[0][1] == '2'
        self.is_distance = True
        if line[0][0] == 'S':
            self.is_distance = False

        # read and process line 2 (target sigset)
        line = f.readline().split()
        self.target_filename = os.path.basename(line[0])

        # read and process line 3 (query sigset)
        line = f.readline().split()
        self.query_filename = os.path.basename(line[0])

        # read and process line 4 (MF n_queries n_targets magic_number)
        line = f.readline().split()
        assert line[0] in ['MF','MB']
        type = line[0][1]
        
        self.n_queries = int(line[1])
        self.n_targets = int(line[2])
        self.magic_number = struct.unpack_from("L",line[3])[0]
        if self.magic_number == 0x12345678:
            byteswap = False
        elif self.magic_number == 0x78563412:
            byteswap = True
        else:
            raise ValueError("Unknown magic number in similarity matrix.")
        
        # Read the matrix data
        if type=='F':
            self.matrix = np.fromfile(f,dtype=np.float32)
        elif type=='B':
            self.matrix = np.fromfile(f,dtype=np.byte)
        else:
            raise TypeError("Unknown matrix type: %s"%type)
        
        if type=='F' and byteswap:
            self.matrix = self.matrix.byteswap()
        assert self.matrix.shape[0] == self.n_targets*self.n_queries
        self.matrix = self.matrix.reshape(self.n_queries,self.n_targets)
        
        # Try to read the sigsets.
        if sigset_dir == None:
            sigset_dir = os.path.dirname(self.filename)
        self.queries = None
        try:
            ss_name = os.path.join(sigset_dir,self.query_filename)
            self.queries = parseSigSet(ss_name)
            assert len(self.queries) == self.n_queries
        except:
            pass
            #print "Warning: cound not read the query sigset for distance matrix %s"%self.shortname
            #print "         SigSet File:",ss_name
            #print "         Expected:",self.n_queries,"Read:",len(self.queries)
        
        self.targets = None
        try:
            ss_name = os.path.join(sigset_dir,self.target_filename)
            self.targets = parseSigSet(ss_name)

            assert len(self.targets) == self.n_targets
        except:
            pass
            #print "Warning: cound not read the target sigset for distance matrix %s"%self.shortname
            #print "         SigSet File:",ss_name
            #print "         Expected:",self.n_targets,"Read:",len(self.targets)
        
        
    def loadMatrix(self, mat, query_filename, target_filename, sigset_dir=None, is_distance=True):
        '''
        Creates a bee matrix from a numpy array.
        '''
        #read the distance matrix header (first four lines of the file)
        # select distance or similarity
        self.is_distance = is_distance

        # read and process line 2 (target sigset)
        self.target_filename = target_filename

        # read and process line 3 (query sigset)
        self.query_filename = query_filename

        # read and process line 4 (MF n_queries n_targets magic_number)        
        self.n_queries = mat.shape[0]
        self.n_targets = mat.shape[1]
        self.magic_number = 0x12345678
        
        # Read the matrix data
        self.matrix = mat
            
        # Try to read the sigsets.
        self.queries = None
        self.targets = None
        if sigset_dir != None:
            try:
                ss_name = os.path.join(sigset_dir,self.query_filename)
                self.queries = parseSigSet(ss_name)
                assert len(self.queries) == self.n_queries
            except:
                print "Warning: cound not read the query sigset for distance matrix"
                print "         SigSet File:",ss_name
                print "         Expected:",self.n_queries,"Read:",len(self.queries)
        
            try:
                ss_name = os.path.join(sigset_dir,self.target_filename)
                self.targets = parseSigSet(ss_name)
    
                assert len(self.targets) == self.n_targets
            except:
                print "Warning: cound not read the target sigset for distance matrix"
                print "         SigSet File:",ss_name
                print "         Expected:",self.n_targets,"Read:",len(self.targets)
        
     
    def znorm(self):
        for i in range(self.matrix.shape[0]):
            a = self.matrix[i,:]
            mn = a.mean()
            sd = a.std()
            self.matrix[i,:] = (self.matrix[i,:]-mn)/sd
            
            
    def getMatchScores(self,mask=None):
        #assert self.queries != None
        #assert self.targets != None
        
        matches = []
        if self.queries != None and self.targets != None:
            queries = np.array([ name for name,sig in self.queries ])
            targets = np.array([ name for name,sig in self.targets ])
        for i in range(self.matrix.shape[0]):
            #print i, len(matches)
            if mask != None:
                matches.append(self.matrix[i,mask.matrix[i,:] == -1])
            else:
                query = queries[i]
                matches.append(self.matrix[i,query==targets])
        total = 0
        for each in matches:
            total += len(each)

        scores = np.zeros(shape=(total),dtype=np.float32)
        i = 0
        for each in matches:
            s = len(each)
            scores[i:i+s] = each
            i += s
        return scores
    
    
    def getMatchScoresBySubject(self,mask=None):
        assert self.queries != None
        assert self.targets != None
        
        matches = {}
        queries = np.array([ name for name,sig in self.queries ])
        targets = np.array([ name for name,sig in self.targets ])
        
        qnames = set(queries)
        tnames = set(targets)
        
        for name in qnames:
            rows = np.nonzero(name == queries)[0]
            cols = np.nonzero(name == targets)[0]
            tmp =  self.matrix[rows][:,cols]
            if mask != None:
                m = mask.matrix[rows][:,cols] != 0x00
                matches[name] = tmp.flatten()[m.flatten()]
            else:
                matches[name] = tmp.flatten()
            
            if len(matches[name]) == 0:
                del matches[name]
                
        return matches
            
    
    def getNonMatchScores(self,mask=None):
        #assert self.queries != None
        #assert self.targets != None
        
        matches = []
        if self.queries != None and self.targets != None:
            queries = np.array([ name for name,sig in self.queries ])
            targets = np.array([ name for name,sig in self.targets ])
        for i in range(self.matrix.shape[0]):
            if mask != None:
                matches.append(self.matrix[i,mask.matrix[i,:] == 127])
            else:
                query = queries[i]
                matches.append(self.matrix[i,query!=targets])
        total = 0
        for each in matches:
            total += len(each)

        scores = np.zeros(shape=(total),dtype=np.float32)
        i = 0
        for each in matches:
            s = len(each)
            scores[i:i+s] = each
            i += s
        return scores


    def getNonMatchScoresByPairs(self,mask=None):
        assert self.queries != None
        assert self.targets != None
        
        matches = {}
        queries = np.array([ name for name,sig in self.queries ])
        targets = np.array([ name for name,sig in self.targets ])
        
        rows = queries.argsort()
        cols = targets.argsort()
        #print rows
        #print cols
        qnames = list(set(queries))
        tnames = list(set(targets))
        
        matrix = (self.matrix[rows,:][:,cols])
        if mask != None:
            mask = (mask.matrix[rows,:][:,cols])
        queries = queries[rows]
        targets = targets[cols]

        q_blocks = {}
        for qname in qnames:
            rows = np.nonzero(qname == queries)[0]
            q_blocks[qname] = (rows[0],rows[-1]+1)
        
        t_blocks = {}
        for tname in tnames:
            cols = np.nonzero(tname == targets)[0]
            t_blocks[tname] = (cols[0],(cols[-1]+1))
        
        
        total = len(qnames)*len(tnames)
        
        i = 0
        for qname in qnames:
            matches[qname]= {}
            for tname in tnames:
                
                if qname == tname:
                    continue

                r1,r2 = q_blocks[qname]
                c1,c2 = t_blocks[tname]

                tmp = matrix[r1:r2,c1:c2]
                if mask != None:
                    m = mask[r1:r2,c1:c2] != 0x00
                    matches[qname][tname] = tmp.flatten()[m.flatten()]
                else:
                    matches[qname][tname] = tmp.flatten()
                
                if len(matches[qname][tname]) == 0:
                    del matches[qname][tname]
            
        return matches
            
    
    def printInfo(self):
        print "BEEDistanceMatrix:",self.filename
        print "    is_distance     :",self.is_distance
        print "    target_filename :",self.target_filename
        print "    query_filename  :",self.query_filename
        print "    n_queries       :",self.n_queries
        print "    n_targets       :",self.n_targets
        print "    <total size>    :",self.n_targets*self.n_queries
        print "    magic_number    : %x"%self.magic_number
        print "    matrix.shape    :",self.matrix.shape
        #print "    <matrix sample> :",self.matrix[0,0:4]
        #print "    <matrix sample> :",self.matrix[1,0:4]
        #print "    <matrix sample> :",self.matrix[2,0:4]
        #print "    <matrix sample> :",self.matrix[3,0:4]
        
    def write(self,filename):
        self.save(filename)
  
    def save(self,filename):
        '''
        Writes the BEE distance matrix to file. WARNING: DOES NOT HANDLE MASK MATRICES CORRECTLY!
        '''
        #maybe check for overwrite? and add param for allowing overwrite
        file = open(filename, "w")
        # write line 1 : type and version
        if self.is_distance:
            file.write('D') #doesn't handle mask matrices!
        else:
            file.write('S') #doesn't handle mask matrices!
            
        file.write("2\n")
        # write lines 2 and 3 (target and query sigsets)
        file.writelines([self.target_filename+"\n", self.query_filename+"\n"])
        # write line 4 (MF n_queries n_targets magic_number)
        file.write("MF %d %d %s\n" %(self.n_queries, self.n_targets, struct.pack("L", self.magic_number)))
        # write the data
        file.write(self.matrix)
        file.close()

    def histogram(self,value_range=None,bins=100,type="ALL",normed=False,mask=None):
        if type == "ALL":
            scores = self.matrix
        elif type == 'MATCH':
            scores = self.getMatchScores(mask=mask)
        elif type == 'NONMATCH':
            scores = self.getNonMatchScores(mask=mask)
        else:
            raise ValueError("Histogram of type %s is not supported use 'ALL', 'MATCH', or 'NONMATCH'.")

        if value_range == None:
            value_range = (self.matrix.min(),self.matrix.max())

        counts,vals = np.histogram(scores,range=value_range,bins=bins,normed=normed)
                
        hist = pv.Table()
        for i in range(len(counts)):
            hist[i,'min'] = vals[i]
            hist[i,'center'] = 0.5*(vals[i]+vals[i+1])
            hist[i,'max'] = vals[i+1]
            hist[i,'count'] = counts[i]
        return hist
    
    
    def getROC(self,mask=None):
        nonmatch = self.getNonMatchScores(mask=mask)
        match = self.getMatchScores(mask=mask)
        return pv.analysis.roc.ROC(match,nonmatch,is_distance=self.is_distance)
        


    def stats(self):
        table = pv.Table()
        table['Mean','Value'] = self.matrix.mean()
        # not computed effecently: table['Std','Value'] = self.matrix.flatten().std()
        table['Min','Value'] = self.matrix.min()
        table['Max','Value'] = self.matrix.max()
        return table

            
    def __str__(self):
        '''
        Returns a string describing the matrix.
        '''
        type = {True:"Distance",False:"Similarity"}[self.is_distance]
        return "BEE[file=%s;type=%s]"%(self.shortname,type)
    
