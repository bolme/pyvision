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

@authors:  David S. Bolme (CSU) and C.J. Carey (NIST)

see: <a href="http://www.bee-biometrics.org">http://www.bee-biometrics.org</a>
'''

import xml.etree.cElementTree as ET
import os.path
import struct
import binascii
import numpy as np
import scipy as sp
import scipy.io as spio
import pyvision as pv
import pyvision.analysis.roc as roc
import gzip

BIOMETRIC_SIGNATURE = '{http://www.bee-biometrics.org/schemas/sigset/0.1}biometric-signature'
PRESENTATION = '{http://www.bee-biometrics.org/schemas/sigset/0.1}presentation'

COMPLEX_BIOMETRIC_SIGNATURE = '{http://www.bee-biometrics.org/schemas/sigset/0.1}complex-biometric-signature'
COMPLEX_PRESENTATION = '{http://www.bee-biometrics.org/schemas/sigset/0.1}complex-presentation'
COMPLEX_COMPONENT = '{http://www.bee-biometrics.org/schemas/sigset/0.1}presentation-component'
COMPLEX_DATA = '{http://www.bee-biometrics.org/schemas/sigset/0.1}data'

BEE_NONMATCH = 0x7f
BEE_MATCH    = -1 #0xff
BEE_DONTCARE = 0x00

BEE_CODE_MAP = {
                0x7f:"NONMATCH",
                0xff:"MATCH",
                -1:"MATCH",
                0x00:"DONTCARE",             
                }

##
# Parse a BEE sigset.
def parseSigSet(filename):
    '''
    the format of a sigset is::
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
    if isinstance(filename,str) and filename.endswith('.gz'):
        # assume the file is compressed
        filename = gzip.open(filename,'rb')

    sigset = ET.parse(filename)
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

def saveSigset(ss,filename):
    '''
    save a sigset to a file.
    
    @param ss: a sigset structured list
    @param filename: a file object or filename
    '''
    if isinstance(filename,str) and filename.endswith('.gz'):
        # assume the file should be compressed
        filename = gzip.open(filename,'wb')
        
    xmlss = sigset2xml(ss)
    xmlss.write(filename)

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
        
        line = f.readline()
        # Test line endings
        if len(line) != 3 or line[-1] != "\x0a":
            # Note: \x0a is the "official" line ending char as of 
            #       \x0d is also supported in the Java and C++ tools but it will cause a failure in this implementation.
            #       see IARPA BEST - Challenge Problem Specification and Executable Application Program Interface
            #       thanks to Todd Scruggs
            raise ValueError("Unsupported line ending.  Should two characters followed by LF (0x0A).")
        # Check Format
        line = line.strip()
        if line not in ['D2','S2','M2']:
            raise ValueError('Unknown matrix Format "%s".  Should be D2, S2, or M2.'%line)
        
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
        
        big_endian = struct.pack(">I",0x12345678)
        little_endian = struct.pack("<I",0x12345678)
        
        if line[3] != big_endian and line[3] != little_endian:
            print "Warning unsupported magic number is BEE matrix: 0x%s"%binascii.hexlify(line[3])
            
        self.magic_number = struct.unpack_from("=I",line[3])[0]
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
        self.shortname=None
        
        #read the distance matrix header (first four lines of the file)
        if mat.dtype != np.byte:    
            mat = mat.astype(np.float32)
            
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
        
     
    def cohort_norm(self):
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
            queries = np.array([ name for name,_ in self.queries ])
            targets = np.array([ name for name,_ in self.targets ])
        for i in range(self.matrix.shape[0]):
            #print i, len(matches)
            if mask != None:
                matches.append(self.matrix[i,mask.matrix[i,:] == BEE_MATCH])
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
        queries = np.array([ name for name,_ in self.queries ])
        targets = np.array([ name for name,_ in self.targets ])
        
        qnames = set(queries)
        #tnames = set(targets)
        
        for name in qnames:
            rows = np.nonzero(name == queries)[0]
            cols = np.nonzero(name == targets)[0]
            tmp =  self.matrix[rows][:,cols]
            if mask != None:
                m = mask.matrix[rows][:,cols] == BEE_MATCH
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
            queries = np.array([ name for name,_ in self.queries ])
            targets = np.array([ name for name,_ in self.targets ])
        for i in range(self.matrix.shape[0]):
            if mask != None:
                matches.append(self.matrix[i,mask.matrix[i,:] == BEE_NONMATCH])
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

    def asFlatArray(self,mask=None):
        '''query,target,score,type'''
        r,c = self.matrix.shape
        result = np.zeros((r*c,4),dtype=np.object)
        for i in range(r):
            for j in range(c):
                result[c*i+j,0] = i 
                result[c*i+j,1] = j
                result[c*i+j,2] = self.matrix[i,j]
                if BEE_CODE_MAP.has_key(mask[i,j]):
                    result[c*i+j,3] = BEE_CODE_MAP[mask[i,j]]
                else:
                    result[c*i+j,3] = "0x%02x"%mask[i,j]
        return result
            
        

            
    
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
        
    def write(self,filename):
        self.save(filename)
  
    def save(self,filename):
        '''
        Writes the BEE distance matrix to file. WARNING: DOES NOT HANDLE MASK MATRICES CORRECTLY!
        '''
        if filename.endswith('.mtx'):
            # save a BEE formated matrix
            self.saveBeeFormat(filename)
        elif filename.endswith('.mat'):
            # save a matlab formated matrix
            if self.is_distance:
                matrix_name = 'dist_matrix'
            else:
                matrix_name = 'sim_matrix'
            spio.savemat(filename, {matrix_name:self.matrix})
        else:
            return NotImplementedError("Unsupported matrix format for filename %s"%filename)
        
    def saveBeeFormat(self,filename):
        #maybe check for overwrite? and add param for allowing overwrite
        file = open(filename, "wb")
        
        # write line 1 : type and version
        type = 'D'
        if self.matrix.dtype == np.byte:    
            type = 'M'
        elif self.is_distance:
            type = 'D'
        else:
            type = 'S'
            
        file.write(type)
        file.write("2\x0a")
        
        # write lines 2 and 3 (target and query sigsets)
        file.write(self.target_filename+"\x0a")
        file.write(self.query_filename+"\x0a")
        
        # write line 4 (MF n_queries n_targets magic_number)
        magic_number = struct.pack('=I',0x12345678)
        assert len(magic_number) == 4 # Bug fix: verify the magic number is really 4 bytes
        if type == 'M':
            file.write("MB %d %d %s\x0a" %(self.n_queries, self.n_targets, magic_number))
        else:
            file.write("MF %d %d %s\x0a" %(self.n_queries, self.n_targets, magic_number))
        
        # write the data
        file.write(self.matrix)
        file.close()

    def histogram(self,value_range=None,bins=100,normed=True,mask=None):
        match_scores = self.getMatchScores(mask=mask)
        nonmatch_scores = self.getNonMatchScores(mask=mask)
        if value_range == None:
            value_range = (self.matrix.min(),self.matrix.max())

        match_counts,vals = np.histogram(match_scores,range=value_range,bins=bins,normed=normed)
        nonmatch_counts,vals = np.histogram(nonmatch_scores,range=value_range,bins=bins,normed=normed)
                       
        hist = pv.Table()
        for i in range(len(match_counts)):
            hist[i,'min'] = vals[i]
            hist[i,'center'] = 0.5*(vals[i]+vals[i+1])
            hist[i,'max'] = vals[i+1]
            hist[i,'match_count'] = match_counts[i]
            hist[i,'nonmatch_count'] = nonmatch_counts[i]
        return hist
    
    
    def getROC(self,mask=None):
        nonmatch = self.getNonMatchScores(mask=mask)
        match = self.getMatchScores(mask=mask)
        return roc.ROC(match,nonmatch,is_distance=self.is_distance)
    
    def getRank1(self,mask=None):
        rows,cols = self.matrix.shape
        
        queries = np.array([ name for name,sig in self.queries ])
        targets = np.array([ name for name,sig in self.targets ])

        success = 0.0
        count = 0.0
        for i in range(rows):
            row = self.matrix[i]
            if self.is_distance:
                j = row.argmin()
            else:
                j = row.argmax()
            if queries[i] == targets[j]:
                success += 1
            count += 1
        
        #print success, count, success/count
        return success/count
        


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
    
    def __getitem__(self,index):
        '''An accessor to quickly read matrix data'''
        return self.matrix.__getitem__(index)
    
    def shape(self):
        '''@returns: the number of rows and columns.'''
        return self.matrix.shape
    

def computeMaskMatrix(target_sigset,query_sigset,target_filename,query_filename,symmetric = True):
    '''
    Computes a mask matrix from two sigsets.
    
    @param target_sigset: the target sigset to use.
    @param query_sigset: the query sigset to use.
    @param symmetric: if true and the sigsets are equal it assumes that the matrix is symmetric and will treat the low left triangle as DONT_CARE's.
    @returns: a bee mask matrix.
    '''
    assert len(target_sigset) > 0
    assert len(query_sigset) > 0
    target_subid = np.array([each[0] for each in target_sigset])  
    query_subid = np.array([each[0] for each in query_sigset])
    target_recid = np.array([each[1][0]['name'] for each in target_sigset])  
    query_recid = np.array([each[1][0]['name'] for each in query_sigset])

    cols = target_subid.shape[0]
    rows = query_subid.shape[0]
    
    target_subid.shape = (1,cols)
    query_subid.shape = (rows,1)
    target_recid.shape = (1,cols)
    query_recid.shape = (rows,1)
    
    # Initialize matrix to non match
    mat = np.zeros((rows,cols),dtype=np.byte)
    mat[:,:] = pv.BEE_NONMATCH
    
    # Set matches to match
    matches = target_subid == query_subid
    mat[matches] = pv.BEE_MATCH
    
    # Set duplicates to don't care.
    duplicates = target_recid == query_recid
    mat[duplicates] = pv.BEE_DONTCARE
    
    # Check for symetric matrix
    if symmetric and rows == cols:
        ts = target_recid.flatten()
        qs = query_recid.flatten()
        if (ts == qs).sum() == rows:
            # Exclude the lower triangle
            r = np.arange(rows)
            c = np.arange(cols)
            r.shape = (rows,1)
            c.shape = (1,cols)
            tmp = r > c
            mat[tmp] = pv.BEE_DONTCARE
    
    return pv.BEEDistanceMatrix(mat, query_filename, target_filename)




