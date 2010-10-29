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

from pyvision.analysis import Table
from pyvision.analysis.roc import ROC
from pyvision.analysis.stats import cibinom


def firstFive(im_id):
    ''' 
    Parses the identity from the image id using the first five 
    charicters.  This is a very common scheme for many face datasets.
    
    This will work for: FERET or CSU_Scraps
    '''
    return im_id[:5]
    
    
# SCORE_LOW is for distance like measures where low scores 
# indicate a better match
SCORE_TYPE_LOW  = 0
# SCORE_HIGH is for similarity like measures where high scores 
# indicate a better match
SCORE_TYPE_HIGH = 1

class FaceRecognitionTest:
    
    def __init__(self, name=None, id_func=firstFive, score_type=SCORE_TYPE_LOW):
        '''
        Create a face recognition test object.
        '''
        self.name = name
        self.id_func = id_func
        self.score_type = score_type
        
        self.probes_table   = Table.Table()
        self.nonmatch_table = Table.Table()
        self.rank_table    = Table.Table()
        
        self.total_probes = 0
        self.total_gallery = 0
        self.rank1_success = 0
        
        self.positives = []
        self.negatives = []
        
    def getROCAnalysis(self):
        return ROC(self.positives,self.negatives)
    
    def addSample(self,probe_id,scores):
        '''
        Adds a sample to the test.  The similarity scores is a list of 
        tuples that contain tuples of (gallery_id, score) 
        '''
        if self.score_type == SCORE_TYPE_LOW:
            scores.sort( lambda x,y: cmp(x[1],y[1]) )    
        elif self.score_type == SCORE_TYPE_HIGH:        
            scores.sort( lambda x,y: -cmp(x[1],y[1]) )
        else:
            raise ValueError("Unknown score type: %s"%self.score_type)
        
        best_match_id       = None
        best_match_score    = None
        best_match_rank     = None
        best_nonmatch_id    = None
        best_nonmatch_score = None
        best_nonmatch_rank  = None
        
        pid = self.id_func(probe_id)
        rank = 0
        for gallery_id,score in scores:
            if probe_id == gallery_id:
                # Probe and gallery should not be the same image
                continue
            gid = self.id_func(gallery_id)
            match = False
            if pid == gid:
                match = True
            
            if match and best_match_id == None:
                best_match_id       = gallery_id
                best_match_score    = score
                best_match_rank     = rank
            
            if not match and best_nonmatch_id == None:
                best_nonmatch_id    = gallery_id
                best_nonmatch_score = score
                best_nonmatch_rank  = rank
            
            if match:
                self.positives.append(score)
            else:
                self.negatives.append(score)
                
            rank += 1
            
        self.total_probes  += 1
        self.total_gallery = max(self.total_gallery,rank)

        success = False
        if best_match_rank==0:
            success = True
            self.rank1_success += 1
                
        self.probes_table.setElement(probe_id,'MatchId',best_match_id)
        self.probes_table.setElement(probe_id,'MatchScore',best_match_score)
        self.probes_table.setElement(probe_id,'MatchRank',best_match_rank)
        self.probes_table.setElement(probe_id,'Success',best_match_rank==0)
        self.probes_table.setElement(probe_id,'NonMatchId',best_nonmatch_id)
        self.probes_table.setElement(probe_id,'NonMatchScore',best_nonmatch_score)
        self.probes_table.setElement(probe_id,'NonMatchRank',best_nonmatch_rank)
            
        self.rank1_rate = float(self.rank1_success)/float(self.total_probes)
        self.rank1_bounds = cibinom(self.total_probes,self.rank1_success)

            
            
            
            
            
            
            