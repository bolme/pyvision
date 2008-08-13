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
import time
import math

from pyvision.types.Rect import Rect,BoundingRect
from pyvision.types.Image import Image
from pyvision.analysis.Table import Table
from pyvision.analysis.stats import cibinom
from pyvision.analysis.FaceAnalysis.FaceDetectionTest import face_from_eyes,is_success

class EyeDetectionTest:
    '''
    BUGFIX: 20080813 Bailey Draper found a bug that the field dl in the full 
    report was really dl^2.
    '''
    def __init__(self,name=None,threshold=0.25):
        ''''''
        self.name = name
        self.table = Table()
        self.summary_table = Table()
        self.face_successes = 0
        self.both25_successes = 0
        self.left25_successes = 0
        self.right25_successes = 0
        self.both10_successes = 0
        self.left10_successes = 0
        self.right10_successes = 0
        self.both05_successes = 0
        self.left05_successes = 0
        self.right05_successes = 0
        self.bothsse = 0.0
        self.rightsse = 0.0
        self.leftsse = 0.0
        self.pixels = 0
        self.images = 0
        self.faces = 0
        self.start_time = time.time()
        self.stop_time = None
        
    def addSample(self, truth_eyes, detected_eyes, im=None, annotate=False):
        ''''''
        self.images += 1
        
        if isinstance(im,Image):
            name = im.filename
            if self.pixels != None:
                self.pixels += im.asPIL().size[0] * im.asPIL().size[1]
        elif isinstance(im,str):
            name = im
            self.pixels = None
        else:
            name = "%d"%self.sample_id
            self.pixels = None
            
        self.stop_time = time.time()

        for tl,tr in truth_eyes:
            tface = face_from_eyes(tl,tr)

            detect_face = False
            eye_dist = None
            detect_b25  = False
            detect_b10  = False
            detect_b05  = False
            detect_l25  = False
            detect_l10  = False
            detect_l05  = False
            detect_r25  = False
            detect_r10  = False
            detect_r05  = False
            eye_dist = None
            tl_x  = None
            tl_y  = None
            tr_x  = None
            tr_y  = None
            pl_x  = None
            pl_y  = None
            pr_x  = None
            pr_y  = None
            dlx   = None
            dly   = None
            dl2   = None
            dl    = None
            dlfrac= None
            drx   = None
            dry   = None
            dr2   = None
            dr    = None
            drfrac= None
            deye  = None
            dmean = None
            
            for pl,pr in detected_eyes:
                dface = face_from_eyes(pl,pr)
                
                if is_success(tface,dface):
                    tl_x = tl.X()
                    tl_y = tl.Y()
                    tr_x = tr.X()
                    tr_y = tr.Y()
                    eye_dist = math.sqrt((tl_x-tr_x)*(tl_x-tr_x) + (tl_y-tr_y)*(tl_y-tr_y))
                    pl_x = pl.X()
                    pl_y = pl.Y()
                    pr_x = pr.X()
                    pr_y = pr.Y()
                    
                    detect_face = True
                    
                    eye_dist = math.sqrt((tl_x-tr_x)*(tl_x-tr_x) + (tl_y-tr_y)*(tl_y-tr_y))
                    
                    dlx = pl_x-tl_x
                    dly = pl_y-tl_y
                    dl2 = dlx*dlx + dly*dly
                    dl = math.sqrt(dl2)
                    dlfrac = dl/eye_dist
                    
                    drx = pr_x-tr_x
                    dry = pr_y-tr_y
                    dr2 = drx*drx + dry*dry
                    dr = math.sqrt(dr2)
                    drfrac = dr/eye_dist
                    
                    deye = max(drfrac,dlfrac)
                    
                    dmean = 0.5*(dr+dl)
                    
                    detect_l25  = 0.25 > dlfrac
                    detect_l10  = 0.10 > dlfrac
                    detect_l05  = 0.05 > dlfrac
                    detect_r25  = 0.25 > drfrac
                    detect_r10  = 0.10 > drfrac
                    detect_r05  = 0.05 > drfrac
                    detect_b25  = 0.25 > deye
                    detect_b10  = 0.10 > deye
                    detect_b05  = 0.05 > deye

                    break
                            
            self.table.setElement(self.faces,'name',name)              
            self.table.setElement(self.faces,'detect_face',detect_face)              
            self.table.setElement(self.faces,'detect_l25',detect_l25)              
            self.table.setElement(self.faces,'detect_l10',detect_l10)              
            self.table.setElement(self.faces,'detect_l05',detect_l05)              
            self.table.setElement(self.faces,'detect_r25',detect_r25)              
            self.table.setElement(self.faces,'detect_r10',detect_r10)              
            self.table.setElement(self.faces,'detect_r05',detect_r05)              
            self.table.setElement(self.faces,'detect_b25',detect_b25)              
            self.table.setElement(self.faces,'detect_b10',detect_b10)              
            self.table.setElement(self.faces,'detect_b05',detect_b05)              
            self.table.setElement(self.faces,'eye_dist',eye_dist)
                          
            self.table.setElement(self.faces,'truth_lx',tl_x)              
            self.table.setElement(self.faces,'truth_ly',tl_y)              
            self.table.setElement(self.faces,'truth_rx',tr_x)              
            self.table.setElement(self.faces,'truth_ry',tr_y)              
            
            self.table.setElement(self.faces,'pred_lx',pl_x)              
            self.table.setElement(self.faces,'pred_ly',pl_y)              
            self.table.setElement(self.faces,'pred_rx',pr_x)              
            self.table.setElement(self.faces,'pred_ry',pr_y)              
            
            self.table.setElement(self.faces,'dlx',dlx)              
            self.table.setElement(self.faces,'dly',dly)              
            #self.table.setElement(self.faces,'dl2',dl2)              
            self.table.setElement(self.faces,'dl',dl) # BUGFIX: 20080813 This was outputing dl2.             
            self.table.setElement(self.faces,'dlfrac',dlfrac)              
            self.table.setElement(self.faces,'drx',drx)              
            self.table.setElement(self.faces,'dry',dry)              
            #self.table.setElement(self.faces,'dr2',dr2)              
            self.table.setElement(self.faces,'dr',dr)              
            self.table.setElement(self.faces,'drfrac',drfrac)              
            self.table.setElement(self.faces,'deye',deye)              
            self.table.setElement(self.faces,'dmean',dmean) 
                         
            self.faces += 1
            if dlfrac != None:
                self.bothsse += dlfrac**2 + drfrac**2
                self.leftsse += dlfrac**2
                self.rightsse += drfrac**2
            
            if detect_face: self.face_successes    += 1
            if detect_b25:  self.both25_successes  += 1
            if detect_l25:  self.left25_successes  += 1
            if detect_r25:  self.right25_successes += 1
            if detect_b10:  self.both10_successes  += 1
            if detect_l10:  self.left10_successes  += 1
            if detect_r10:  self.right10_successes += 1
            if detect_b05:  self.both05_successes  += 1
            if detect_l05:  self.left05_successes  += 1
            if detect_r05:  self.right05_successes += 1
            

    def finish(self):
            self.face_rate    = float(self.face_successes)/self.faces
            self.face_ci      = cibinom(self.faces,self.face_successes,alpha=0.05)
            self.both25_rate  = float(self.both25_successes)/self.faces
            self.both25_ci    = cibinom(self.faces,self.both25_successes,alpha=0.05)
            self.both10_rate  = float(self.both10_successes)/self.faces
            self.both10_ci    = cibinom(self.faces,self.both10_successes,alpha=0.05)
            self.both05_rate  = float(self.both05_successes)/self.faces
            self.both05_ci    = cibinom(self.faces,self.both05_successes,alpha=0.05)
            self.left25_rate  = float(self.left25_successes)/self.faces
            self.left25_ci    = cibinom(self.faces,self.left25_successes,alpha=0.05)
            self.left10_rate  = float(self.left10_successes)/self.faces
            self.left10_ci    = cibinom(self.faces,self.left10_successes,alpha=0.05)
            self.left05_rate  = float(self.left05_successes)/self.faces
            self.left05_ci    = cibinom(self.faces,self.left05_successes,alpha=0.05)
            self.right25_rate  = float(self.right25_successes)/self.faces
            self.right25_ci    = cibinom(self.faces,self.right25_successes,alpha=0.05)
            self.right10_rate  = float(self.right10_successes)/self.faces
            self.right10_ci    = cibinom(self.faces,self.right10_successes,alpha=0.05)
            self.right05_rate  = float(self.right05_successes)/self.faces
            self.right05_ci    = cibinom(self.faces,self.right05_successes,alpha=0.05)
            if self.face_successes > 0:
                self.bothrmse = math.sqrt(self.bothsse/(2*self.face_successes))
                self.leftrmse = math.sqrt(self.leftsse/self.face_successes)
                self.rightrmse = math.sqrt(self.rightsse/self.face_successes)
            self.elapse_time  = self.stop_time - self.start_time
            self.time_per_image = self.elapse_time / self.images
            self.time_per_face  = self.elapse_time / self.faces
                

        
    def createSummary(self): 
        ''''''   
        self.finish()
        self.summary_table.setElement('FaceRate','Estimate',self.face_rate)  
        self.summary_table.setElement('FaceRate','Lower95',self.face_ci[0])  
        self.summary_table.setElement('FaceRate','Upper95',self.face_ci[1])  
        self.summary_table.setElement('Both25Rate','Estimate',self.both25_rate)  
        self.summary_table.setElement('Both25Rate','Lower95',self.both25_ci[0])  
        self.summary_table.setElement('Both25Rate','Upper95',self.both25_ci[1])  
        self.summary_table.setElement('Both10Rate','Estimate',self.both10_rate)  
        self.summary_table.setElement('Both10Rate','Lower95',self.both10_ci[0])  
        self.summary_table.setElement('Both10Rate','Upper95',self.both10_ci[1])  
        self.summary_table.setElement('Both05Rate','Estimate',self.both05_rate)  
        self.summary_table.setElement('Both05Rate','Lower95',self.both05_ci[0])  
        self.summary_table.setElement('Both05Rate','Upper95',self.both05_ci[1])  
        self.summary_table.setElement('BothRMSE','Estimate',self.bothrmse)
        self.summary_table.setElement('Left25Rate','Estimate',self.left25_rate)  
        self.summary_table.setElement('Left25Rate','Lower95',self.left25_ci[0])  
        self.summary_table.setElement('Left25Rate','Upper95',self.left25_ci[1])  
        self.summary_table.setElement('Left10Rate','Estimate',self.left10_rate)  
        self.summary_table.setElement('Left10Rate','Lower95',self.left10_ci[0])  
        self.summary_table.setElement('Left10Rate','Upper95',self.left10_ci[1])  
        self.summary_table.setElement('Left05Rate','Estimate',self.left05_rate)  
        self.summary_table.setElement('Left05Rate','Lower95',self.left05_ci[0])  
        self.summary_table.setElement('Left05Rate','Upper95',self.left05_ci[1])  
        self.summary_table.setElement('LeftRMSE','Estimate',self.leftrmse)
        self.summary_table.setElement('Right25Rate','Estimate',self.right25_rate)  
        self.summary_table.setElement('Right25Rate','Lower95',self.right25_ci[0])  
        self.summary_table.setElement('Right25Rate','Upper95',self.right25_ci[1])  
        self.summary_table.setElement('Right10Rate','Estimate',self.right10_rate)  
        self.summary_table.setElement('Right10Rate','Lower95',self.right10_ci[0])  
        self.summary_table.setElement('Right10Rate','Upper95',self.right10_ci[1])  
        self.summary_table.setElement('Right05Rate','Estimate',self.right05_rate)  
        self.summary_table.setElement('Right05Rate','Lower95',self.right05_ci[0])  
        self.summary_table.setElement('Right05Rate','Upper95',self.right05_ci[1])
        self.summary_table.setElement('RightRMSE','Estimate',self.rightrmse)
        self.summary_table.setElement('ElapsedTime','Estimate',self.elapse_time)
        self.summary_table.setElement('ImageTime','Estimate',self.time_per_image)
        self.summary_table.setElement('FaceTime','Estimate',self.time_per_face)
        self.summary_table.setElement('ImageCount','Estimate',self.images)
        self.summary_table.setElement('FaceCount','Estimate',self.faces)
        return self.summary_table  

    def __str__(self):
        ''' One line summary of the test '''
        return "EyeDetectionTest(name:%s,FaceRate:%0.4f,Both25Rate:%0.4f,Both10Rate:%0.4f,Both05Rate:%0.4f,NFaces:%d,Time:%0.2f)"%(self.name,self.face_rate,self.both25_rate,self.both10_rate,self.both05_rate,self.faces,self.elapse_time)

#############################################################################
def summarizeEyeDetectionTests(tests):
    '''
    Create a summary table for a list containing FaceDetectionTest objects.
    '''
    print 'Creating summaries...'
    summary25 = Table()
    summary25.setColumnFormat('Face_Rate','%0.4f')
    summary25.setColumnFormat('Rate_25','%0.4f')
    summary25.setColumnFormat('Lower95_25','%0.4f')
    summary25.setColumnFormat('Upper95_25','%0.4f')
    summary25.setColumnFormat('Time','%0.2f')
    for test in tests:
        print test.name
        summary25.setElement(test.name,'Face_Rate',test.face_rate)
        summary25.setElement(test.name,'Rate_25',test.both25_rate)
        summary25.setElement(test.name,'Lower95_25',test.both25_ci[0])
        summary25.setElement(test.name,'Upper95_25',test.both25_ci[1])
        summary25.setElement(test.name,'Time',test.elapse_time)
        
        
    summary10 = Table()
    summary10.setColumnFormat('Face_Rate','%0.4f')
    summary10.setColumnFormat('Rate_10','%0.4f')
    summary10.setColumnFormat('Lower95_10','%0.4f')
    summary10.setColumnFormat('Upper95_10','%0.4f')
    summary10.setColumnFormat('Time','%0.2f')
    for test in tests:
        summary10.setElement(test.name,'Face_Rate',test.face_rate)
        summary10.setElement(test.name,'Rate_10',test.both10_rate)
        summary10.setElement(test.name,'Lower95_10',test.both10_ci[0])
        summary10.setElement(test.name,'Upper95_10',test.both10_ci[1])
        summary10.setElement(test.name,'Time',test.elapse_time)
        
        
    summary05 = Table()
    summary05.setColumnFormat('Face_Rate','%0.4f')
    summary05.setColumnFormat('Rate_05','%0.4f')
    summary05.setColumnFormat('Lower95_05','%0.4f')
    summary05.setColumnFormat('Upper95_05','%0.4f')
    summary05.setColumnFormat('Time','%0.2f')
    for test in tests:
        summary05.setElement(test.name,'Face_Rate',test.face_rate)
        summary05.setElement(test.name,'Rate_05',test.both05_rate)
        summary05.setElement(test.name,'Lower95_05',test.both05_ci[0])
        summary05.setElement(test.name,'Upper95_05',test.both05_ci[1])
        summary05.setElement(test.name,'Time',test.elapse_time)
        
    return summary05,summary10,summary25
