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

from time import *
from os import mkdir,system
from os.path import join
import csv
import pickle
import sys

class ImageLog:
    
    def __init__(self,topdir = "/tmp",name=None):
        self.date = strftime("%Y%m%d_%H%M%S")
        self.name=name
        if name:
            self.dir = topdir+'/'+self.date+'_'+name
        else: 
            self.dir = topdir + "/" + self.date + "_pyvis_log"
        mkdir(self.dir)
        self.count = 0
        #print self.dir
        
        
    def log(self,image,message=None,label="NOLABEL"):
        image.asAnnotated().save(self.dir+'/%012d_%s.png'%(self.count,label))
        self.count += 1
        #print message
    
    def table(self,table,label="NOLABEL"):
        filename = join(self.dir,'%012d_%s.csv'%(self.count,label))
        table.save(filename)
        self.count += 1
        
    def csv(self,data,headers=None,label="NOLABEL"):
        filename = join(self.dir,'%012d_%s.csv'%(self.count,label))
        writer = csv.writer(open(filename, "wb"))
        if headers:
            writer.writerow(headers)
        writer.writerows(data)
        self.count += 1
        #print message
    
    def pickle(self,object,label="NOLABEL"):
        '''
        Pickle an object to the log directory.
        '''
        filename = join(self.dir,'%012d_%s.pkl'%(self.count,label))
        f = open(filename,'wr')
        pickle.dump(object, f)
        f.close()
        self.count += 1
    
    def show(self):
        if sys.platform.startswith("darwin"):
            system("open %s/*.png"%self.dir)
        elif sys.platform.startswith("linux"):
            system("gqview %s/*.png"%self.dir)
        elif sys.platform.startswith("windows"):
            pass
        
        