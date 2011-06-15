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
from os import makedirs,system,listdir
from os.path import join
import csv
import pickle
import sys
import pyvision as pv

class ImageLog:
    '''
    An image log is used to collect data about an experiment.   This data can be
    images, tables, pickled python objects.
    '''
    
    def __init__(self,topdir = "/tmp",name="pyvis_log"):
        '''
        Create an image log.  By default a log is created in the /tmp dir.  
        The name of the log directory start with the date and time the log
        was created and ends with 'name'.
        
        @param topdir: The location where the log directory will be created.
        @param name: a name to append to the directory name.
        '''
        self.date = strftime("%Y%m%d_%H%M%S")
        self.name=name
        self.dir = topdir+'/'+self.date+'_'+name
        makedirs(self.dir)
        self.count = 0
        #print self.dir
        
    def __call__(self,item,*args, **kwargs):
        if isinstance(item,pv.Image):
            self.log(item,*args,**kwargs) 
        elif isinstance(item,pv.Timer):
            self.table(item.table,*args,**kwargs) 
        elif isinstance(item,pv.Table):
            self.table(item,*args,**kwargs) 
        elif isinstance(item,pv.Plot):
            self.plot(item,*args,**kwargs) 
        else:
            self.pickle(item,*args,**kwargs) 
        
    def log(self,image,label="NOLABEL",format='png'):
        '''
        Save a pyvision image to the log.
        '''
        image.asAnnotated().save(self.dir+'/%06d_%s.%s'%(self.count,label,format),quality=95)
        self.count += 1
        #print message
    
    def plot(self,plot,label="NOLABEL",format='png'):
        '''
        Save a pyvision plot to the log.
        '''
        plot.asImage().asAnnotated().save(self.dir+'/%06d_%s.%s'%(self.count,label,format),quality=95)
        self.count += 1
        #print message
    
    def table(self,table,label="NOLABEL"):
        '''
        Save a pyvision table to the log as a csv file.
        '''
        filename = join(self.dir,'%06d_%s.csv'%(self.count,label))
        table.save(filename)
        self.count += 1
        
    def csv(self,data,label="NOLABEL",headers=None):
        '''
        Save a list of lists or matrix to the log as a csv file.
        '''
        filename = join(self.dir,'%06d_%s.csv'%(self.count,label))
        writer = csv.writer(open(filename, "wb"))
        if headers:
            writer.writerow(headers)
        writer.writerows(data)
        self.count += 1
        #print message
    
    def pickle(self,object,label="NOLABEL"):
        '''
        Pickle a python object to the log directory.  This may not be supported 
        by all objects. 
        
        '''
        filename = join(self.dir,'%06d_%s.pkl'%(self.count,label))
        f = open(filename,'wr')
        pickle.dump(object, f)
        f.close()
        self.count += 1
        
    def file(self,data=None,label="NOLABEL",ext='.dat'):
        '''
        Write the buffer data to a file.  If data == None then the file object is returned.
        '''
        filename = join(self.dir,'%06d_%s%s'%(self.count,label,ext))
        self.count += 1
        f = open(filename,'wb')
        if data != None:
            f.write(data)
            f.close()
        else:
            return f
    
    def show(self):
        '''
        Show any images that are in the directory.
        '''
        files = listdir(self.dir)
        file_list = ""
        for each in files:
            if len(file_list) > 30000:
                sys.stderr.write("<ImageLog> Warning can't display all images.\n")
                break
            if each.split('.')[-1].upper() in ['JPG','PNG','TIFF','TIF','GIF']:
                file_list += " %s/%s"%(self.dir,each)

        if file_list == "":
            sys.stderr.write('<ImageLog> No images to show.\n')
            return
            
        if sys.platform.startswith("darwin"):
            system("open %s"%file_list)
        elif sys.platform.startswith("linux"):
            files.sort()
            startfile = join(self.dir, files.pop(0))
            #gthumb will show thumbnails for all files in same directory as startfile.
            #If you use KDE, gwenview might be better...
            system("gthumb %s"%startfile)
        elif sys.platform.startswith("windows"):
            print "ImageLog.show() is not supported on windows."
        
        