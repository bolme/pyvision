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

import csv
import StringIO
import pyvision as pv

def convertVal(val):
    if val in ("True","False"):
        return val == "True"
    try:
        flt = float(val)
    except:
        return val
    
    if '.' in val:
        return flt
    else: 
        return int(round(flt))

class Table:
    '''
    Store and manipulate table data
    '''
    
    def __init__(self,filename=None,default_value=None):
        self.col_headers = []
        self.col_label = None
        self.row_headers = []
        self.row_label = None
        self.col_format = {}
        self.hlines = False
        self.default_value = default_value
        self.data = {}
        self.rebuildTable()
        
        if filename != None:
            if isinstance(filename,str):
                f = open(filename,'rb')
                self.load(f)
            else:
                #Assume this is a file pointer
                self.load(filename)
        
    def load(self,f):
        reader = csv.reader(f)
        header=None
        #j = 0 
        for data in reader:
            if header == None:
                header = data
                assert header[0] == "row"
                continue
            assert len(header) == len(data)
            
            for i in range(1,len(header)):
                row = convertVal(data[0])
                col = convertVal(header[i])
                val = convertVal(data[i])
                #print type(row),row,type(col),col,type(val),val
                self[row,col] = val
            
            #j += 1
            #if j > 300:
            #    break
            
    def sortByRowHeader(self,comp_func = cmp):
        self.row_headers.sort(comp_func)
        
        
    def sortByColHeader(self,comp_func = cmp):
        self.col_headers.sort(comp_func)
        
                
    def accumulateData(self,row,col,value):
        new_value = self.element(row,col) + value
        self.setData(row,col,new_value)


    def setColumnFormat(self,col,format):
        if format == None and self.col_format.has_key(col):
            # Clear formating for that column
            del self.col_format[col]
        else:
            # Set the formating for that column
            self.col_format[col] = format
            
    def __setitem__(self,key,value):
        return self.setElement(key[0],key[1],value)
    
    def setElement(self,row,col,value,accumulate=False):
        self.setData(row,col,value,accumulate=accumulate)
        
    def setData(self,row,col,value,accumulate=False):
        # Set an entry in the table
        labels_updated = False
        
        if col not in self.col_headers:
            self.col_headers.append(col)
            labels_updated = True
            
        if row not in self.row_headers:
            self.row_headers.append(row)
            labels_updated = True
            
        if labels_updated:
            self.rebuildTable()

        self.data[row][col] = value
        
        
    def rebuildTable(self):
        for row in self.row_headers:
            if not self.data.has_key(row): self.data[row] = {}
            for col in self.col_headers:
                if not self.data[row].has_key(col): self.data[row][col] = self.default_value
        
        
    def hasElement(self,row,col):
        return (self.data.has_key(row) and self.data[row].has_key(col))

     
    def __getitem__(self,key):
        return self.element(key[0],key[1])
       
    def element(self,row,col):
        if self.hasElement(row,col):
            return self.data[row][col]
        else:
            return self.default_value
        
    def elementAsText(self,row,col):
        if isinstance(self.col_format,str):
            return self.col_format%self.element(row,col)
        if isinstance(self.col_format,dict) and self.col_format.has_key(col):
            return self.col_format[col]%self.element(row,col)
        # Todo it would be nice to support callable objects
            
        # Otherwize just format as a string:
        return "%s"%(self.element(row,col),)
    
    def justifyText(self,text,width,side='l'):
        assert  side in ['c','l','r']
        l = len(text)
        if l > width:
            return text[:width]
        else:
            pad = width - l
            if side=='c':
                front = " "*(pad/2)
                back  = " "*(pad - pad/2)
                return front + text + back
            elif side=='r':
                return " "*pad + text
            else:
                return text + " "*pad

        return text #TODO:

    def asHtml(self, print_col_headers = True, print_row_headers = False, equal_cols = False, style='simple'):
        result = "<TABLE CELLPADDING=6 CELLSPACING=0>\n"
        result += '  <TR BGCOLOR="#D3C6AD">\n'
        result += '    '
        if print_row_headers:
            result += '<TD ALIGN=LEFT></TD>'
        
        for col in self.col_headers:
            result += '<TH>%s</TH>'%col
        result += '\n'
        result += '  </TR>\n'
        
        i = 0
        for row in self.row_headers:
            bgcolor = '#FBEBCE'
            if i % 2 == 1:
                bgcolor = '#EFE0C4'
            result += '  <TR BGCOLOR="%s">\n'%bgcolor
            result += '    '
            if print_row_headers:
                result += '<TD ALIGN=LEFT>%s</TD>'%(row,)
            for col in self.col_headers:
                align = 'LEFT'
                try:
                    #Check to see if the text looks like a number
                    val = float(self[row,col])
                    align = 'RIGHT'
                except:
                    # Default, justify left
                    pass

                val = self.elementAsText(row,col)
                result += '<TD ALIGN=%s>%s</TD>'%(align,val)
            result += '\n'
            result += '  </TR>\n'
            i += 1
        result += "</TABLE>\n"
        return result
        

    
    def asPlainText(self, print_col_headers = True, print_row_headers = True, equal_cols = False, separator="|"):
        '''Returns a text string which is a formated table.''' 
        assert len(separator) == 1

        rows = self.row_headers
        cols = self.col_headers

        col_widths = {}
        for col in self.col_headers:
            if print_col_headers:
                col_widths[col] = len(str(col))
            else:
                col_widths[col] = 0
            for row in self.row_headers:
                w = len(self.elementAsText(row,col))
                if w > col_widths[col]:
                    col_widths[col] = w
        if equal_cols:
            new_widths = {}
            max_width = 0
            for key,value in col_widths.iteritems():
                max_width = max(max_width,value)
            for key in col_widths.keys():
                col_widths[key]=max_width
        
        row_header_width = 0
        for row in rows:
            row_header_width = max(row_header_width,len(str(row)))
                       

        out = ""
        
        #Horizontal Rule
        if print_row_headers:
            out += "|" + "-"*(row_header_width+2)
        out += "|"
        for col in cols:
            out += "-"*(col_widths[col]+2)+"|"
        out = out[:-1]
        out += "|\n"

        if print_col_headers:
            out += "|"
            if print_row_headers:
                out += " "*(row_header_width+2)+"|"
            for col in cols:
                text = self.justifyText(str(col),col_widths[col],'l')
                out += " "+text+" |"
            out = out[:-1]
            out += "|\n"
                
            #Horizontal Rule
            if print_row_headers:
                out += "|" + "-"*(row_header_width+2)
            out += "|"
            for col in cols:
                out += "-"*(col_widths[col]+2)+"|"
            out = out[:-1]
            out += "|\n"

        for row in rows:
            out +="|"
            if print_row_headers:
                out += " " + self.justifyText(str(row),row_header_width,'l')+" |"
            for col in cols:
                text = self.elementAsText(row,col)
                try:
                    #Check to see if the text looks like a number
                    val = float(text)
                    # Numbers should be justifed right.
                    text = self.justifyText(text,col_widths[col],'r')
                except:
                    # Default, justify left
                    text = self.justifyText(text,col_widths[col],'l')
                assert len(text) == col_widths[col]
                out += " "+text+" "+separator
            #strip the last separator
            out = out[:-1]
            out += "|\n"

        #Horizontal Rule
        if print_row_headers:
            out += "|" + "-"*(row_header_width+2)
        out += "|"
        for col in cols:
            out += "-"*(col_widths[col]+2)+"|"
        out = out[:-1]
        out += "|\n"

        return out
    
            
            
                
                
    def nRows(self):
        return len(self.row_headers)

    def nCols(self):
        return len(self.col_headers)
            
    
    def asTex(self):
        '''Returns a text string which as a table formated for latex'''
        
    def asLists(self,headers=True):
        '''Returns the table data as a list of lists'''
        rows = self.row_headers
        cols = self.col_headers
        
        result = []
        
        if headers:
            tmp = ['row']
            for col in cols:
                tmp.append(col)
            result.append(tmp)

        for row in rows:
            tmp = []
            if headers:
                tmp.append(row)
            for col in cols:
                tmp.append(self.element(row,col))
            result.append(tmp)
        
        return result
        
        
    def head(self,N=10):
        '''Returns a table from the first N rows.'''
        rows = self.row_headers
        cols = self.col_headers
        
        result = pv.Table()
        
        for row in rows[:N]:
            for col in cols:
                result[row,col] = self[row,col]
            
        return result
        
        
    def tail(self,N=10):
        '''Returns a table from the last N rows.'''
        rows = self.row_headers
        cols = self.col_headers
        
        result = pv.Table()
        
        for row in rows[-N:]:
            for col in cols:
                result[row,col] = self[row,col]
            
        return result
        
        
    def save(self,filename,headers=True):
        '''Save the table to CSV'''
        if isinstance(filename,str):
            f = open(filename,'wb')
        else:
            f = filename # assume file pointer
        writer = csv.writer(f)
        writer.writerows(self.asLists(headers=headers))
        
    def __str__(self):
        return self.asPlainText()
    
    
import unittest
class _TestTable(unittest.TestCase):
    def setUp(self):
        color = Table(default_value=0)
        color.accumulateData('red','red',1)
        color.accumulateData('red','red',1)
        color.accumulateData('red','red',1)
        color.accumulateData('blue','blue',1)
        color.accumulateData('blue','blue',1)
        color.accumulateData('blue','blue',1)
        color.accumulateData('blue','blue',1)
        color.accumulateData('pink','pink',1)
        color.accumulateData('pink','pink',1)
        color.accumulateData('pink','pink',1)
        color.accumulateData('pink','pink',1)
        color.accumulateData('pink','pink',1)
        color.accumulateData('pink','red',1)
        color.accumulateData('pink','red',1)
        color.accumulateData('blue','red',1)
        color.accumulateData('blue','red',1)
        color.accumulateData('red','blue',1)
        color.accumulateData('green','green',1)
        color.accumulateData('red','green',1)
        self.color = color
        
        # Simulate a face recognition problem with a
        # probe set of 1000 and a gallery set of 1000
        # 0.001 FAR and 0.100 FRR
        sim_face = Table(default_value=0)
        sim_face.setData('accept','accept',900)
        sim_face.setData('reject','reject',998001)
        sim_face.setData('accept','reject',100)
        sim_face.setData('reject','accept',999)
        self.sim_face = sim_face
        
    def test__str__(self):
        
        self.color.asPlainText()
        self.sim_face.asPlainText()
        
    def test_asLists(self):
        self.color.asLists()

    def test_save(self):
        expected = 'row,red,blue,pink,green\r\nred,3,1,0,1\r\nblue,2,4,0,0\r\npink,2,0,5,0\r\ngreen,0,0,0,1\r\n'
        output = StringIO.StringIO()
        self.color.save(output)
        self.assertEqual(output.getvalue(),expected)

        expected = '3,1,0,1\r\n2,4,0,0\r\n2,0,5,0\r\n0,0,0,1\r\n'
        output = StringIO.StringIO()
        self.color.save(output,headers=False)
        self.assertEqual(output.getvalue(),expected)
        
    def test_verification(self):
        self.sim_face

    def test_rowsort(self):
        tab = Table()
        
        tab[2,1] = 'b'
        tab[1,1] = 'a'
        tab[3,1] = 'c'

        #print tab        
        tab.sortByRowHeader()
        #print tab
        
     