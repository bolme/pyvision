'''
Created on Nov 29, 2012

@author: bolme
'''

import os
import pyvision as pv
import io
import base64

class HtmlReport(object):
    '''
    classdocs
    '''

    def __init__(self,title="untitled"):
        '''
        Constructor
        '''
        self.title='untitled'
        self.elements = []
        self.hidden_id = 0
        
    def table(self,table):
        self.elements.append(table)
        
    def hr(self):
        '''
        Add a horizontal rule.
        '''
        self.elements.append("\n\n<!---------------------------------------------------------------->\n<HR>\n\n")
    
    def br(self):
        '''
        Add a horizontal rule.
        '''
        self.elements.append("\n<BR>\n\n")
    
    def section(self,name):
        '''
        Add a section header.
        '''
        self.elements.append("<H2>%s</H2>\n\n"%(name))
        
    def comment(self,name):
        '''
        Add a comment.
        '''
        self.elements.append("<!-- %s -->\n\n"%(name))
        
    def html(self,name):
        '''
        Insert raw unmodified html.
        '''
        self.elements.append(name)
        
    def p(self,text):
        '''
        Insert text as a paragraph.
        '''
        self.elements.append('<p>%s</p>\n\n'%(text))
        
    def line(self,text=""):
        '''
        insert text followed by a line break
        '''
        self.elements.append('%s<br/>\n'%(text))
        
    def asText(self):
        text = ""
        text += "<HTML>\n"
        text += "<HEAD>\n"
        text += "  <TITLE>%s</TITLE>\n"%(self.title,)
        text += "</HEAD>\n"
        text += "<BODY>\n\n\n"
        for element in self.elements:
            if isinstance(element,str):
                text += element
            if isinstance(element,pv.Table):
                text += element.asHtml(print_row_headers=True)
        text += "\n\n</BODY>\n"
        text += "</HTML>\n"
        return text
    
    def save(self,path,show=False):
        f = open(path,'wb')
        f.write(self.asText())
        f.close()
        if show:
            os.system('open %s'%path)
            
    def start_hidden(self):
        self.hidden_id += 1
        check_id = "check_%s"%self.hidden_id
        obj_id = "hideable_%s"%self.hidden_id
        self.elements.append('''[<font color='blue'><a id="%(check_id)s" onClick="
                                    if(document.getElementById('%(check_id)s').innerHTML=='show'){
                                        document.getElementById('%(check_id)s').innerHTML='hide';
                                        document.getElementById('%(hidden_id)s').style.display = 'block';
                                    } 
                                    else {
                                        document.getElementById('%(check_id)s').innerHTML='show';
                                        document.getElementById('%(hidden_id)s').style.display = 'none';
                                    };"
                                    >show</a></font>]<br>\n'''%{'hidden_id':obj_id,'check_id':check_id} )
        self.elements.append('''<div id="%(hidden_id)s" style="display:none">\n'''%{'hidden_id':obj_id})

    def end_hidden(self):
        self.elements.append('''</div>''')
        
    def image(self,im,format='jpg'):
        f = io.StringIO()
        im.asAnnotated().save(f,format)
        #f.flush()
        #print "Saving Image",len(f.getvalue())
        encoded = base64.b64encode(f.getvalue())
        im_tag = '<img src="data:image/%s;base64,%s" />'%(format,encoded)
        self.elements.append(im_tag)

        
if __name__ == '__main__':
    rpt = HtmlReport()
    rpt.section("Summary")
    rpt.hr()
    rpt.comment("Hello world.")
    rpt.section('Section 1')
    rpt.p("Here is some text.")
    rpt.p("Here are some more text.")
    rpt.html("AAPL data ")
    rpt.start_hidden()
    rpt.line()
    rpt.line("Here is a number: 27.89")
    rpt.line("Here is a number: 29.89")
    rpt.line("Here is a number: 3.89")
    rpt.line()
    rpt.end_hidden()
    #rpt.line()
    rpt.html("XOM data ")
    rpt.start_hidden()
    rpt.line("Here is a number: 27.89")
    rpt.line("Here is a number: 29.89")
    rpt.line("Here is a number: 3.89")
    rpt.line()
    rpt.end_hidden()
    #rpt.line()
    print(rpt.asText())
    rpt.save("/Users/bolme/test_report.html",show=True)
    
    
    