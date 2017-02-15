'''
Copyright 2016 Oak Ridge National Laboratory
Created on Jun 6, 2016

@author: qdb
'''
import BaseHTTPServer
import SocketServer
import StringIO
import os
import pyvision as pv

BASIC_HTML = '''
<HTML>
<HEAD>
<TITLE>BASIC_HTML</TITLE>
</HEAD>
<BODY>
Replace This.
</BODY>
</HTML>
'''


IMAGELIST_HTML = '''
<HTML>
<HEAD>
<TITLE>Image List</TITLE>
</HEAD>
<BODY>

<TABLE BORDER=1>
<TR><TH>Image Name</TH><TH>Complete</TH><TH>Date</TH></TR>
%s
</TABLE>
</BODY>
</HTML>
'''

class ImageSurvey(BaseHTTPServer.BaseHTTPRequestHandler):
    '''
    Runs and image survey through html.
    '''


    #def __init__(self,*args,**kwargs):
    #    '''
    #    Constructor
    #    '''
    #    
    #    SimpleHTTPServer.SimpleHTTPRequestHandler.__init__(self,*args,**kwargs)
        
    def do_GET(self):
        self.sendImageList()
        #print 'Processing GET',self.headers
        #self.send_response(200)
        #self.send_header('Content-Type', 'text/html')
        #self.end_headers()
        
        #self.wfile.write(IMAGELIST_HTML)
        #BaseHTTPServer.BaseHTTPRequestHandler.do_GET(self)
    

    #def handleRequest(self):
    #    print "test"
    #    print dir(self)

    def sendImageList(self):
        print 'Processing GET',self.headers
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        
        table_data = StringIO.StringIO()
        for image_name in os.listdir('images'):
            print image_name
            if pv.isImage(image_name):
                table_data.write("<TR><TD> %s </TD></TR>\n"%image_name)
            
        tmp = table_data.getvalue()
        print tmp
        self.wfile.write(IMAGELIST_HTML%(tmp,))

if __name__ == '__main__':
    PORT = 8000
    
    
    #Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
    
    httpd = SocketServer.TCPServer(("127.0.0.1", PORT), ImageSurvey)
    
    print "serving at port", PORT
    httpd.serve_forever()