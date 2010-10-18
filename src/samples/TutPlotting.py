'''
Created on Oct 17, 2010

@author: bolme
'''

import pyvision as pv
import numpy as np

if __name__ == "__main__":
    ilog = pv.ImageLog()
    
    plot = pv.Plot(size=(600,600),title="Test Plot 1",xrange=(0,25),yrange=(0,25))
    
    plot.label([0,25],"Shapes (Size=7)",align='right')
    for i in range(0,26):
        plot.point((i,24),shape=i,size=7,color='red')
        
    plot.label([0,23],"Shapes (Size=5)",align='right')
    for i in range(0,26):
        plot.point((i,22),shape=i,size=5,color='red')
        
    plot.label([0,21],"Shapes (Size=3)",align='right')
    for i in range(0,26):
        plot.point((i,20),shape=i,size=3,color='red')
        
    plot.label([0,19],"Upper Case Alphabet Large (Size=7)",align='right')
    for i in range(0,26):
        c = chr(ord('A')+i)
        plot.point((i,18),shape=c,size=7,color='red')
        
    plot.label([0,17],"Upper Case Alphabet Small (Size=5)",align='right')
    for i in range(0,26):
        c = chr(ord('A')+i)
        plot.point((i,16),shape=c,size=5,color='red')
        
    plot.label([0,15],"Lower Case Alphabet Large (Size=7)",align='right')
    for i in range(0,26):
        c = chr(ord('a')+i)
        plot.point((i,14),shape=c,size=7,color='red')
        
    plot.label([0,13],"Lower Case Alphabet Small (Size=5)",align='right')
    for i in range(0,26):
        c = chr(ord('a')+i)
        plot.point((i,12),shape=c,size=5,color='red')
        
    plot.label([0,11],"Numbers Large (Size=7)",align='right')
    for i in range(0,26):
        plot.point((i,10),shape="%d"%i,size=8,color='red')
        
    plot.label([0,9],"Numbers Small (Size=5)",align='right')
    for i in range(0,26):
        plot.point((i,8),shape="%d"%i,size=5,color='red',label="%d"%i)

    plot.label([0,6],"Here is an example of some lines...",align='right')
        
    data = []
    for i in range(0,30):
        x = float(i)
        data.append((x,np.sqrt(x)))
    plot.lines(data,color='red',width=3)
    plot.points(data,color='blue')
    
    print "The plot with charecters can take a while to render..."
    ilog(plot,"Test_Plot_1")
    print "done."    
    ilog.show()
