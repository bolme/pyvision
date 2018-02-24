'''
A very simple 2D plotting package that outputs images.

@author: bolme
'''

import pyvision as pv
import numpy as np
import os.path
import PIL.ImageFont
import PIL.ImageDraw
random = np.random
import io
import sys

arial_path = os.path.join(pv.__path__[0],'config','Arial.ttf')
huge_font = PIL.ImageFont.truetype(arial_path, 36)
large_font = PIL.ImageFont.truetype(arial_path, 18)
small_font = PIL.ImageFont.truetype(arial_path, 12)


def drawLabel(plot_image, pt, label, size='small',align='center',rotate=False,color='black'):
    '''
    @param plot_image: a PIL image to use as a plot_image
    '''
    font = small_font
    if size == 'large':
        font = large_font
    elif size == "huge":
        font = huge_font
        
    # Render the mask
    size = font.getsize(label)
    im = PIL.Image.new("L",size,'black')
    draw = PIL.ImageDraw.Draw(im)
    draw.text((0,0),label,fill='white',font=font)
    del draw
    
    # Rotate
    if rotate:
        im = im.transpose(PIL.Image.ROTATE_90)
    
    # Render the text
    x,y = pt
    w,h = im.size
    if align == 'center':
        x = x - w/2
        y = y - h/2 
    elif align == 'left':
        x = x - w
        y = y - h/2 
    elif align == 'right':
        x = x
        y = y - h/2 
    elif align == 'above':
        x = x - w/2
        y = y - h
    elif align == 'bellow':
        x = x - w/2
        y = y
    else:
        raise ValueError("Unknown alignment: %s"%align)
    draw = PIL.ImageDraw.Draw(plot_image)
    draw.bitmap((x,y),im,fill=color)
    del draw
    #plot_image.(color,(x,y,x+w,y+h),im)
    
    
def dataToFormatedList(data):
    out = ""
    # group data into rows of 8
    for i in range(0,len(data),8):
        part = data[i:i+8]
        for each in part:
            out += "%s,"%each
        if i + 8 >= len(data):
            # Finish
            out = out[:-1] # Remove the last comma
        else:
            # Continue with next row
            out += "\n        "
    return out

class Label:
    def __init__(self,point,label,size='small',align='center',rotate=False,color='black'):
        self.point = point[:2]
        self.label = label
        self.size = size
        self.align = align
        self.rotate = rotate
        self.color = color
        
    def draw(self,plot,plot_image,bounds):
        x,y = self.point
        x = plot.x(x,bounds)
        y = plot.y(y,bounds)
        drawLabel(plot_image,[x,y],self.label,size=self.size,align=self.align,rotate=self.rotate,color=self.color)
        
    def drawR(self,f):
        sys.stderr.write("WARNING> Plot draw label is not implemented for R.\n")
        
    def range(self):
        x,y = self.point

        return x,x,y,y

        

class Points:
    def __init__(self,points,graphic_type,color='black',shape=0,size=3,label=None,lty=None,width=1):
        ''''''
        self.graphic_type = graphic_type
        self.points = points
        self.color = color
        self.shape = shape
        self.size = size
        self.label = label
        self.width = width
        self.lty = lty
        
        
    def draw(self,plot,plot_image,bounds):
        ''''''
        points = [ (plot.x(x,bounds),plot.y(y,bounds)) for x,y in self.points]
        
        if self.graphic_type == 'lines':
            self.drawCurve(points,plot_image)
        elif self.graphic_type == 'points':
            self.drawPoints(points,plot_image)
        elif self.graphic_type == 'polygon':
            self.drawPolygon(points,plot_image)
        else:
            raise ValueError("unknown graphic_type: %s"%(self.graphic_type,))

    def drawR(self,f):
        ''''''
        xpoints = [ float(x) for x,y in self.points]
        ypoints = [ float(y) for x,y in self.points]
        
        f.write("xpoints=c(%s)\n"%(dataToFormatedList(xpoints),))
        f.write("\n")
        f.write("ypoints=c(%s)\n"%(dataToFormatedList(ypoints),))
        f.write("\n")
        
        if self.graphic_type == 'lines':
            f.write("lines(xpoints,ypoints,lty=%s,col='%s',lwd='%s')"%(self.lty,self.color,self.width))
        elif self.graphic_type == 'points':
            f.write("points(xpoints,ypoints,col='%s',pch=%s,cex=%s,lwd=%s)"%(self.color,repr(self.shape),self.size/3.0,self.width))
        elif self.graphic_type == 'polygon':
            xpoints.append(xpoints[0])
            ypoints.append(ypoints[0])
            f.write("lines(xpoints,ypoints,lty=%s,col='%s',lwd='%s')"%(self.lty,self.color,self.width))
        else:
            raise ValueError("unknown graphic_type: %s"%(self.graphic_type,))

    def drawCurve(self,points,plot_image):
        draw = PIL.ImageDraw.Draw(plot_image)
        prev_x,prev_y = points[0]
        for x,y in points[1:]:
            draw.line([prev_x,prev_y,x,y],fill=self.color,width=self.width)
            prev_x = x
            prev_y = y
        del draw        
        
    def drawPolygon(self,points,plot_image):
        if len(points) < 3:
            return # No Op
        draw = PIL.ImageDraw.Draw(plot_image)
        draw.polygon(points, fill=self.color)
        del draw        
        
    def drawPoints(self,points,plot_image):
        ''' Render points on the plot '''
        shape = self.shape
        size = self.size
        color = self.color

        draw = PIL.ImageDraw.Draw(plot_image)
        
        for i in range(len(points)):
            x,y = points[i]
            
            if isinstance(shape,int) and shape >= 0:
                shape = (shape) % 21
                
            if shape == '.':
                # fill in a pixel
                draw.point((x,y),fill=color)                
            elif shape == 0:
                # hollow square
                draw.rectangle((x-size,y-size,x+size,y+size),outline=color,fill=None)
            elif shape == 1:
                # hollow circle
                draw.ellipse((x-size,y-size,x+size,y+size),outline=color,fill=None)
            elif shape == 2:
                # hollow triangle
                draw.polygon((x,y-size,x-size,y+size,x+size,y+size,x,y-size),outline=color,fill=None)
            elif shape == 3:
                # plus
                draw.line((x-size,y,x+size,y),fill=color)
                draw.line((x,y-size,x,y+size),fill=color)
            elif shape == 4:
                # cross
                draw.line((x-size,y-size,x+size,y+size),fill=color)
                draw.line((x-size,y+size,x+size,y-size),fill=color)
            elif shape == 5:
                # hollow diamond
                draw.polygon((x-size,y,x,y-size,x+size,y,x,y+size,x-size,y),outline=color,fill=None)
            elif shape == 6:
                # hollow inv triangle
                draw.polygon((x,y+size,x-size,y-size,x+size,y-size,x,y+size),outline=color,fill=None)
            elif shape == 7:
                # box cross
                draw.rectangle((x-size,y-size,x+size,y+size),outline=color,fill=None)
                draw.line((x-size,y-size,x+size,y+size),fill=color)
                draw.line((x-size,y+size,x+size,y-size),fill=color)
            elif shape == 8:
                # star
                draw.line((x-size,y,x+size,y),fill=color)
                draw.line((x,y-size,x,y+size),fill=color)
                draw.line((x-size,y-size,x+size,y+size),fill=color)
                draw.line((x-size,y+size,x+size,y-size),fill=color)
            elif shape == 9:
                # diamond plus
                draw.polygon((x-size,y,x,y-size,x+size,y,x,y+size,x-size,y),outline=color,fill=None)
                draw.line((x-size,y,x+size,y),fill=color)
                draw.line((x,y-size,x,y+size),fill=color)
            elif shape == 10:
                # circle plus
                draw.ellipse((x-size,y-size,x+size,y+size),outline=color,fill=None)
                draw.line((x-size,y,x+size,y),fill=color)
                draw.line((x,y-size,x,y+size),fill=color)
            elif shape == 11:
                # double triangle
                draw.polygon((x,y-size,x-size,y+0.707*size,x+size,y+0.707*size,x,y-size),outline=color,fill=None)
                draw.polygon((x,y+size,x-size,y-0.707*size,x+size,y-0.707*size,x,y+size),outline=color,fill=None)
            elif shape == 12:
                # box plus
                draw.rectangle((x-size,y-size,x+size,y+size),outline=color,fill=None)
                draw.line((x-size,y,x+size,y),fill=color)
                draw.line((x,y-size,x,y+size),fill=color)
            elif shape == 13:
                # circle cross
                draw.ellipse((x-size,y-size,x+size,y+size),outline=color,fill=None)
                draw.line((x-size,y-size,x+size,y+size),fill=color)
                draw.line((x-size,y+size,x+size,y-size),fill=color)
            elif shape == 14:
                # box triangle
                draw.rectangle((x-size,y-size,x+size,y+size),outline=color,fill=None)
                draw.polygon((x,y-size,x-size,y+size,x+size,y+size,x,y-size),outline=color,fill=None)
            elif shape == 15:
                # solid square
                draw.rectangle((x-size,y-size,x+size,y+size),outline=color,fill=color)
            elif shape == 16:
                # solid circle
                draw.ellipse((x-size,y-size,x+size,y+size),outline=color,fill=color)
            elif shape == 17:
                # solid triangle
                draw.polygon((x,y-size,x-size,y+size,x+size,y+size,x,y-size),outline=color,fill=color)
            elif shape == 18:
                # solid diamond
                draw.polygon((x-size,y,x,y-size,x+size,y,x,y+size,x-size,y),outline=color,fill=color)
            elif shape == 19:
                # solid circle
                draw.ellipse((x-size,y-size,x+size,y+size),outline=color,fill=color)
            elif shape == 20:
                # solid circle
                draw.ellipse((x-0.707*size,y-0.707*size,x+0.707*size,y+0.707*size),outline=color,fill=color)

#            elif shape == 11:
#                # solid inv triangle
#                draw.polygon((x,y+size,x-size,y-size,x+size,y-size,x,y+size),outline=color,fill=color)
#            elif shape == 13:
#                # solid right triangle
#                draw.polygon((x-size,y-size,x+size,y,x-size,y+size,x-size,y-size),outline=color,fill=color)
#            elif shape == 14:
#                # hollow right triangle
#                draw.polygon((x-size,y-size,x+size,y,x-size,y+size,x-size,y-size),outline=color,fill=None)
#            elif shape == 15:
#                # solid left triangle
#                draw.polygon((x+size,y-size,x+size,y+size,x-size,y,x+size,y-size),outline=color,fill=color)
#            elif shape == 16:
#                # hollow left triangle
#                draw.polygon((x+size,y-size,x+size,y+size,x-size,y,x+size,y-size),outline=color,fill=None)
            elif type(shape) == str:
                # render as a text label
                if size < 6:
                    drawLabel(plot_image, (x,y), shape, size='small',align='center',rotate=False,color=color)
                else:
                    drawLabel(plot_image, (x,y), shape, size='large',align='center',rotate=False,color=color)
            else:
                # fill in a pixel
                draw.point((x,y),fill=color)
            
        del draw

    def range(self):
        points = self.points
        
        points = np.array(points)
        
        
        xmin,xmax,ymin,ymax = points[0][0],points[0][0],points[0][1],points[0][1]
                
        for x,y in self.points:
            #print x,y,xmin,ymin
            xmin = np.min([xmin,x])
            xmax = np.max([xmax,x])
            ymin = np.min([ymin,y])
            ymax = np.max([ymax,y])
        #print "PointRange:", xmin,xmax,ymin,ymax
        return xmin,xmax,ymin,ymax
            
            


class Plot:
    def __init__(self,size=(600,600),x_range=None,y_range=None,title="No Title",ylabel="Y Axis",xlabel="X Axis",pad=True):
        # Save plot information
        self.size = size
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x_at = None
        self.x_labels = None
        
        self.x_range = x_range
        if x_range != None:
            self.x_range = (np.min(x_range),np.max(x_range))
        
        self.y_range = y_range
        if y_range != None:
            self.y_range = (np.min(y_range),np.max(y_range))
        
        self.top = 60
        self.left = 60
        self.bottom = 60
        self.right = 30
        
        self.pad = pad
        
        self.graphics = []
        
        
        
        
    def asImage(self):
        w,h = self.size

        fw = self.left+w+self.right
        fh = self.top+h+self.bottom

        pil = PIL.Image.new("RGB",(fw,fh),'white')
        
        drawLabel(pil,(0.5*fw,0.5*self.top),self.title,align='center',size='huge')
        drawLabel(pil,(5,self.top+0.5*h),self.ylabel,align='right',size='large',rotate=True)
        drawLabel(pil,(self.left+0.5*w,fh-5),self.xlabel,align='above',size='large',rotate=False)
        
        tmp_image = PIL.Image.new("RGB",(w,h),'white')
        bounds = self.range()
        for each in self.graphics:
            each.draw(self,tmp_image,bounds)
            
        pil.paste(tmp_image,(self.left,self.top))
        
        draw = PIL.ImageDraw.Draw(pil)
        draw.rectangle((self.left,self.top,self.left+w,self.top+h),outline='black',fill=None)
        del draw
        
        # Draw the X axis labels
        at,labels = self.xAxis()
        
        # Tickmarks
        draw = PIL.ImageDraw.Draw(pil)
        for i in range(len(at)):
            x = self.left+self.x(at[i],bounds)
            y = self.top+h
            draw.line((x,y,x,y+5),fill='black')
        del draw
        
        # Labels
        for i in range(len(at)):
            x = self.left+self.x(at[i],bounds)
            y = self.top+h
            drawLabel(pil, (x,y+5), labels[i], size='large',align='bellow',rotate=False,color='black')

        # Draw the Y axis labels
        at,labels = self.yAxis()
        
        # Tickmarks
        draw = PIL.ImageDraw.Draw(pil)
        for i in range(len(at)):
            x = self.left
            y = self.top+self.y(at[i],bounds)
            draw.line((x,y,x-5,y),fill='black')
        del draw
        
        # Labels
        for i in range(len(at)):
            x = self.left
            y = self.top+self.y(at[i],bounds)
            drawLabel(pil, (x-5,y), labels[i], size='large',align='left',rotate=True,color='black')
        
        return pv.Image(pil)
            
    def xLabels(self,at,labels=None):
        at = np.array(at,np.float64)
        if labels != None:
            assert len(at) == len(labels)
            labels = np.array(labels,np.str)
        else:
            labels = np.array(at,np.str)
        self.x_at = at
        self.x_labels = labels
        
    def x(self,x,bounds):
        minx,maxx,_,_ = bounds
        w,_ = self.size
        return w*(x-minx)/float(maxx-minx)


    def y(self,y,bounds):
        _,_,miny,maxy = bounds
        _,h = self.size
        return h-h*(y-miny)/float(maxy-miny)
    
    def range(self):
        xmin,xmax,ymin,ymax = -1.0,1.0,-1.0,1.0
        if len(self.graphics) > 0:
            xmin,xmax,ymin,ymax = self.graphics[0].range()
            for each in self.graphics:
                t1,t2,t3,t4 = each.range()
                xmin = np.min([xmin,t1])
                xmax = np.max([xmax,t2])
                ymin = np.min([ymin,t3])
                ymax = np.max([ymax,t4])
        
        # Override default x_range
        if self.x_range != None:
            xmin,xmax = self.x_range

        # Override default y_range        
        if self.y_range != None:
            ymin,ymax = self.y_range
        
        #print "Range:", self.x_range
        #print "Range:", xmin,xmax,ymin,ymax
        if self.pad:
            rg = xmax-xmin
            xmin -= 0.05*rg
            xmax += 0.05*rg

            rg = ymax-ymin
            ymin -= 0.05*rg
            ymax += 0.05*rg
        
        return xmin,xmax,ymin,ymax
    
    
    def xAxis(self):
        minx,maxx,_,_ = self.range()
        w,_ = self.size
        if self.x_at == None:
            at,labels = self.autoAxis(minx,maxx,w)
        else:
            at,labels = self.x_at,self.x_labels
        return at,labels


    def yAxis(self):
        _,_,miny,maxy = self.range()
        _,h = self.size
        at,labels = self.autoAxis(miny,maxy,h)
        return at,labels
    
    def autoAxis(self,low,high,size):
        split = (0.1,0.2,0.5,1,2,5)
        if high == low:
            high = high+1
            low  = low -1
        rg = high - low
        l10 = np.log(rg)/np.log(10)
        f10 = np.floor(l10)
        best_at = []
        best_count = 500000000000000000000000 
        target_count = size/100
        scale = 10**f10
        low = low/scale
        high = high/scale

        for inc in split:
            tmp = np.arange(np.floor(low),high+0.001,inc)
            idx = (tmp >= low) & (tmp <= high)
            tmp = tmp[idx]
            count = len(tmp)
            if np.abs(count-target_count) < np.abs(target_count - best_count):
                best_count = count
                best_at = tmp*scale
                
        if f10 >= 1.0:
            label_format = "%0.0f"
            labels = [label_format%x for x in best_at]
        else:
            label_format = "%0." + "%d"%(-int(f10-1)) + "f"
            labels = [label_format%x for x in best_at]
        
        return best_at,labels
    
    def convertPoints(self,points):
        try:
            tmp = []
            for pt in points:
                tmp.append((pt.X(),pt.Y()))
            return tmp
        except:
            try:
                tmp = []
                for pt in points:
                    #print pt
                    tmp.append((float(pt[0]),float(pt[1])))
                return tmp
            except:
                raise
                #raise ValueError("Could not read points.")

    def label(self,point,label,**kwargs):
        ''' render a label at multiple points'''
        label = Label(point,label,**kwargs)
        self.graphics.append(label)
    
    def points(self,points,color='black',shape=0,size=3,label=None,lty=None,width=1):
        ''' render multiple points'''
        if len(points) < 1:
            return
        points = self.convertPoints(points)
        points = Points(points,'points',color=color,shape=shape,size=size,label=label,lty=lty,width=width)
        self.graphics.append(points)
    
    def point(self,point,color='black',shape=0,size=3,label=None,lty=None,width=1):
        ''' render a single point '''
        points = self.convertPoints([point])
        points = Points(points,'points',color=color,shape=shape,size=size,label=label,lty=lty,width=width)
        self.graphics.append(points)
    
    def lines(self,points,color='black',shape=None,size=3,label=None,lty=1,width=1):
        ''' render some lines '''
        if len(points) < 1:
            return
        points = self.convertPoints(points)
        points = Points(points,'lines',color=color,shape=shape,size=size,label=label,lty=lty,width=width)
        self.graphics.append(points)
        
    def polygon(self,points,color='black',shape=None,size=3,label=None,lty=1,width=1):
        ''' render a closed polygon '''
        if len(points) < 1:
            return
        points = self.convertPoints(points)
        points = Points(points,'polygon',color=color,shape=shape,size=size,label=label,lty=lty,width=width)
        self.graphics.append(points)
        
    def show(self,**kwargs):
        self.asImage().show(**kwargs)
        
    def asR(self,plot_pdf="out.pdf",run_r=True):
        '''
        Generate an R script that will reproduce this plot.
        '''
        # Create a file object for output
        f = io.StringIO()
        
        # Generate Configuration
        f.write("# This is an R script that will generate a plot.\n")
        f.write("# These first few lines are configuration options.\n")
        f.write("filename='%s';\n"%plot_pdf)
        f.write("plot_width=6; # width in inches\n")
        f.write("plot_height=6; # height in inches\n")
        f.write("title='%s'\n"%self.title)
        f.write("xlabel='%s'\n"%self.xlabel)
        f.write("ylabel='%s'\n"%self.ylabel)
        f.write("xrange=c(%f,%f)\n"%self.range()[:2])
        f.write("yrange=c(%f,%f)\n"%self.range()[2:])
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("\n")
        
        # Create the plot
        f.write("pdf(file=filename,width=plot_width,height=plot_height)\n")
        f.write("\n")
        f.write("par(mai=c(0.5,0.5,0.5,0.1)) # minimal inner margin\n")
        f.write("par(mgp=c(1.5,0.5,0.0)) # small axes label spacing\n")
        f.write("\n")
        f.write("plot(xrange,yrange,type='n',main=title,xlab=xlabel,ylab=ylabel)\n")
        f.write("# Log scale...\n")
        f.write("# Custom axis...\n")
        f.write("\n")
        
        # Generate stubs for common plot additions such as labels.
        f.write("\n")
        f.write("# axis(1,at=c(1,3,4),labels =c(1,3,4)) # xaxis.  Also add xaxt='n' to the plot command.\n")
        f.write("# axis(2,at=c(1,3,4),labels =c(1,3,4)) # yaxis.  Also add yaxt='n' to the plot command.\n")
        f.write("# legend('topright',c('Label 1','Label 2')),fill=c('blue','green'))\n")
        f.write("\n")
        
        # Render the data
        for each in self.graphics:
            f.write("# DRAWING: %s\n"%(each.__class__))
            each.drawR(f)
            f.write("\n")
            #print each

        # Finish up
        f.write("dev.off()\n")
        f.write("\n")
        f.write("\n")

        f.flush()
        
        if run_r:
            p_in,p_out = os.popen2("R --no-save")
            p_in.write(f.getvalue())
            p_in.close()
            print(p_out.read())
            p_out.close()
        
        return f.getvalue()
    
    
        
import unittest

class TestPlot(unittest.TestCase):
    
    def testAutoAxis(self):
        "Plot: Auto Axis"
        plot = Plot()
        plot.autoAxis(0.0,10.0,600)
        plot.autoAxis(-1.0,100.0,600)
        plot.autoAxis(-1.0,1.0,600)
        plot.autoAxis(-10.0,20.,600)
        plot.autoAxis(200.0,500.,600)
        plot.autoAxis(.210,.500,600)
        
    def testRangeZero(self):
        "Plot: Range = 0"
        plot = Plot(x_range=[0,0],y_range=[0,0])
        plot.asImage()
        
    def testNoData(self):
        "Plot: No Data"
        plot = Plot()
        plot.asImage()
        
        
    def testDataToFormatedList(self):
        print()
        print(dataToFormatedList(list(range(4))))
        print(dataToFormatedList(list(range(5))))
        print(dataToFormatedList(list(range(6))))
        print(dataToFormatedList(list(range(7))))
        print(dataToFormatedList(list(range(8))))
        print(dataToFormatedList(list(range(9))))
        print(dataToFormatedList(list(range(10))))
        print(dataToFormatedList(list(range(15))))
        print(dataToFormatedList(list(range(16))))
        print(dataToFormatedList(list(range(17))))
        print()

        
        
        

    
