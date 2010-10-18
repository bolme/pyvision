'''
A very simple 2D plotting package that outputs images.

@author: bolme
'''

import pyvision as pv
import numpy as np
import PIL
import os.path
import PIL.ImageFont
random = np.random

arial_path = os.path.join(pv.__path__[0],'config','Arial.ttf')
huge_font = PIL.ImageFont.truetype(arial_path, 36)
large_font = PIL.ImageFont.truetype(arial_path, 18)
small_font = PIL.ImageFont.truetype(arial_path, 12)


def drawLabel(buffer, pt, label, size='small',align='center',rotate=False,color='black'):
    '''
    @param buffer: a PIL image to use as a buffer
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
    draw = PIL.ImageDraw.Draw(buffer)
    draw.bitmap((x,y),im,fill=color)
    del draw
    #buffer.(color,(x,y,x+w,y+h),im)
    
    def range(self):
        x,y = self.point

        return x,x,y,y



class Label:
    def __init__(self,point,label,size='small',align='center',rotate=False,color='black'):
        self.point = point
        self.label = label
        self.size = size
        self.align = align
        self.rotate = rotate
        self.color = color
        
    def draw(self,plot,buffer):
        x,y = self.point
        x = plot.x(x)
        y = plot.y(y)
        drawLabel(buffer,[x,y],self.label,size=self.size,align=self.align,rotate=self.rotate,color=self.color)
        
    def range(self):
        x,y = self.point

        return x,x,y,y

        
        

class Points:
    def __init__(self,points,color='black',shape=0,size=2,label=None,lty=None,width=1):
        ''''''
        self.points = points
        self.color = color
        self.shape = shape
        self.size = size
        self.label = label
        self.width = width
        self.lty = lty
        
        
    def draw(self,plot,buffer):
        ''''''
        
        points = [ (plot.x(x),plot.y(y)) for x,y in self.points]
        
        if self.lty != None:
            self.drawCurve(points,buffer)

        if self.shape != None:
            self.drawPoints(points,buffer)


    def drawCurve(self,points,buffer):
        draw = PIL.ImageDraw.Draw(buffer)
        prev_x,prev_y = points[0]
        for x,y in points[1:]:
            draw.line([prev_x,prev_y,x,y],fill=self.color,width=self.width)
            prev_x = x
            prev_y = y
        del draw        
        
    def drawPoints(self,points,buffer):
        ''' Render points on the plot '''
        shape = self.shape
        size = self.size
        color = self.color

        draw = PIL.ImageDraw.Draw(buffer)
        
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
                    drawLabel(buffer, (x,y), shape, size='small',align='center',rotate=False,color=color)
                else:
                    drawLabel(buffer, (x,y), shape, size='large',align='center',rotate=False,color=color)
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
    def __init__(self,size=(600,600),xrange=None,yrange=None,title="No Title",ylabel="Y Axis",xlabel="X Axis",pad=True):
        # Save plot information
        self.size = size
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        
        self.xrange = xrange
        if xrange != None:
            self.xrange = (np.min(xrange),np.max(xrange))
        
        self.yrange = yrange
        if yrange != None:
            self.yrange = (np.min(yrange),np.max(yrange))
        
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
        
        buffer = PIL.Image.new("RGB",(w,h),'white')
        for each in self.graphics:
            each.draw(self,buffer)
            
        pil.paste(buffer,(self.left,self.top))
        
        draw = PIL.ImageDraw.Draw(pil)
        draw.rectangle((self.left,self.top,self.left+w,self.top+w),outline='black',fill=None)
        del draw
        
        # Draw the X axis labels
        at,labels = self.xAxis()
        
        # Tickmarks
        draw = PIL.ImageDraw.Draw(pil)
        for i in range(len(at)):
            x = self.left+self.x(at[i])
            y = self.top+h
            draw.line((x,y,x,y+5),fill='black')
        del draw
        
        # Labels
        for i in range(len(at)):
            x = self.left+self.x(at[i])
            y = self.top+h
            drawLabel(pil, (x,y+5), labels[i], size='large',align='bellow',rotate=False,color='black')

        # Draw the Y axis labels
        at,labels = self.yAxis()
        
        # Tickmarks
        draw = PIL.ImageDraw.Draw(pil)
        for i in range(len(at)):
            x = self.left
            y = self.top+self.y(at[i])
            draw.line((x,y,x-5,y),fill='black')
        del draw
        
        # Labels
        for i in range(len(at)):
            x = self.left
            y = self.top+self.y(at[i])
            drawLabel(pil, (x-5,y), labels[i], size='large',align='left',rotate=True,color='black')
        
        return pv.Image(pil)
            
        
    def x(self,x):
        minx,maxx,_,_ = self.range()
        w,_ = self.size
        return w*(x-minx)/float(maxx-minx)


    def y(self,y):
        _,_,miny,maxy = self.range()
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
        
        # Override default xrange
        if self.xrange != None:
            xmin,xmax = self.xrange

        # Override default yrange        
        if self.yrange != None:
            ymin,ymax = self.yrange
        
        #print "Range:", self.xrange
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
        at,labels = self.autoAxis(minx,maxx,w)
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
            format = "%0.0f"
            labels = [format%x for x in best_at]
        else:
            format = "%0." + "%d"%(-int(f10-1)) + "f"
            labels = [format%x for x in best_at]
        
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
                    tmp.append((float(pt[0]),float(pt[1])))
                return tmp
            except:
                #raise
                raise ValueError("Could not read points.")

    def label(self,point,label,**kwargs):
        ''' render multiple points'''
        label = Label(point,label,**kwargs)
        self.graphics.append(label)
    
    def points(self,points,**kwargs):
        ''' render multiple points'''
        points = self.convertPoints(points)
        points = Points(points,**kwargs)
        self.graphics.append(points)
    
    def point(self,point,color='black',shape=0,size=3,label=None,lty=None,width=1):
        ''' render a single point '''
        points = self.convertPoints([point])
        points = Points(points,color=color,shape=shape,size=size,label=label,lty=lty,width=width)
        self.graphics.append(points)
    
    def lines(self,points,color='black',shape=None,size=3,label=None,lty=1,width=1):
        ''' render a single point '''
        points = self.convertPoints(points)
        points = Points(points,color=color,shape=shape,size=size,label=label,lty=lty,width=width)
        self.graphics.append(points)
        
    def show(self):
        self.asImage().show()
    
    
        
import unittest

class TestPlot(unittest.TestCase):
    
    def testAutoAxis(self):
        "Plot: Auto Axis"
        plot = Plot()
        at = plot.autoAxis(0.0,10.0,600)
        #print at

        at = plot.autoAxis(-1.0,100.0,600)
        #print at

        at = plot.autoAxis(-1.0,1.0,600)
        #print at

        at = plot.autoAxis(-10.0,20.,600)
        #print at

        at = plot.autoAxis(200.0,500.,600)
        #print at
        
        at = plot.autoAxis(.210,.500,600)
        #print at
        
    def testRangeZero(self):
        "Plot: Range = 0"
        plot = Plot(xrange=[0,0],yrange=[0,0])
        plot.asImage()
        
    def testNoData(self):
        "Plot: No Data"
        plot = Plot()
        plot.asImage()
        
        

    
