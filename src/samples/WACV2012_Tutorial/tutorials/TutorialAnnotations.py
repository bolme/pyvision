import pyvision as pv
import scipy as sp

if __name__ == '__main__':
    im = pv.Image(sp.zeros((128,128)))

    pts = [pv.Point(48,55),pv.Point(80,55)]
    im.annotatePoints(pts)
    
    elipse = pv.CenteredRect(64,64,96,96)
    im.annotateEllipse(elipse)
    
    im.annotateLabel(pv.Point(40,36),"MMM")
    im.annotateLabel(pv.Point(72,36),"MMM")
    im.annotateLabel(pv.Point(58,64),"db")
    
    im.annotatePolygon([pv.Point(48,90),
        pv.Point(80,90),pv.Point(64,100)])
    
    im.show(delay=0)