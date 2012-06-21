import pyvision as pv
import pyvision.face.headdetector as hd

if __name__ == '__main__':
    
    detector = hd.HeadDetector()
    
    cam = pv.Webcam()
    for frame in cam:
        rects = detector(frame)

        for i in range(len(rects)):
            rect = rects[i]
            frame.annotateLabel(pv.Point(10,10+15*i),"%6.2f - %s"%(rect.score,rect.detector),)
        
        if len(rects) > 0:
            rect = rects[0]
            frame.annotatePolygon(rect.asPolygon(),width=3,color='red')
            rects = rects[1:]
        
        for rect in rects:
            frame.annotateRect(rect,color='yellow')
        
        frame.show(delay=30)