import pyvision as pv
import pyvision.face.CascadeDetector as cd

if __name__ == '__main__':
    
    detector = cd.CascadeDetector()
    
    cam = pv.Webcam()
    while True:
        frame = cam.query()
        rects = detector(frame)
        for rect in rects:
            frame.annotateRect(rect)
        frame.show()