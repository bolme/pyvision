import pyvision as pv
import pyvision.face.CascadeDetector as cd
import pyvision.face.FilterEyeLocator as ed

face_detect = cd.CascadeDetector()
eye_detect = ed.FilterEyeLocator()

im = pv.Image("face.png",bw_annotate=True)

faces = face_detect(im)
eyes = eye_detect(im,faces)

for face,eye1,eye2 in eyes:
    im.annotatePolygon(face.asPolygon(),width=4)
    im.annotatePoints([eye1,eye2])
    
im.show(delay=0)

