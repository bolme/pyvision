import pyvision as pv
import PIL, cv
ilog = pv.ImageLog()

im = pv.Image("baboon.jpg")

# In PIL
pil = im.asPIL()
gray = pil.convert('L')
thresh = PIL.Image.eval(gray, lambda x: 255*(x>127.5) )   
ilog(pv.Image(thresh),"PILThresh")

#in Scipy
mat = im.asMatrix2D()    
thresh = mat > 127.5
ilog(pv.Image(1.0*thresh),"ScipyThresh")

#in OpenCV
cvim = im.asOpenCVBW()
dest = cv.CreateImage(im.size,cv.IPL_DEPTH_8U,1)
cv.CmpS(cvim,127.5,dest,cv.CV_CMP_GT)
ilog(pv.Image(dest),"OpenCVThresh")

ilog.show()