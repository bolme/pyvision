import pyvision as pv
from pyvision.edge.canny import canny  # An interface to the OpenCV Canny.

'''
This code is from part 1 of the PyVision Quick Start Guide.
'''
if __name__ == '__main__':
    # (1) Load an image from a file.
    im = pv.Image(pv.__path__[0]+"/data/nonface/NONFACE_16.jpg")
    
    # (2) Rescale the image
    im = pv.AffineScale(0.5,(320,240)).transformImage(im)
    
    # (3) Run the canny function to locate the edges.
    edge_im1 = canny(im)
    
    # (4) Run the canny function with different defaults.
    edge_im2 = canny(im,threshold1=100,threshold2=250)
    
    # (5) Save the results to a log.
    ilog = pv.ImageLog("../..")
    ilog.log(im,label="Source")    
    ilog.log(edge_im1,label="Canny1")
    ilog.log(edge_im2,label="Canny2")
    
    # (6) Display the results.
    ilog.show()
    
    
    