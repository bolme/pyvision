import pyvision as pv
from pyvision.edge.canny import canny  # An interface to the OpenCV Canny.
from scipy.ndimage import gaussian_filter # scipy gaussian filter function
from numpy import arange;

#This code is from part 2 of the PyVision Quick Start Guide.

if __name__ == '__main__':
    # Create the image log
    ilog = pv.ImageLog("../..")

    # Load an image from a file.
    im = pv.Image(pv.__path__[0]+"/data/nonface/NONFACE_16.jpg")
    
    # Rescale the image
    im = pv.AffineScale(0.5,(320,240)).transformImage(im)
    ilog.log(im,label="Source")    
    
    # Try a range of sigmas
    for sigma in arange(1.0,5.1,0.5):
        
        # Perform a Gaussian Blur
        mat = im.asMatrix2D()
        mat = gaussian_filter(mat,sigma)
        blur = pv.Image(mat)
        blur.annotateLabel(pv.Point(10,10),"Sigma: " + str(sigma))
        ilog.log(blur,label="Blur")    

        #Try a range of thresholds
        for thresh in arange(50,150,10):
    
            # Run the canny function with different defaults.
            edge = canny(blur,threshold1=thresh/2,threshold2=thresh)
            
            # Annotate the edge image
            edge.annotateLabel(pv.Point(10,10),"Sigma: " + str(sigma))
            edge.annotateLabel(pv.Point(10,20),"Thresh: " + str(thresh))
    
            # Save the results to a log.
            ilog.log(edge,label="Canny")
    
    # Display the results.
    ilog.show()
    
    
    