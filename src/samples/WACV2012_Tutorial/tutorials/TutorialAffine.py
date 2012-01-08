import pyvision as pv

if __name__ == '__main__':
    im = pv.Image("face.png")
    eye1,eye2 = pv.Point(140,165),pv.Point(212,165)
    out1,out2 = pv.Point(64,128),pv.Point(192,128)
    
    im.annotatePoints([eye1,eye2])
    im.show(delay=0)
    
    affine = pv.AffineFromPoints(eye1,eye2,
                    out1,out2,(256,320))
    tile = affine(im)
    tile.show(delay=0)
    
    affine = pv.AffineRotate(3.1415,(256,320),
            center=pv.Point(128,160))*affine;
    tile = affine(im)
    tile.show(delay=0)
