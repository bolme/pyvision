import unittest
import os
import pyvision as pv
import numpy as np
import weakref

# TODO: Add unit tests
class PerspectiveTest(unittest.TestCase):
    
    def setUp(self):
        fname_a = os.path.join(pv.__path__[0],'data','test','perspective1a.jpg')
        fname_b = os.path.join(pv.__path__[0],'data','test','perspective1b.jpg')
        
        self.im_a = pv.Image(fname_a)
        self.im_b = pv.Image(fname_b)
        
        #corners clockwize: upper left, upper right, lower right, lower left
        self.corners_a = (pv.Point(241,136),pv.Point(496,140),pv.Point(512,343),pv.Point(261,395))
        self.corners_b = (pv.Point(237,165),pv.Point(488,177),pv.Point(468,392),pv.Point(222,347))
        self.corners_t = (pv.Point(0,0),pv.Point(639,0),pv.Point(639,479),pv.Point(0,479))
        
        for pt in self.corners_a:
            self.im_a.annotatePoint(pt)

        #self.im_a.show()
        #self.im_b.show()
            
    def test_four_points_a(self):
        p = pv.PerspectiveFromPoints(self.corners_a,self.corners_t,(640,480))
        _ = p.transformPoints(self.corners_a)
        #for pt in pts:
        #    print "Point: %7.2f %7.2f"%(pt.X(), pt.Y())
            
        _ = p.transformImage(self.im_a)
        self.im_a.show()
        #_.show()

    def test_four_points_b(self):
        p = pv.PerspectiveFromPoints(self.corners_b,self.corners_t,(640,480))
        _ = p.transformPoints(self.corners_b)
        #for pt in pts:
        #    print "Point: %7.2f %7.2f"%(pt.X(), pt.Y())
            
        _ = p.transformImage(self.im_b)
        #_.show()
        
    def test_four_points_ab(self):
        p = pv.PerspectiveFromPoints(self.corners_a,self.corners_b,(640,480))
        #pts = p.transformPoints(self.corners_b)
        #for pt in pts:
        #    print "Point: %7.2f %7.2f"%(pt.X(), pt.Y())
            
        _ = p.transformImage(self.im_a)
        #_.show()
        #self.im_b.show()
        
# TODO: Add unit tests
class LogPolarTest(unittest.TestCase):
    
    def setUp(self):
        fname_a = os.path.join(pv.__path__[0],'data','test','perspective1a.jpg')
        fname_b = os.path.join(pv.__path__[0],'data','test','perspective1b.jpg')
        
        self.im_a = pv.Image(fname_a)
        self.im_b = pv.Image(fname_b)
        
        #corners clockwize: upper left, upper right, lower right, lower left
        self.corners_a = (pv.Point(241,136),pv.Point(496,140),pv.Point(512,343),pv.Point(261,395))
        self.corners_b = (pv.Point(237,165),pv.Point(488,177),pv.Point(468,392),pv.Point(222,347))
        self.corners_t = (pv.Point(0,0),pv.Point(639,0),pv.Point(639,479),pv.Point(0,479))
        
        for pt in self.corners_a:
            self.im_a.annotatePoint(pt)

        #self.im_a.show()
        #self.im_b.show()
            
    def testLogPolar(self):
        # Just run the code for now
        pv.logPolar(self.im_a)
        # TODO: add a test to make sure it worked.

        
# TODO: Add unit tests
class AffineTest(unittest.TestCase):
    
    def setUp(self):
        fname = os.path.join(pv.__path__[0],'data','nonface','NONFACE_13.jpg')
        self.test_image = pv.Image(fname)
        #self.test_image.show()
    
    def test_rotation(self):
        transform = pv.AffineRotate(3.14/8,(640,480))
        _ = transform.transformImage(self.test_image)
        # im_a.show()
        
        pt = transform.transformPoint(pv.Point(320,240))
        self.assertAlmostEqual(pt.X(),203.86594448424472)
        self.assertAlmostEqual(pt.Y(),344.14920700118842)

        pt = transform.invertPoint(pv.Point(320,240))
        self.assertAlmostEqual(pt.X(),387.46570317672939)
        self.assertAlmostEqual(pt.Y(),99.349528744542198)
        
    def test_scale(self):
        transform = pv.AffineScale(1.5,(640,480))
        _ = transform.transformImage(self.test_image)
        #im_a.show()
        
        pt = transform.transformPoint(pv.Point(320,240))
        self.assertAlmostEqual(pt.X(),480.)
        self.assertAlmostEqual(pt.Y(),360.)

        pt = transform.invertPoint(pv.Point(320,240))
        self.assertAlmostEqual(pt.X(),213.33333333333331)
        self.assertAlmostEqual(pt.Y(),160.)
        
    def test_translate(self):
        transform = pv.AffineTranslate(10.,15.,(640,480))
        _ = transform.transformImage(self.test_image)
        #im_a.show()
        
        pt = transform.transformPoint(pv.Point(320,240))
        self.assertAlmostEqual(pt.X(),330.)
        self.assertAlmostEqual(pt.Y(),255.)

        pt = transform.invertPoint(pv.Point(320,240))
        self.assertAlmostEqual(pt.X(),310.)
        self.assertAlmostEqual(pt.Y(),225.)
        
    def test_from_rect(self):
                
        transform = pv.AffineFromRect(pv.Rect(100,100,300,300),(100,100))
        _ = transform.transformImage(self.test_image)
        #im_a.show()
        
        pt = transform.transformPoint(pv.Point(320,240))
        self.assertAlmostEqual(pt.X(),73.333333333333329)
        self.assertAlmostEqual(pt.Y(),46.666666666666671)

        pt = transform.invertPoint(pv.Point(50.,50.))
        self.assertAlmostEqual(pt.X(),250.0)
        self.assertAlmostEqual(pt.Y(),250.0)
        
    def test_from_points(self):
        # TODO: Fix this test
        pass
        
    def test_sim_least_sqr(self):
        # TODO: Fix this test
        pass
        
    def test_affine_least_sqr(self):
        # TODO: Fix this test
        pass

    def test_affine_mul(self):
        # TODO: FIx this test
        pass
        
    def test_affine_Matrix2D(self):
        im = pv.Image(pv.BABOON)
        test_im = pv.Image(im.asMatrix2D())
        affine = pv.AffineFromRect(pv.CenteredRect(256,256,128,128),(64,64))

        # Transform the images
        im = affine(im)
        test_im = affine(test_im)

        # Correlate the resulting images
        vec1 = pv.unit(im.asMatrix2D().flatten())
        vec2 = pv.unit(test_im.asMatrix2D().flatten())
        score = np.dot(vec1,vec2)
        
        self.assertGreater(score, 0.998)


    def test_affine_OpenCV2BW(self):
        im = pv.Image(pv.BABOON)
        test_im = pv.Image(im.asOpenCV2BW())
        affine = pv.AffineFromRect(pv.CenteredRect(256,256,128,128),(64,64))

        # Transform the images
        im = affine(im)
        test_im = affine(test_im)

        # Correlate the resulting images
        vec1 = pv.unit(im.asMatrix2D().flatten())
        vec2 = pv.unit(test_im.asMatrix2D().flatten())
        score = np.dot(vec1,vec2)
        
        self.assertGreater(score, 0.998)


    def test_affine_OpenCV2(self):
        im = pv.Image(pv.BABOON)
        test_im = pv.Image(im.asOpenCV2())
        affine = pv.AffineFromRect(pv.CenteredRect(256,256,128,128),(64,64))

        # Transform the images
        im = affine(im)
        test_im = affine(test_im)

        # Correlate the resulting images
        vec1 = pv.unit(im.asMatrix3D().flatten())
        vec2 = pv.unit(test_im.asMatrix3D().flatten())
        score = np.dot(vec1,vec2)
        
        self.assertGreater(score, 0.998)


    def test_affine_Matrix3D(self):
        im = pv.Image(pv.BABOON)
        test_im = pv.Image(im.asMatrix3D())
        affine = pv.AffineFromRect(pv.CenteredRect(256,256,128,128),(64,64))

        # Transform the images
        im = affine(im)
        test_im = affine(test_im)

        # Correlate the resulting images
        vec1 = pv.unit(im.asMatrix3D().flatten())
        vec2 = pv.unit(test_im.asMatrix3D().flatten())
        score = np.dot(vec1,vec2)
        
        self.assertGreater(score, 0.998)


    def test_affine_opencv(self):
        # TODO: FIx this test
        pass
        
    def test_prev_ref1(self):
        fname = os.path.join(pv.__path__[0],'data','nonface','NONFACE_13.jpg')
        im_a = pv.Image(fname)
        ref  = weakref.ref(im_a)

        self.assertEqual(ref(), im_a)
        
        tmp = im_a
        del im_a
        
        self.assertEqual(ref(), tmp)
        
        del tmp
        
        self.assertEqual(ref(), None)
        
 
    def test_prev_ref2(self):
        fname = os.path.join(pv.__path__[0],'data','nonface','NONFACE_13.jpg')
        im_a = pv.Image(fname)
        #im_a.show()
        w,h = im_a.size
        
        # Try scaling down and then scaling back up
        tmp1 = pv.AffineScale(0.1,(w/10,h/10)).transformImage(im_a)
        #tmp1.show()
        
        tmp2 = pv.AffineScale(10.0,(w,h)).transformImage(tmp1,use_orig=False)
        tmp2.annotateLabel(pv.Point(10,10), "This image should be blurry.")
        #tmp2.show()
       
        tmp3 = pv.AffineScale(10.0,(w,h)).transformImage(tmp1,use_orig=True)
        tmp3.annotateLabel(pv.Point(10,10), "This image should be sharp.")
        #tmp3.show()
        
        del im_a
        
        tmp4 = pv.AffineScale(10.0,(w,h)).transformImage(tmp1,use_orig=True)
        tmp4.annotateLabel(pv.Point(10,10), "This image should be blurry.")
        #tmp4.show()
        
    def test_prev_ref3(self):
        fname = os.path.join(pv.__path__[0],'data','nonface','NONFACE_13.jpg')
        torig = tprev = im_a = pv.Image(fname)
        #im_a.show()
        w,h = im_a.size
        
        # Scale
        aff = pv.AffineScale(0.5,(w/2,h/2))
        accu = aff
        torig = aff.transformImage(torig)
        tprev = aff.transformImage(tprev,use_orig=False)
        taccu = accu.transformImage(im_a)
        
        torig.annotateLabel(pv.Point(10,10), "use_orig = True")
        tprev.annotateLabel(pv.Point(10,10), "use_orig = False")
        taccu.annotateLabel(pv.Point(10,10), "accumulated")
        
        #torig.show()
        #tprev.show()
        #taccu.show()
        
        # Translate
        aff = pv.AffineTranslate(20,20,(w/2,h/2))
        accu = aff*accu
        torig = aff.transformImage(torig)
        tprev = aff.transformImage(tprev,use_orig=False)
        taccu = accu.transformImage(im_a)
        
        torig.annotateLabel(pv.Point(10,10), "use_orig = True")
        tprev.annotateLabel(pv.Point(10,10), "use_orig = False")
        taccu.annotateLabel(pv.Point(10,10), "accumulated")
        
        #torig.show()
        #tprev.show()
        #taccu.show()
        
        
        # Rotate
        aff = pv.AffineRotate(np.pi/4,(w/2,h/2))
        accu = aff*accu
        torig = aff.transformImage(torig)
        tprev = aff.transformImage(tprev,use_orig=False)
        taccu = accu.transformImage(im_a)
        
        torig.annotateLabel(pv.Point(10,10), "use_orig = True")
        tprev.annotateLabel(pv.Point(10,10), "use_orig = False")
        taccu.annotateLabel(pv.Point(10,10), "accumulated")
        
        #torig.show()
        #tprev.show()
        #taccu.show()
        
        
        
        # Translate
        aff = pv.AffineTranslate(100,-10,(w/2,h/2))
        accu = aff*accu
        torig = aff.transformImage(torig)
        tprev = aff.transformImage(tprev,use_orig=False)
        taccu = accu.transformImage(im_a)
        
        torig.annotateLabel(pv.Point(10,10), "use_orig = True")
        tprev.annotateLabel(pv.Point(10,10), "use_orig = False")
        taccu.annotateLabel(pv.Point(10,10), "accumulated")
        
        #torig.show()
        #tprev.show()
        #taccu.show()
        
        
        # Scale
        aff = pv.AffineScale(2.0,(w,h))
        accu = aff*accu
        torig = aff.transformImage(torig)
        tprev = aff.transformImage(tprev,use_orig=False)
        taccu = accu.transformImage(im_a)
        
        torig.annotateLabel(pv.Point(10,10), "use_orig = True")
        tprev.annotateLabel(pv.Point(10,10), "use_orig = False")
        taccu.annotateLabel(pv.Point(10,10), "accumulated")
        
        #torig.show()
        #tprev.show()
        #taccu.show()
        
