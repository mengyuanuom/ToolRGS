import unittest
import cv2
import numpy as np
from skimage.draw import polygon
from utils.grasp_raster import rectangle_iou, grasp_matches, angle_distance


def reference_pixels(r, shape):
    b=cv2.boxPoints(((float(r[0]),float(r[1])),(float(r[2]),float(r[3])),-float(r[4]))).astype(np.int64)
    y,x=polygon(b[:,1],b[:,0],shape)
    return set(zip(y,x))


class RasterTest(unittest.TestCase):
    def test_right_side(self):
        r=[1000.,360.,80.,20.,0.]
        self.assertEqual(rectangle_iou(r,r),1.)

    def test_periodic(self):
        self.assertEqual(angle_distance(89,-89),2.)
        self.assertEqual(angle_distance(45,-45),90.)

    def test_width_cap_explicit(self):
        p=[[1000.,360.,300.,20.,0.]]
        g=[[1000.,360.,300.,20.,0.,0.]]
        self.assertEqual(grasp_matches(p,g,300)[0][1],1.)
        self.assertLess(grasp_matches(p,g,100)[0][1],.4)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            rectangle_iou([1,1,20,20,float('nan')],[1,1,20,20,0])

    def test_exact_pixel_reference(self):
        rng=np.random.default_rng(20260912)
        for _ in range(250):
            shape=(int(rng.integers(40,180)),int(rng.integers(100,250)))
            p=np.array([rng.uniform(-10,shape[1]+10),rng.uniform(-10,shape[0]+10),rng.uniform(1,100),rng.uniform(1,30),rng.uniform(-90,90)])
            t=p.copy();t[:2]+=rng.uniform(-20,20,2);t[4]+=rng.uniform(-30,30)
            a,b=reference_pixels(p,shape),reference_pixels(t,shape)
            ref=len(a&b)/len(a|b) if a|b else 0.
            self.assertAlmostEqual(rectangle_iou(p,t,shape,float('inf')),ref,12)


if __name__=='__main__':unittest.main()
