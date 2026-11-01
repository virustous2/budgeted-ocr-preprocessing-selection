import unittest
import cv2
import numpy as np
from measure_vn_projection_skew import angle_grid,estimate


class SkewTests(unittest.TestCase):
    def test_grid(self):
        grid=angle_grid();self.assertEqual(len(grid),201)
        self.assertAlmostEqual(grid[100],0)
        with self.assertRaises(ValueError):angle_grid(5,.3)

    def test_rotated_lines(self):
        image=np.full((260,380),255,np.uint8)
        for y in range(40,220,25):
            for x in range(50,320,35):cv2.rectangle(image,(x,y),(x+22,y+5),0,-1)
        grid=angle_grid(3,.1)
        for angle in [-1.4,0,1.4]:
            m=cv2.getRotationMatrix2D((190,130),angle,1)
            rotated=cv2.warpAffine(image,m,(380,260),borderValue=255)
            correction,_,flags=estimate(rotated,grid)
            self.assertLessEqual(abs(correction+angle),.15)
            self.assertFalse(flags['boundary_maximum'])

    def test_blank(self):
        correction,_,flags=estimate(np.full((80,120),255,np.uint8),angle_grid())
        self.assertEqual(correction,0)
        self.assertTrue(flags['flat_objective'])


if __name__=='__main__':unittest.main()
