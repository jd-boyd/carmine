import unittest
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quad import Quad

class TestQuadClass(unittest.TestCase):

    def test_initialization(self):
        """Test proper initialization of the Quad class."""
        quad_points = [(0, 0), (0, 100), (100, 100), (100, 0)]
        quad = Quad(quad_points)
        self.assertEqual(quad.quad_points, quad_points)

        # Test initialization with wrong number of points
        with self.assertRaises(ValueError):
            Quad([(0, 0), (0, 100), (100, 100)])


    def test_quad_exists(self):
        """Test that the Quad class exists and has the required methods."""
        # Check that class and methods exist
        self.assertTrue(hasattr(Quad, '__init__'))
        self.assertTrue(hasattr(Quad, 'point_to_uv'))
        self.assertTrue(hasattr(Quad, '_calculate_transform_matrix'))


    def test_homography_creation(self):
        """Test that the homography matrix is created correctly."""
        # Simple square

        quad_points = [(0, 0), (0, 100), (100, 100), (100, 0)]
        quad = Quad(quad_points)

        # Check that H matrix exists and has correct shape
        self.assertIsNotNone(quad.H)
        self.assertEqual(quad.H.shape, (3, 3))

    def test_sanity_img_to_uv(self):
        quad_points = [
            (9, 402),
            (1217, 267),
            (1890, 304),
            (1068, 862)
        ]
        quad = Quad(quad_points)

        def pairAssert(p1, p2):
            self.assertAlmostEqual(p1[0], p2[0], delta=0.1)
            self.assertAlmostEqual(p1[1], p2[1], delta=0.1)

        # Test with a delta since floating point calculations might not be exact
        for i, uv in enumerate([(0,0), (0,1), (1,1), (1,0)]):
            result = quad.point_to_uv(*quad_points[i])
            pairAssert(result, uv)

    def test_sanity_uv_img(self):
        quad_points = [
            (9, 402),
            (1217, 267),
            (1890, 304),
            (1068, 862)
        ]
        quad = Quad(quad_points)

        def pairAssert(p1, p2):
            self.assertAlmostEqual(p1[0], p2[0], delta=0.1)
            self.assertAlmostEqual(p1[1], p2[1], delta=0.1)

        # Test with a delta since floating point calculations might not be exact
        for i, uv in enumerate([(0,0), (0,1), (1,1), (1,0)]):
            result = quad.uv_to_point(*uv)
            pairAssert(result, quad_points[i])

    def test_sanity_uv_field(self):
        quad_points = [
            (9, 0),
            (0, 300),
            (160, 300),
            (160, 0)
        ]
        quad = Quad(quad_points)

        def pairAssert(p1, p2):
            self.assertAlmostEqual(p1[0], p2[0], delta=0.1)
            self.assertAlmostEqual(p1[1], p2[1], delta=0.1)

        # Test with a delta since floating point calculations might not be exact
        for i, uv in enumerate([(0,0), (0,1), (1,1), (1,0)]):
            result = quad.uv_to_point(*uv)
            pairAssert(result, quad_points[i])


    def test_sanity_field_uv(self):
        quad_points = [
            (9, 0),
            (0, 300),
            (160, 300),
            (160, 0)
        ]
        quad = Quad(quad_points)

        def pairAssert(p1, p2):
            self.assertAlmostEqual(p1[0], p2[0], delta=0.1)
            self.assertAlmostEqual(p1[1], p2[1], delta=0.1)

        # Test with a delta since floating point calculations might not be exact
        uv_points = [(0,0), (0,1), (1,1), (1,0)]
        for i in range(4):
            result = quad.point_to_uv(*quad_points[i])
            pairAssert(result, uv_points[i])


    def test_uv_to_point_roundtrip(self):
        # Test with a simple square
        quad_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        quad = Quad(quad_points)

        # Test the four corners
        for i, uv in enumerate([(0,0), (0,1), (1,1), (1,0)]):
            # Convert from image to UV
            result = quad.point_to_uv(*quad_points[i])
            # Should be approximately the expected UV
            if result is not None:
                self.assertAlmostEqual(result[0], uv[0], delta=0.1)
                self.assertAlmostEqual(result[1], uv[1], delta=0.1)

            # Convert back from UV to image
            reverse_result = quad.uv_to_point(*uv)
            # Should get back approximately the original point
            if reverse_result is not None:
                self.assertAlmostEqual(reverse_result[0], quad_points[i][0], delta=0.1)
                self.assertAlmostEqual(reverse_result[1], quad_points[i][1], delta=0.1)

        # Test a point in the middle
        center_point = (50, 50)
        center_uv = (0.5, 0.5)

        # Test image -> UV
        result = quad.point_to_uv(*center_point)
        if result is not None:
            self.assertAlmostEqual(result[0], center_uv[0], delta=0.1)
            self.assertAlmostEqual(result[1], center_uv[1], delta=0.1)

        # Test UV -> image
        reverse_result = quad.uv_to_point(*center_uv)
        if reverse_result is not None:
            self.assertAlmostEqual(reverse_result[0], center_point[0], delta=0.1)
            self.assertAlmostEqual(reverse_result[1], center_point[1], delta=0.1)



if __name__ == "__main__":
    unittest.main()
