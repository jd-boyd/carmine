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
        self.assertTrue(hasattr(Quad, "__init__"))
        self.assertTrue(hasattr(Quad, "point_to_uv"))
        self.assertTrue(hasattr(Quad, "_calculate_transform_matrix"))

    def test_homography_creation(self):
        """Test that the homography matrix is created correctly."""
        # Simple square

        quad_points = [(0, 0), (0, 100), (100, 100), (100, 0)]
        quad = Quad(quad_points)

        # Check that H matrix exists and has correct shape
        self.assertIsNotNone(quad.H)
        self.assertEqual(quad.H.shape, (3, 3))

    def test_sanity_img_to_uv(self):
        quad_points = [(9, 402), (1217, 267), (1890, 304), (1068, 862)]
        quad = Quad(quad_points)

        def pairAssert(p1, p2):
            self.assertAlmostEqual(p1[0], p2[0], delta=0.1)
            self.assertAlmostEqual(p1[1], p2[1], delta=0.1)

        # Test with a delta since floating point calculations might not be exact
        for i, uv in enumerate([(0, 0), (0, 1), (1, 1), (1, 0)]):
            result = quad.point_to_uv(*quad_points[i])
            pairAssert(result, uv)

    def test_sanity_uv_img(self):
        quad_points = [(9, 402), (1217, 267), (1890, 304), (1068, 862)]
        quad = Quad(quad_points)

        def pairAssert(p1, p2):
            self.assertAlmostEqual(p1[0], p2[0], delta=0.1)
            self.assertAlmostEqual(p1[1], p2[1], delta=0.1)

        # Test with a delta since floating point calculations might not be exact
        for i, uv in enumerate([(0, 0), (0, 1), (1, 1), (1, 0)]):
            result = quad.uv_to_point(*uv)
            pairAssert(result, quad_points[i])

    def test_sanity_uv_field(self):
        quad_points = [(9, 0), (0, 300), (160, 300), (160, 0)]
        quad = Quad(quad_points)

        def pairAssert(p1, p2):
            self.assertAlmostEqual(p1[0], p2[0], delta=0.1)
            self.assertAlmostEqual(p1[1], p2[1], delta=0.1)

        # Test with a delta since floating point calculations might not be exact
        for i, uv in enumerate([(0, 0), (0, 1), (1, 1), (1, 0)]):
            result = quad.uv_to_point(*uv)
            pairAssert(result, quad_points[i])

    def test_sanity_field_uv(self):
        quad_points = [(9, 0), (0, 300), (160, 300), (160, 0)]
        quad = Quad(quad_points)

        def pairAssert(p1, p2):
            self.assertAlmostEqual(p1[0], p2[0], delta=0.1)
            self.assertAlmostEqual(p1[1], p2[1], delta=0.1)

        # Test with a delta since floating point calculations might not be exact
        uv_points = [(0, 0), (0, 1), (1, 1), (1, 0)]
        for i in range(4):
            result = quad.point_to_uv(*quad_points[i])
            pairAssert(result, uv_points[i])

    def test_uv_to_point_roundtrip(self):
        # Test with a simple square
        quad_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        quad = Quad(quad_points)

        # Test the four corners
        for i, uv in enumerate([(0, 0), (0, 1), (1, 1), (1, 0)]):
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

    def test_point_to_field(self):
        # Test with default field size (100x300)
        quad_points = [(0, 0), (0, 100), (100, 100), (100, 0)]
        quad = Quad(quad_points)

        # Test the four corners
        expected_field_corners = [(0, 0), (0, 300), (100, 300), (100, 0)]

        for i in range(4):
            result = quad.point_to_field(*quad_points[i])
            if result is not None:
                self.assertAlmostEqual(
                    result[0], expected_field_corners[i][0], delta=1.0
                )
                self.assertAlmostEqual(
                    result[1], expected_field_corners[i][1], delta=1.0
                )

        # Test middle point
        middle_point = (50, 50)
        expected_field_middle = (50, 150)  # middle of 100x300 field

        result = quad.point_to_field(*middle_point)
        if result is not None:
            self.assertAlmostEqual(result[0], expected_field_middle[0], delta=1.0)
            self.assertAlmostEqual(result[1], expected_field_middle[1], delta=1.0)

    def test_field_to_point(self):
        # Test with default field size (100x300)
        quad_points = [(0, 0), (0, 100), (100, 100), (100, 0)]
        quad = Quad(quad_points)

        # Test field corners mapping to quad corners
        field_corners = [(0, 0), (0, 300), (100, 300), (100, 0)]

        for i in range(4):
            result = quad.field_to_point(*field_corners[i])
            if result is not None:
                self.assertAlmostEqual(result[0], quad_points[i][0], delta=1.0)
                self.assertAlmostEqual(result[1], quad_points[i][1], delta=1.0)

        # Test middle of field
        field_middle = (50, 150)
        expected_point = (50, 50)

        result = quad.field_to_point(*field_middle)
        if result is not None:
            self.assertAlmostEqual(result[0], expected_point[0], delta=1.0)
            self.assertAlmostEqual(result[1], expected_point[1], delta=1.0)

    def test_field_point_roundtrip(self):
        # Test that converting to field and back preserves the original point
        quad_points = [(10, 30), (50, 200), (250, 180), (300, 20)]
        quad = Quad(quad_points)

        # Test some sample points
        test_points = [
            (100, 100),  # middle-ish
            (50, 50),  # upper left
            (200, 150),  # lower right
            (75, 175),  # random point
        ]

        for point in test_points:
            # Point -> Field -> Point roundtrip
            field_coords = quad.point_to_field(*point)
            if field_coords is not None:
                round_trip = quad.field_to_point(*field_coords)
                if round_trip is not None:
                    self.assertAlmostEqual(round_trip[0], point[0], delta=1.0)
                    self.assertAlmostEqual(round_trip[1], point[1], delta=1.0)

    def test_custom_field_size(self):
        # Test with a custom field size
        quad_points = [(0, 0), (0, 100), (100, 100), (100, 0)]
        custom_field_size = [200, 400]
        quad = Quad(quad_points, field_size=custom_field_size)

        # Test a corner
        corner_point = quad_points[2]  # bottom right
        expected_field_corner = (200, 400)  # bottom right of field

        result = quad.point_to_field(*corner_point)
        if result is not None:
            self.assertAlmostEqual(result[0], expected_field_corner[0], delta=1.0)
            self.assertAlmostEqual(result[1], expected_field_corner[1], delta=1.0)

        # Test the reverse
        reverse_result = quad.field_to_point(*expected_field_corner)
        if reverse_result is not None:
            self.assertAlmostEqual(reverse_result[0], corner_point[0], delta=1.0)
            self.assertAlmostEqual(reverse_result[1], corner_point[1], delta=1.0)


if __name__ == "__main__":
    unittest.main()
