import unittest
import numpy as np
from quad_to_uv import Quad

class TestQuadClass(unittest.TestCase):
    
    def test_initialization(self):
        """Test proper initialization of the Quad class."""
        quad_points = [(0, 0), (0, 100), (100, 100), (100, 0)]
        quad = Quad(quad_points)
        self.assertEqual(quad.quad_points, quad_points)
        
        # Test initialization with wrong number of points
        with self.assertRaises(ValueError):
            Quad([(0, 0), (0, 100), (100, 100)])
    
    def test_square_mapping(self):
        """Test mapping with a perfect square."""
        # Square from (0,0) to (100,100)
        quad_points = [(0, 0), (0, 100), (100, 100), (100, 0)]
        quad = Quad(quad_points)

        # Test corners (should map exactly to UV corners)
        self.assertAlmostEqual(quad.point_to_uv(0, 0), (0, 0))
        self.assertAlmostEqual(quad.point_to_uv(0, 100), (0, 1))
        self.assertAlmostEqual(quad.point_to_uv(100, 100), (1, 1))
        self.assertAlmostEqual(quad.point_to_uv(100, 0), (1, 0))

        # Test center
        self.assertAlmostEqual(quad.point_to_uv(50, 50), (0.5, 0.5))

        # Test midpoints of edges
        self.assertAlmostEqual(quad.point_to_uv(0, 50), (0, 0.5))
        self.assertAlmostEqual(quad.point_to_uv(50, 100), (0.5, 1))
        self.assertAlmostEqual(quad.point_to_uv(100, 50), (1, 0.5))
        self.assertAlmostEqual(quad.point_to_uv(50, 0), (0.5, 0))

    def test_rectangle_mapping(self):
        """Test mapping with a rectangle."""
        # Rectangle (0,0) to (200,100)
        quad_points = [(0, 0), (0, 100), (200, 100), (200, 0)]
        quad = Quad(quad_points)

        # Test corners
        self.assertAlmostEqual(quad.point_to_uv(0, 0), (0, 0))
        self.assertAlmostEqual(quad.point_to_uv(0, 100), (0, 1))
        self.assertAlmostEqual(quad.point_to_uv(200, 100), (1, 1))
        self.assertAlmostEqual(quad.point_to_uv(200, 0), (1, 0))

        # Test center
        self.assertAlmostEqual(quad.point_to_uv(100, 50), (0.5, 0.5))

    def test_arbitrary_quadrilateral(self):
        """Test mapping with an irregular quadrilateral."""
        quad_points = [(100, 100), (150, 300), (400, 350), (450, 50)]
        quad = Quad(quad_points)

        # Test corners
        self.assertAlmostEqual(quad.point_to_uv(100, 100), (0, 0))
        self.assertAlmostEqual(quad.point_to_uv(150, 300), (0, 1))
        self.assertAlmostEqual(quad.point_to_uv(400, 350), (1, 1))
        self.assertAlmostEqual(quad.point_to_uv(450, 50), (1, 0))

        # Test what should be approximately the center
        # For an irregular quad, this is a bit approximate
        center_x = (100 + 150 + 400 + 450) / 4
        center_y = (100 + 300 + 350 + 50) / 4
        uv = quad.point_to_uv(center_x, center_y)

        # The center of the quadrilateral shouldn't map exactly to (0.5, 0.5)
        # but it should be somewhere in the middle region
        self.assertTrue(0.3 < uv[0] < 0.7)
        self.assertTrue(0.3 < uv[1] < 0.7)

    def test_handling_collinear_points(self):
        """Test mapping with a degenerate quadrilateral (collinear points)."""
        # This quad has collinear points, which can cause numerical issues
        quad_points = [(0, 0), (0, 100), (100, 100), (50, 100)]  # Last two points along with second are collinear
        quad = Quad(quad_points)

        # This test may cause singular matrix depending on implementation
        # We're testing that the function either returns None or raises an appropriate error
        result = quad.point_to_uv(50, 50)
        # If we get a result, it should be None or a valid UV coordinate
        if result is not None:
            self.assertTrue(0 <= result[0] <= 1)
            self.assertTrue(0 <= result[1] <= 1)

    def test_point_outside_quad(self):
        """Test with a point outside the quadrilateral.

        The function should still return UV coordinates, but they'll be outside the 0-1 range.
        """
        quad_points = [(100, 100), (100, 300), (400, 300), (400, 100)]
        quad = Quad(quad_points)

        # Point well outside the quad
        uv = quad.point_to_uv(500, 500)

        # UV should be outside the unit square
        self.assertTrue(uv[0] > 1 or uv[0] < 0 or uv[1] > 1 or uv[1] < 0)

    def test_numerical_stability(self):
        """Test with very large coordinates to check numerical stability."""
        # Very large coordinates
        quad_points = [(1e6, 1e6), (1e6, 2e6), (2e6, 2e6), (2e6, 1e6)]
        quad = Quad(quad_points)

        # Center should map to (0.5, 0.5)
        uv = quad.point_to_uv(1.5e6, 1.5e6)
        self.assertAlmostEqual(uv[0], 0.5, places=5)
        self.assertAlmostEqual(uv[1], 0.5, places=5)

    def test_bilinear_interpolation_case(self):
        """Test a case where bilinear interpolation would be used in a simple case."""
        # Perfect square
        quad_points = [(0, 0), (0, 1), (1, 1), (1, 0)]
        quad = Quad(quad_points)

        # Test interpolated points along edges and diagonals
        for t in [0.25, 0.5, 0.75]:
            # Points along bottom edge
            self.assertAlmostEqual(quad.point_to_uv(t, 0), (t, 0))

            # Points along top edge
            self.assertAlmostEqual(quad.point_to_uv(t, 1), (t, 1))

            # Points along left edge
            self.assertAlmostEqual(quad.point_to_uv(0, t), (0, t))

            # Points along right edge
            self.assertAlmostEqual(quad.point_to_uv(1, t), (1, t))

            # Points along diagonal
            self.assertAlmostEqual(quad.point_to_uv(t, t), (t, t))

    def assertAlmostEqual(self, uv_result, expected_uv, places=5):
        """Custom assertion for comparing UV tuples with tolerance."""
        if uv_result is None:
            self.fail("UV result is None")

        u, v = uv_result
        expected_u, expected_v = expected_uv

        self.assertAlmostEqual(u, expected_u, places=places)
        self.assertAlmostEqual(v, expected_v, places=places)


if __name__ == "__main__":
    unittest.main()