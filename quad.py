import numpy as np
import cv2

class Quad:
    """
    A class representing a quadrilateral with methods to transform points to UV coordinates.

    The quadrilateral is defined by four points in counter-clockwise order,
    starting from top-left: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """

    def __init__(self, quad_points):
        """
        Initialize a Quad with four corner points.

        Args:
            quad_points (list): List of four (x,y) tuples defining the quadrilateral vertices
                               in counter-clockwise order, starting from top-left:
                               [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        if len(quad_points) != 4:
            raise ValueError("Quad must be initialized with exactly 4 points")

        self.quad_points = quad_points
        self.uv_coords = [(0,0), (0,1), (1,1), (1,0)]

        self.src_points = np.array(self.quad_points, dtype=np.float32)
        self.dst_points = np.array(self.uv_coords, dtype=np.float32)

        self._calculate_transform_matrix()

    def _calculate_transform_matrix(self):
        """Calculate the perspective transform matrix for the quad."""

        # Calculate the homography matrix using cv2.findHomography
        # RANSAC method helps eliminate outliers
        self.H, _ = cv2.findHomography(self.src_points, self.dst_points, cv2.RANSAC, 5.0)


    def point_to_uv(self, point_x, point_y):
        """
        Convert a point in image space to UV coordinates inside the quadrilateral.

        Args:
            point_x (float): X-coordinate of the point in image space
            point_y (float): Y-coordinate of the point in image space

        Returns:
            tuple: (u,v) coordinates of the point in UV space (0,0 to 1,1)
                  or None if the transform is invalid or division by zero occurs
        """
        if self.H is None:
            return None

        # Apply perspective transform to the point
        point = np.array([point_x, point_y, 1])
        result = self.H @ point

        # Normalize by dividing by w
        if result[2] != 0:
            u = result[0] / result[2]
            v = result[1] / result[2]
        else:
            # Handle division by zero edge case
            return None

        return (u, v)

    def uv_to_point(self, u, v):
        """
        Convert UV coordinates back to a point in image space.

        Args:
            u (float): U-coordinate in UV space (0 to 1)
            v (float): V-coordinate in UV space (0 to 1)

        Returns:
            tuple: (x,y) coordinates of the point in image space
                  or None if the transform is invalid or division by zero occurs
        """
        if self.H is None:
            return None

        # Invert the homography matrix
        try:
            H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            # Matrix is not invertible
            return None

        # Apply inverse transform to the UV point
        uv_point = np.array([u, v, 1])
        result = H_inv @ uv_point

        # Normalize by dividing by w
        if result[2] != 0:
            x = result[0] / result[2]
            y = result[1] / result[2]
        else:
            # Handle division by zero edge case
            return None

        return (x, y)
