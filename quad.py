import numpy as np

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
        # Map quad corners to UV coordinates
        # UV coordinates for the quad corners in counter-clockwise order

        # We're solving for matrix M in the equation:
        # [u*w, v*w, w]^T = M * [x, y, 1]^T

        A = np.zeros((8, 8))
        b = np.zeros(8)

        for i in range(4):
            x, y = self.src_points[i]
            u, v = self.dst_points[i]

            A[i*2, 0] = x
            A[i*2, 1] = y
            A[i*2, 2] = 1
            A[i*2, 6] = -u * x
            A[i*2, 7] = -u * y
            b[i*2] = u

            A[i*2+1, 3] = x
            A[i*2+1, 4] = y
            A[i*2+1, 5] = 1
            A[i*2+1, 6] = -v * x
            A[i*2+1, 7] = -v * y
            b[i*2+1] = v

        try:
            # Solve the system of linear equations
            m = np.linalg.solve(A, b)
            M = np.vstack([m.reshape(8), [1]])
            self.H = M.reshape(3, 3)
        except np.linalg.LinAlgError:
            # Handle case where the matrix is singular (e.g., collinear points)
            self.H = None

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
