import json
import os
import numpy as np
from quad import Quad

# Constants
CONFIG_FILE = "carmine_config.json"

class State:
    """
    Class to hold application state, separate from UI rendering.
    """
    def __init__(self, camera_list):
        self.camera_list = camera_list

        # Camera selection
        self.selected_camera1 = 0
        self.selected_camera2 = 0

        # Camera points
        self.camera1_points = [[0, 0] for _ in range(4)]  # 4 points (x,y) for camera 1
        self.camera2_points = [[0, 0] for _ in range(4)]  # 4 points (x,y) for camera 2

        # Point selection state
        self.waiting_for_camera1_point = -1  # Index of point we're waiting to set (-1 means not waiting)
        self.waiting_for_camera2_point = -1  # Index of point we're waiting to set (-1 means not waiting)
        self.waiting_for_poi_point = -1      # Index of POI we're waiting to set (-1 means not waiting)

        # Highlighted car tracking
        self.highlighted_car = None  # Will store [x, y, width, height, confidence] of highlighted car in video coordinates
        self.car_field_position = None  # Will store (x, y) of the car position on the field (normalized coordinates)

        # Field dimensions (aspect ratio)
        self.field_size = [160, 300]  # [width, height]

        # POI positions (normalized 0.0-1.0 coordinates on the field)
        self.poi_positions = [
            (0.2, 0.3),  # Example position for POI 1
            (0.5, 0.5),  # Example position for POI 2
            (0.8, 0.7),  # Example position for POI 3
            (0.3, 0.8),  # Example position for POI 4
            (0.7, 0.2),  # Example position for POI 5
            (0.1, 0.9),  # Example position for POI 6
            (0.9, 0.1),  # Example position for POI 7
            (0.4, 0.6),  # Example position for POI 8
            (0.6, 0.4),  # Example position for POI 9
            (0.5, 0.8),  # Example position for POI 10
        ]

        # Load configuration if exists
        self.load_config()

    def set_camera_point(self, camera_num, point_index, x, y):
        """Set a camera point to the given coordinates"""
        if camera_num == 1:
            self.camera1_points[point_index] = [x, y]
            self.waiting_for_camera1_point = -1  # Reset waiting state
        elif camera_num == 2:
            self.camera2_points[point_index] = [x, y]
            self.waiting_for_camera2_point = -1  # Reset waiting state

        # Save configuration after updating points
        self.save_config()

    def get_camera1_id(self):
        """Get the selected camera1 ID"""
        if self.selected_camera1 < len(self.camera_list):
            return self.camera_list[self.selected_camera1][0]
        return None

    def get_camera2_id(self):
        """Get the selected camera2 ID"""
        if self.selected_camera2 < len(self.camera_list):
            return self.camera_list[self.selected_camera2][0]
        return None

    def set_poi_position(self, index, x, y):
        """Set a POI position to the given normalized coordinates (0-1)"""
        if 0 <= index < len(self.poi_positions):
            self.poi_positions[index] = (x, y)
            self.waiting_for_poi_point = -1  # Reset waiting state
            self.save_config()

    def camera_to_field_position(self, camera_x, camera_y, camera_num=1):
        """
        Convert a camera point (in pixel coordinates) to field position (normalized coordinates)

        Args:
            camera_x: X coordinate in the camera frame
            camera_y: Y coordinate in the camera frame
            camera_num: Camera number (1 or 2)

        Returns:
            tuple: (x, y) normalized field coordinates
        """
        if camera_num == 1:
            camera_points = self.camera1_points
        else:
            camera_points = self.camera2_points

        # Check if we have valid quad points
        if not all(isinstance(p, list) and len(p) == 2 for p in camera_points):
            print("Invalid camera points for field position calculation")
            return None

        try:
            # Create a quad from the camera points
            quad = Quad(camera_points)

            # Convert to UV coordinates (normalized 0-1 coordinates)
            uv = quad.point_to_uv(camera_x, camera_y)

            if uv is None:
                print("Could not calculate field position - invalid transformation")
                return None

            # Return the UV coordinates as our field position
            return uv
        except Exception as e:
            print(f"Error converting camera to field position: {e}")
            return None

    def highlight_car(self, car_detection):
        """
        Highlight a car and calculate its position on the field

        Args:
            car_detection: [x1, y1, x2, y2, conf, cls_id] car detection
        """
        if car_detection is None:
            self.highlighted_car = None
            self.car_field_position = None
            return

        # Store the car detection
        self.highlighted_car = car_detection

        # Calculate the center of the car
        x1, y1, x2, y2, _, _ = car_detection
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Convert to field position
        self.car_field_position = self.camera_to_field_position(center_x, center_y)

    def save_config(self):
        """Save the current configuration to a JSON file"""
        try:
            config = {
                'selected_camera1': self.selected_camera1,
                'selected_camera2': self.selected_camera2,
                'camera1_points': self.camera1_points,
                'camera2_points': self.camera2_points,
                'field_size': self.field_size,
                'poi_positions': self.poi_positions
            }

            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Configuration saved to {CONFIG_FILE}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def load_config(self):
        """Load configuration from a JSON file if it exists"""
        try:
            if not os.path.exists(CONFIG_FILE):
                print(f"No configuration file found at {CONFIG_FILE}")
                return

            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)

            # Load camera selection
            if 'selected_camera1' in config:
                self.selected_camera1 = config['selected_camera1']
            if 'selected_camera2' in config:
                self.selected_camera2 = config['selected_camera2']

            # Load camera points
            if 'camera1_points' in config:
                self.camera1_points = config['camera1_points']
            if 'camera2_points' in config:
                self.camera2_points = config['camera2_points']

            # Load field size
            if 'field_size' in config:
                self.field_size = config['field_size']

            # Load POI positions
            if 'poi_positions' in config:
                self.poi_positions = config['poi_positions']

            print(f"Configuration loaded from {CONFIG_FILE}")
        except Exception as e:
            print(f"Error loading configuration: {e}")

    def reset_config(self):
        """Reset configuration to default values"""
        # Reset camera points
        self.camera1_points = [[0, 0] for _ in range(4)]
        self.camera2_points = [[0, 0] for _ in range(4)]

        # Reset field size
        self.field_size = [160, 300]

        # Reset POI positions to defaults
        self.poi_positions = [
            (0.2, 0.3), (0.5, 0.5), (0.8, 0.7), (0.3, 0.8), (0.7, 0.2),
            (0.1, 0.9), (0.9, 0.1), (0.4, 0.6), (0.6, 0.4), (0.5, 0.8)
        ]

        # Save the reset configuration
        self.save_config()
        print("Configuration reset to defaults")

    def calculate_poi_distances(self):
        """
        Calculate distances between the car and each point of interest.

        Returns:
            list: List of (poi_index, distance) tuples sorted by distance,
                  or None if car position is not available
        """
        if self.car_field_position is None:
            return None

        car_x, car_y = self.car_field_position
        distances = []

        for i, (poi_x, poi_y) in enumerate(self.poi_positions):
            # Convert normalized coordinates to actual field dimensions
            car_field_x = car_x * self.field_size[0]
            car_field_y = car_y * self.field_size[1]
            poi_field_x = poi_x * self.field_size[0]
            poi_field_y = poi_y * self.field_size[1]

            # Calculate Euclidean distance
            distance = np.sqrt((car_field_x - poi_field_x)**2 + (car_field_y - poi_field_y)**2)
            distances.append((i, distance))

        # Sort by distance
        return distances
