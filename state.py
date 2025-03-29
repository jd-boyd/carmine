import json
import os
import numpy as np
from quad import Quad

# Constants
PRIMARY_CONFIG_FILE = "config.json"
LEGACY_CONFIG_FILE = "carmine_config.json"

class State:
    """
    Class to hold application state, separate from UI rendering.
    """
    def __init__(self, camera_list):
        self.camera_list = camera_list

        # Camera selection
        self.selected_camera1 = 0
        self.selected_camera2 = 0

        self.camera1_points = []
        self.camera2_points = []

        self.reset_config()

        # Point selection state
        self.waiting_for_camera1_point = -1  # Index of point we're waiting to set (-1 means not waiting)
        self.waiting_for_camera2_point = -1  # Index of point we're waiting to set (-1 means not waiting)
        self.waiting_for_poi_point = -1      # Index of POI we're waiting to set (-1 means not waiting)

        # Car tracking - extended to handle multiple cars
        self.highlighted_cars = []  # Will store list of [x, y, width, height, confidence, class_id] of highlighted cars
        self.car_field_positions = []  # Will store list of (x, y) positions of cars on the field
        # Keep these for backward compatibility
        self.highlighted_car = None
        self.car_field_position = None
        
        self.max_cars = 10  # Maximum number of cars to track
        self.c1_show_carbox = True
        self.c1_show_mines = True
        self.c1_cursor = []
        self.c1_cursor_field_position = None

        self.car_detections = []  # Raw detections from YOLO

        # Load configuration if exists
        self.load_config()

        self.camera1_quad = Quad(self.camera1_points)
        self.camera2_quad = Quad(self.camera2_points)


    def set_c1_cursor(self, c):
        self.c1_cursor = c
        
        # Calculate field position from cursor coordinates if cursor is not empty
        if c and len(c) == 2:
            self.c1_cursor_field_position = self.camera_to_field_position(c[0], c[1])
        else:
            self.c1_cursor_field_position = None

    def set_car_detections(self, car_detections):
        self.car_detections = car_detections
        
        # Clear previous highlighted cars
        self.highlighted_cars = []
        self.car_field_positions = []
        
        # Process up to max_cars detections
        cars_to_process = min(len(car_detections), self.max_cars)
        
        # Process each car in the detections
        for i in range(cars_to_process):
            car = car_detections[i]
            self.highlight_car(car)
            
        # For backward compatibility - set the first car as the primary highlighted car
        if len(self.highlighted_cars) > 0:
            self.highlighted_car = self.highlighted_cars[0]
            self.car_field_position = self.car_field_positions[0] if len(self.car_field_positions) > 0 else None


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
        """Set a POI position to the given field coordinates"""
        if 0 <= index < len(self.poi_positions):
            self.poi_positions[index] = (x, y)
            self.waiting_for_poi_point = -1  # Reset waiting state
            self.save_config()

    def field_to_uv(self, field_x, field_y):
        """
        Convert field coordinates to UV coordinates (0-1)
        
        Args:
            field_x: X coordinate in field units
            field_y: Y coordinate in field units
            
        Returns:
            tuple: (u, v) normalized coordinates (0-1)
        """
        return (field_x / self.field_size[0], field_y / self.field_size[1])
    
    def uv_to_field(self, u, v):
        """
        Convert UV coordinates (0-1) to field coordinates
        
        Args:
            u: U coordinate (0-1)
            v: V coordinate (0-1)
            
        Returns:
            tuple: (x, y) field coordinates
        """
        return (u * self.field_size[0], v * self.field_size[1])
    
    def camera_to_field_position(self, camera_x, camera_y, camera_num=1):
        """
        Convert a camera point (in pixel coordinates) to field position (in field coordinates)

        Args:
            camera_x: X coordinate in the camera frame
            camera_y: Y coordinate in the camera frame
            camera_num: Camera number (1 or 2)

        Returns:
            tuple: (x, y) field coordinates
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
            quad = Quad(camera_points)

            uv = quad.point_to_uv(camera_x, camera_y)

            if uv is None:
                print("Could not calculate field position - invalid transformation")
                return None

            # Convert UV to field coordinates
            field_x, field_y = self.uv_to_field(uv[0], uv[1])
            return (field_x, field_y)
        except Exception as e:
            print(f"Error converting camera to field position: {e}")
            return None

    def highlight_car(self, car_detection):
        """
        Highlight a car and calculate its position on the field
        Now handles multiple cars by appending to a list

        Args:
            car_detection: [x1, y1, x2, y2, conf, cls_id] car detection
        """
        if car_detection is None:
            return

        # Add car detection to the list of highlighted cars
        self.highlighted_cars.append(car_detection)

        # Calculate the center of the car
        x1, y1, x2, y2, conf, cls_id = car_detection
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Convert to field position (already in field coordinates)
        field_position = self.camera_to_field_position(center_x, center_y)
        
        # Add field position to the list
        if field_position is not None:
            self.car_field_positions.append(field_position)

    def save_config(self):
        """Save the current configuration to the primary JSON file"""
        try:
            config = {
                'selected_camera1': self.selected_camera1,
                'selected_camera2': self.selected_camera2,
                'camera1_points': self.camera1_points,
                'camera2_points': self.camera2_points,
                'field_size': self.field_size,
                'poi_positions': self.poi_positions
            }

            with open(PRIMARY_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Configuration saved to {PRIMARY_CONFIG_FILE}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def load_config(self):
        """Load configuration from a JSON file if it exists"""
        try:
            # Try to load the primary config file first
            if os.path.exists(PRIMARY_CONFIG_FILE):
                config_file = PRIMARY_CONFIG_FILE
            # Fall back to legacy config file if primary doesn't exist
            elif os.path.exists(LEGACY_CONFIG_FILE):
                config_file = LEGACY_CONFIG_FILE
                print(f"Using legacy config file: {LEGACY_CONFIG_FILE}")
            else:
                print(f"No configuration file found")
                return

            with open(config_file, 'r') as f:
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

            print(f"Configuration loaded from {config_file}")

            # If we loaded from legacy file, save to the primary file for future use
            if config_file == LEGACY_CONFIG_FILE:
                self.save_config()
                print(f"Migrated configuration to {PRIMARY_CONFIG_FILE}")
        except Exception as e:
            print(f"Error loading configuration: {e}")

    def reset_config(self):
        """Reset configuration to default values"""
        # Reset camera points
        self.camera1_points = [[0, 0] for _ in range(4)]
        self.camera2_points = [[0, 0] for _ in range(4)]

        self.field_size = [160, 300]

        # POI positions in field coordinates (previously normalized 0-1)
        self.poi_positions = [
            (32, 90), (80, 150), (128, 210), (48, 240), (112, 60),
            (16, 270), (144, 30), (64, 180), (96, 120), (80, 240)
        ]

    def calculate_poi_distances(self, car_index=0):
        """
        Calculate distances between a car and each point of interest.

        Args:
            car_index: Index of the car to calculate distances for (defaults to first car)

        Returns:
            list: List of (poi_index, distance) tuples sorted by distance,
                  or None if car position is not available
        """
        # For backward compatibility
        if car_index == 0 and self.car_field_position is not None:
            car_pos = self.car_field_position
        elif car_index < len(self.car_field_positions):
            car_pos = self.car_field_positions[car_index]
        else:
            return None

        distances = []
        car_x, car_y = car_pos

        for i, (poi_x, poi_y) in enumerate(self.poi_positions):
            # Calculate Euclidean distance directly (since all coordinates are now in field units)
            distance = np.sqrt((car_x - poi_x)**2 + (car_y - poi_y)**2)
            distances.append((i, distance))

        # Sort by distance
        distances.sort(key=lambda x: x[1])
        return distances
        
    def calculate_all_car_poi_distances(self):
        """
        Calculate distances between all cars and each point of interest.

        Returns:
            list: List of car distances, where each entry is a list of (poi_index, distance) tuples
                  for that car, sorted by distance
        """
        all_distances = []
        
        # Use car_field_positions list for all cars
        for i in range(len(self.car_field_positions)):
            car_distances = self.calculate_poi_distances(i)
            if car_distances:
                all_distances.append(car_distances)
                
        return all_distances
