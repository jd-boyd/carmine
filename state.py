import json
import os
import numpy as np
from quad import Quad
import config_db

# Constants
PRIMARY_CONFIG_FILE = "config.json"
CURRENT_CONFIG_NAME = "current"  # Default config name in database

class State:
    """
    Class to hold application state, separate from UI rendering.
    """
    def __init__(self, camera_list):
        self.camera_list = camera_list

        # Camera selection
        self.camera1_points = []
        self.selected_camera1 = 0
        self.reset_config()
        
        # Track the currently loaded config name
        self.current_config_name = CURRENT_CONFIG_NAME

        # Point selection state
        self.waiting_for_camera1_point = -1  # Index of point we're waiting to set (-1 means not waiting)
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
        self.processing_paused = False  # Flag to control YOLO and detector processing

        self.car_detections = []  # Raw detections from YOLO

        # Load configuration if exists
        self.load_config()

        self.camera1_quad = Quad(self.camera1_points)


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

        # Save configuration after updating points
        self.save_config()

    def get_camera1_id(self):
        """Get the selected camera1 ID"""
        if self.selected_camera1 < len(self.camera_list):
            return self.camera_list[self.selected_camera1][0]
        return None

    def set_poi_position(self, index, x, y):
        """Set a POI position to the given field coordinates"""
        if 0 <= index < len(self.poi_positions):
            self.poi_positions[index] = (x, y)
            self.waiting_for_poi_point = -1  # Reset waiting state
            self.save_config()
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

        # Check if we have valid quad points
        if not all(isinstance(p, list) and len(p) == 2 for p in camera_points):
            print("Invalid camera points for field position calculation")
            return None

        try:
            # Create a quad with the current field size
            quad = Quad(camera_points, field_size=self.field_size)

            # Directly convert camera coordinates to field coordinates
            field_position = quad.point_to_field(camera_x, camera_y)

            if field_position is None:
                #print("Could not calculate field position - invalid transformation")
                return None

            return field_position
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

    def save_config(self, config_name=CURRENT_CONFIG_NAME):
        """
        Save the current configuration to the database

        Args:
            config_name: Name to save the configuration as (defaults to "current")

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Get camera info if available
            camera_name = ""
            if self.selected_camera1 < len(self.camera_list):
                camera_name = self.camera_list[self.selected_camera1][1]
                
            config = {
                'selected_camera1': self.selected_camera1,
                'camera_name': camera_name,  # Store camera name for display
                'camera1_points': self.camera1_points,
                'field_size': self.field_size,
                'poi_positions': self.poi_positions,
                'poi_ranges': self.poi_ranges,
                'c1_show_carbox': self.c1_show_carbox,
                'c1_show_mines': self.c1_show_mines,
                'processing_paused': self.processing_paused
            }

            # Save to database
            success = config_db.save_config_to_db(config_name, config)

            # Also save to file for backwards compatibility
            with open(PRIMARY_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)

            # Update current config name if successful
            if success:
                self.current_config_name = config_name
                print(f"Configuration saved to database as '{config_name}'")
            return success

        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False

    def load_config(self, config_name=CURRENT_CONFIG_NAME):
        """
        Load configuration from the database or file if database config doesn't exist

        Args:
            config_name: Name of the configuration to load (defaults to "current")

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            # First try to load from database
            config = config_db.load_config_from_db(config_name)
            source = f"database ('{config_name}')"

            # If not found in database, try to load from file
            if not config and os.path.exists(PRIMARY_CONFIG_FILE):
                with open(PRIMARY_CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                source = f"file ({PRIMARY_CONFIG_FILE})"

                # Save this to database for future use
                config_db.save_config_to_db(config_name, config)

            if not config:
                print("No configuration found in database or file")
                return False

            # Load camera selection
            if 'selected_camera1' in config:
                self.selected_camera1 = config['selected_camera1']

            # Load camera points
            if 'camera1_points' in config:
                self.camera1_points = config['camera1_points']

            # Load field size
            if 'field_size' in config:
                self.field_size = config['field_size']

            # Load POI positions
            if 'poi_positions' in config:
                self.poi_positions = config['poi_positions']
                if len(self.poi_positions) > 9:
                    self.poi_positions = self.poi_positions[0:9]

            # Load POI ranges
            if 'poi_ranges' in config:
                self.poi_ranges = config['poi_ranges']

            # Load UI settings if available
            if 'c1_show_carbox' in config:
                self.c1_show_carbox = config['c1_show_carbox']
            if 'c1_show_mines' in config:
                self.c1_show_mines = config['c1_show_mines']
            if 'processing_paused' in config:
                self.processing_paused = config['processing_paused']

            # Update current config name to match what was loaded
            self.current_config_name = config_name
            
            print(f"Configuration loaded from {source}")

            # Update quad with new points
            self.camera1_quad = Quad(self.camera1_points)

            return True

        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False

    def reset_config(self):
        """Reset configuration to default values"""
        # Reset camera points
        self.camera1_points = [[0, 0] for _ in range(4)]
        self.field_size = [100, 300]

        # POI positions in field coordinates (previously normalized 0-1)
        self.poi_positions = [
            (32, 90), (80, 150), (99, 210), (48, 240), (99, 60),
            (16, 270), (99, 30), (64, 180), (96, 120)
        ]

        self.poi_ranges = [45, 15, 3]


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
