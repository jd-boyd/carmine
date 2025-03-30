import json
import os
import numpy as np
from quad import Quad
import config_db

# Constants
PRIMARY_CONFIG_FILE = "config.json"
CURRENT_CONFIG_NAME = "current"  # Default config name in database


class Config:
    """
    Class to hold application configuration that gets persisted.
    This includes all settings that should be saved between sessions.
    """

    def __init__(self):
        # Camera selection
        self.selected_camera1 = 0
        self.camera_name = ""

        # Camera points define the field boundary in camera view
        self.camera1_points = [[0, 0] for _ in range(4)]

        # Field dimensions
        self.field_size = [100, 300]

        # Points of Interest
        self.poi_positions = [
            (32, 90),
            (80, 150),
            (99, 210),
            (48, 240),
            (99, 60),
            (16, 270),
            (99, 30),
            (64, 180),
            (96, 120),
        ]
        self.poi_ranges = [45, 15, 3]  # Safe, Caution, Danger thresholds

        # UI and visualization settings
        self.c1_show_carbox = True
        self.c1_show_mines = True
        self.processing_paused = False

        # Video source settings
        self.use_video_file = False
        self.video_file_path = ""

        # Processing settings
        self.optical_flow_scale = 0.75

    def to_dict(self):
        """Convert config to dictionary for storage"""
        return {
            "selected_camera1": self.selected_camera1,
            "camera_name": self.camera_name,
            "camera1_points": self.camera1_points,
            "field_size": self.field_size,
            "poi_positions": self.poi_positions,
            "poi_ranges": self.poi_ranges,
            "c1_show_carbox": self.c1_show_carbox,
            "c1_show_mines": self.c1_show_mines,
            "processing_paused": self.processing_paused,
            "use_video_file": self.use_video_file,
            "video_file_path": self.video_file_path,
            "optical_flow_scale": self.optical_flow_scale,
        }

    def from_dict(self, config_dict):
        """Load config from dictionary"""
        if not config_dict:
            return

        # Load camera selection
        if "selected_camera1" in config_dict:
            self.selected_camera1 = config_dict["selected_camera1"]
        if "camera_name" in config_dict:
            self.camera_name = config_dict["camera_name"]

        # Load camera points
        if "camera1_points" in config_dict:
            self.camera1_points = config_dict["camera1_points"]

        # Load field size
        if "field_size" in config_dict:
            self.field_size = config_dict["field_size"]

        # Load POI positions
        if "poi_positions" in config_dict:
            self.poi_positions = config_dict["poi_positions"]
            if len(self.poi_positions) > 9:
                self.poi_positions = self.poi_positions[0:9]

        # Load POI ranges
        if "poi_ranges" in config_dict:
            self.poi_ranges = config_dict["poi_ranges"]

        # Load UI settings if available
        if "c1_show_carbox" in config_dict:
            self.c1_show_carbox = config_dict["c1_show_carbox"]
        if "c1_show_mines" in config_dict:
            self.c1_show_mines = config_dict["c1_show_mines"]
        if "processing_paused" in config_dict:
            self.processing_paused = config_dict["processing_paused"]

        # Load video file settings if available
        if "use_video_file" in config_dict:
            self.use_video_file = config_dict["use_video_file"]
        if "video_file_path" in config_dict:
            self.video_file_path = config_dict["video_file_path"]

        # Load optical flow settings if available
        if "optical_flow_scale" in config_dict:
            self.optical_flow_scale = config_dict["optical_flow_scale"]


class State:
    """
    Class to hold application state, separate from UI rendering.
    Manages both persisted configuration and runtime state.
    """

    def __init__(self, camera_list):
        # Store camera listing
        self.camera_list = camera_list

        # Persistent configuration
        self.config = Config()

        # Track the currently loaded config name
        self.current_config_name = CURRENT_CONFIG_NAME

        # Point selection state (transient UI state)
        self.waiting_for_camera1_point = (
            -1
        )  # Index of point we're waiting to set (-1 means not waiting)
        self.waiting_for_poi_point = (
            -1
        )  # Index of POI we're waiting to set (-1 means not waiting)

        # Car tracking - runtime state only
        self.highlighted_cars = []  # Stores [x, y, width, height, confidence, class_id] of highlighted cars
        self.car_field_positions = []  # Stores (x, y) positions of cars on the field
        # Backward compatibility fields
        self.highlighted_car = None
        self.car_field_position = None
        self.max_cars = 10  # Maximum number of cars to track

        # Cursor tracking - runtime state only
        self.c1_cursor = []  # Cursor position in window space
        self.c1_cursor_image_pos = (
            None  # Cursor position in image space (pixel coordinates)
        )
        self.c1_cursor_in_image = (
            False  # Flag to indicate if cursor is within the image
        )
        self.c1_cursor_field_position = None  # Cursor position in field space

        # Car detection data - runtime state
        self.car_detections = []  # Raw detections from YOLO
        self.previous_car_detections = []  # Store previous frame's detections for optical flow prediction

        # Cache for POI distance calculations (updated each frame) - runtime state
        self._poi_car_distances = {}  # Maps POI index to (car_index, distance) tuple
        self._closest_poi_car = (
            None  # (poi_index, car_index, distance) of closest overall pair
        )
        self._all_distances = None  # Comprehensive distance data
        self._cached_car_count = 0  # Track how many cars were in the last calculation

        # Setup geometric transform for coordinate conversion - runtime state
        self.camera1_quad = None

        # Load configuration if exists
        self.load_config()

        # Initialize quad after loading config
        self.update_camera_quad()

    # Property getters/setters to provide backward compatibility
    # and bridge between config and state

    @property
    def camera1_points(self):
        return self.config.camera1_points

    @camera1_points.setter
    def camera1_points(self, value):
        self.config.camera1_points = value
        self.update_camera_quad()

    @property
    def field_size(self):
        return self.config.field_size

    @field_size.setter
    def field_size(self, value):
        self.config.field_size = value

    @property
    def poi_positions(self):
        return self.config.poi_positions

    @poi_positions.setter
    def poi_positions(self, value):
        self.config.poi_positions = value
        self._update_distance_cache()

    @property
    def poi_ranges(self):
        return self.config.poi_ranges

    @poi_ranges.setter
    def poi_ranges(self, value):
        self.config.poi_ranges = value

    @property
    def c1_show_carbox(self):
        return self.config.c1_show_carbox

    @c1_show_carbox.setter
    def c1_show_carbox(self, value):
        self.config.c1_show_carbox = value

    @property
    def c1_show_mines(self):
        return self.config.c1_show_mines

    @c1_show_mines.setter
    def c1_show_mines(self, value):
        self.config.c1_show_mines = value

    @property
    def selected_camera1(self):
        return self.config.selected_camera1

    @selected_camera1.setter
    def selected_camera1(self, value):
        self.config.selected_camera1 = value

    @property
    def processing_paused(self):
        return self.config.processing_paused

    @processing_paused.setter
    def processing_paused(self, value):
        self.config.processing_paused = value

    @property
    def use_video_file(self):
        return self.config.use_video_file

    @use_video_file.setter
    def use_video_file(self, value):
        self.config.use_video_file = value

    @property
    def video_file_path(self):
        return self.config.video_file_path

    @video_file_path.setter
    def video_file_path(self, value):
        self.config.video_file_path = value

    @property
    def optical_flow_scale(self):
        return self.config.optical_flow_scale

    @optical_flow_scale.setter
    def optical_flow_scale(self, value):
        self.config.optical_flow_scale = value

    def update_camera_quad(self):
        """Update the camera quad whenever camera points change"""
        self.camera1_quad = Quad(self.camera1_points)

    def set_c1_cursor(self, c):
        """Update cursor position in window space"""
        self.c1_cursor = c

        # Calculate field position from cursor coordinates if cursor is not empty
        if c and len(c) == 2:
            self.c1_cursor_field_position = self.camera_to_field_position(c[0], c[1])
        else:
            self.c1_cursor_field_position = None

    def set_car_detections(self, car_detections):
        """Update car detections and calculate field positions"""
        # Store current detections in car_detections and save the previous ones
        self.previous_car_detections = self.car_detections.copy()
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
            self.car_field_position = (
                self.car_field_positions[0]
                if len(self.car_field_positions) > 0
                else None
            )

        # Update distance caches when car positions change
        self._update_distance_cache()

    def set_camera_point(self, camera_num, point_index, x, y):
        """Set a camera point to the given coordinates"""
        if camera_num == 1:
            self.camera1_points[point_index] = [x, y]
            self.waiting_for_camera1_point = -1  # Reset waiting state
            self.update_camera_quad()  # Update quad with new point

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
            # Update distance cache when POI positions change
            self._update_distance_cache()
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
                # print("Could not calculate field position - invalid transformation")
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
                          or [x1, y1, x2, y2, conf, cls_id, flow_x, flow_y] with flow data
        """
        if car_detection is None:
            return

        # Add car detection to the list of highlighted cars
        self.highlighted_cars.append(car_detection)

        # Extract the basic detection information (first 6 values)
        # Handle both formats: with and without flow data
        x1, y1, x2, y2, conf, cls_id = car_detection[:6]
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
            # Update camera name
            if self.selected_camera1 < len(self.camera_list):
                self.config.camera_name = self.camera_list[self.selected_camera1][1]

            # Get config as dictionary
            config_dict = self.config.to_dict()

            # Save to database
            success = config_db.save_config_to_db(config_name, config_dict)

            # Also save to file for backwards compatibility
            with open(PRIMARY_CONFIG_FILE, "w") as f:
                json.dump(config_dict, f, indent=4)

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
            config_dict = config_db.load_config_from_db(config_name)
            source = f"database ('{config_name}')"

            # If not found in database, try to load from file
            if not config_dict and os.path.exists(PRIMARY_CONFIG_FILE):
                with open(PRIMARY_CONFIG_FILE, "r") as f:
                    config_dict = json.load(f)
                source = f"file ({PRIMARY_CONFIG_FILE})"

                # Save this to database for future use
                config_db.save_config_to_db(config_name, config_dict)

            if not config_dict:
                print("No configuration found in database or file")
                return False

            # Update config object with loaded values
            self.config.from_dict(config_dict)

            # Update current config name to match what was loaded
            self.current_config_name = config_name

            print(f"Configuration loaded from {source}")

            # Update quad with new points
            self.update_camera_quad()

            return True

        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False

    def reset_config(self):
        """Reset configuration to default values"""
        # Simply create a new Config with defaults
        self.config = Config()

        # Update quad with reset points
        self.update_camera_quad()

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
            distance = np.sqrt((car_x - poi_x) ** 2 + (car_y - poi_y) ** 2)
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
        # Return cached value if available and cars haven't changed
        if self._all_distances is not None and self._cached_car_count == len(
            self.car_field_positions
        ):
            return self._all_distances

        all_distances = []

        # Use car_field_positions list for all cars
        for i in range(len(self.car_field_positions)):
            car_distances = self.calculate_poi_distances(i)
            if car_distances:
                all_distances.append(car_distances)

        # Cache the result
        self._all_distances = all_distances
        self._cached_car_count = len(self.car_field_positions)
        return all_distances

    def _update_distance_cache(self):
        """
        Update the internal distance cache.
        Called when car positions or POI positions change.
        """
        # Clear existing cache
        self._poi_car_distances = {}
        self._closest_poi_car = None
        self._all_distances = None
        self._cached_car_count = len(self.car_field_positions)

        # If no cars, nothing to cache
        if not self.car_field_positions:
            return

        # Find closest car for each POI
        for poi_idx, (poi_x, poi_y) in enumerate(self.poi_positions):
            min_distance = float("inf")
            closest_car = -1

            # Check each car's distance to this POI
            for car_idx, (car_x, car_y) in enumerate(self.car_field_positions):
                # Calculate Euclidean distance
                dist = ((car_x - poi_x) ** 2 + (car_y - poi_y) ** 2) ** 0.5
                if dist < min_distance:
                    min_distance = dist
                    closest_car = car_idx

            # Cache the closest car to this POI if one was found
            if closest_car >= 0:
                self._poi_car_distances[poi_idx] = (closest_car, min_distance)

        # Find the overall closest POI-car pair
        if self._poi_car_distances:
            closest_poi = -1
            closest_car = -1
            min_distance = float("inf")

            # Check each POI's closest car distance
            for poi_idx, (car_idx, distance) in self._poi_car_distances.items():
                if distance < min_distance:
                    min_distance = distance
                    closest_poi = poi_idx
                    closest_car = car_idx

            # Cache the closest pair
            if closest_poi >= 0:
                self._closest_poi_car = (closest_poi, closest_car, min_distance)

    def get_closest_car_to_poi(self, poi_index):
        """
        Get the closest car to a specific POI.

        Args:
            poi_index: Index of the POI to check

        Returns:
            tuple: (car_index, distance) or None if no cars or POI doesn't exist
        """
        # Ensure cache is updated
        if self._cached_car_count != len(self.car_field_positions):
            self._update_distance_cache()

        # Return cached result if available
        return self._poi_car_distances.get(poi_index, None)

    def get_closest_poi_car_pair(self):
        """
        Get the closest POI-car pair overall.

        Returns:
            tuple: (poi_index, car_index, distance) or None if no cars
        """
        # Ensure cache is updated
        if self._cached_car_count != len(self.car_field_positions):
            self._update_distance_cache()

        return self._closest_poi_car

    def get_poi_distance_color(self, distance):
        """
        Get the color for a POI based on its distance from a car.

        Args:
            distance: Distance value to check

        Returns:
            tuple: (r, g, b) color values (0-1 range)
        """
        # Default color (red) if no distance provided
        if distance is None or distance == float("inf"):
            return (1.0, 0.0, 0.0)

        # Get thresholds from poi_ranges
        if len(self.poi_ranges) >= 3:
            safe_distance = self.poi_ranges[0]
            caution_distance = self.poi_ranges[1]
            danger_distance = self.poi_ranges[2]
        else:
            # Default values if not enough ranges defined
            safe_distance = 45
            caution_distance = 15
            danger_distance = 3

        # Set color based on distance thresholds
        if distance > safe_distance:
            # Green - safe
            return (0.0, 1.0, 0.0)
        elif distance > caution_distance:
            # Yellow - caution
            return (1.0, 1.0, 0.0)
        elif distance > danger_distance:
            # Orange - approaching danger
            return (1.0, 0.5, 0.0)
        else:
            # Red - danger
            return (1.0, 0.0, 0.0)
