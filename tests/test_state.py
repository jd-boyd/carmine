import unittest
import os
import json
import tempfile
import numpy as np
import sys

# Add the parent directory to sys.path to import modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import state module
import state
from state import State

class TestState(unittest.TestCase):
    def setUp(self):
        # Create a temporary config file for testing
        self.temp_config = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_config.close()  # Close the file so it can be reopened on Windows
        
        # Store the original config file path
        self.original_config_file = state.CONFIG_FILE
        
        # Override the CONFIG_FILE value temporarily
        state.CONFIG_FILE = self.temp_config.name
        
        # Mock camera list for testing
        self.camera_list = [(0, "Camera 0"), (1, "Camera 1")]
        
        # Create a test state instance
        self.state = State(self.camera_list)
    
    def tearDown(self):
        # Clean up the temporary config file
        state.CONFIG_FILE = self.original_config_file
        if os.path.exists(self.temp_config.name):
            try:
                os.unlink(self.temp_config.name)
            except (PermissionError, OSError):
                pass  # Ignore permission errors when cleaning up
    
    def test_initialization(self):
        """Test that State initializes with correct default values"""
        # Test camera list
        self.assertEqual(self.state.camera_list, self.camera_list)
        
        # Test camera selection
        self.assertEqual(self.state.selected_camera1, 0)
        self.assertEqual(self.state.selected_camera2, 0)
        
        # Test camera points
        self.assertEqual(len(self.state.camera1_points), 4)
        self.assertEqual(len(self.state.camera2_points), 4)
        self.assertEqual(self.state.camera1_points[0], [0, 0])
        
        # Test waiting states
        self.assertEqual(self.state.waiting_for_camera1_point, -1)
        self.assertEqual(self.state.waiting_for_camera2_point, -1)
        self.assertEqual(self.state.waiting_for_poi_point, -1)
        
        # Test car tracking
        self.assertIsNone(self.state.highlighted_car)
        self.assertIsNone(self.state.car_field_position)
        
        # Test field size
        self.assertEqual(self.state.field_size, [160, 300])
        
        # Test POI info
        self.assertEqual(len(self.state.poi_info), 10)
        self.assertEqual(self.state.poi_info[0], "")
        
        # Test POI positions
        self.assertEqual(len(self.state.poi_positions), 10)
        self.assertEqual(self.state.poi_positions[0], (0.2, 0.3))
    
    def test_camera_point_management(self):
        """Test setting and getting camera points"""
        # Test setting camera 1 point
        self.state.set_camera_point(1, 0, 100, 200)
        self.assertEqual(self.state.camera1_points[0], [100, 200])
        self.assertEqual(self.state.waiting_for_camera1_point, -1)
        
        # Test setting camera 2 point
        self.state.set_camera_point(2, 1, 300, 400)
        self.assertEqual(self.state.camera2_points[1], [300, 400])
        self.assertEqual(self.state.waiting_for_camera2_point, -1)
        
        # Test getting camera IDs
        self.assertEqual(self.state.get_camera1_id(), 0)
        self.assertEqual(self.state.get_camera2_id(), 0)  # Both are initialized to 0
        
        # Test with invalid camera number
        self.state.set_camera_point(3, 0, 500, 600)
        self.assertNotEqual(self.state.camera1_points[0], [500, 600])
        self.assertNotEqual(self.state.camera2_points[0], [500, 600])
    
    def test_poi_management(self):
        """Test setting and getting POI information and positions"""
        # Test setting POI info
        self.state.set_poi_info(0, "Test POI")
        self.assertEqual(self.state.poi_info[0], "Test POI")
        
        # Test setting POI position
        self.state.set_poi_position(1, 0.4, 0.7)
        self.assertEqual(self.state.poi_positions[1], (0.4, 0.7))
        self.assertEqual(self.state.waiting_for_poi_point, -1)
        
        # Test with invalid index
        self.state.set_poi_info(10, "Invalid POI")
        self.state.set_poi_position(10, 0.1, 0.1)
        self.assertEqual(len(self.state.poi_info), 10)
        self.assertEqual(len(self.state.poi_positions), 10)
    
    def test_config_save_load(self):
        """Test saving and loading configuration"""
        # Set some values
        self.state.selected_camera1 = 1
        self.state.camera1_points[0] = [100, 200]
        self.state.field_size = [200, 350]
        self.state.poi_info[0] = "Test POI"
        self.state.poi_positions[1] = (0.4, 0.7)
        
        # Save config
        self.state.save_config()
        
        # Create a new state instance that should load the config
        new_state = State(self.camera_list)
        
        # Verify loaded values
        self.assertEqual(new_state.selected_camera1, 1)
        self.assertEqual(new_state.camera1_points[0], [100, 200])
        self.assertEqual(new_state.field_size, [200, 350])
        self.assertEqual(new_state.poi_info[0], "Test POI")
        
        # The tuple might be converted to a list during JSON serialization
        # So check values individually
        self.assertAlmostEqual(new_state.poi_positions[1][0], 0.4)
        self.assertAlmostEqual(new_state.poi_positions[1][1], 0.7)
    
    def test_reset_config(self):
        """Test resetting configuration to defaults"""
        # Set some non-default values
        self.state.selected_camera1 = 1
        self.state.camera1_points[0] = [100, 200]
        self.state.field_size = [200, 350]
        self.state.poi_info[0] = "Test POI"
        self.state.poi_positions[1] = (0.4, 0.7)
        
        # Reset config
        self.state.reset_config()
        
        # Verify reset values
        self.assertEqual(self.state.camera1_points[0], [0, 0])
        self.assertEqual(self.state.field_size, [160, 300])
        self.assertEqual(self.state.poi_info[0], "")
        
        # Check POI position values individually
        self.assertAlmostEqual(self.state.poi_positions[1][0], 0.5)
        self.assertAlmostEqual(self.state.poi_positions[1][1], 0.5)
    
    def test_highlight_car(self):
        """Test highlighting a car and calculating field position"""
        # Create a car detection
        car_detection = [10, 20, 50, 80, 0.9, 2]  # [x1, y1, x2, y2, conf, cls_id]
        
        # Highlight the car - it won't actually calculate a field position
        # in the test since we need a valid Quad, but it should still set the car
        self.state.highlight_car(car_detection)
        
        # Verify the car is highlighted
        self.assertEqual(self.state.highlighted_car, car_detection)
        
        # Test unhighlighting
        self.state.highlight_car(None)
        self.assertIsNone(self.state.highlighted_car)
        self.assertIsNone(self.state.car_field_position)
    
    def test_camera_to_field_position_exists(self):
        """Test that the camera_to_field_position method exists and is callable"""
        # We'll just verify that the method exists and is callable
        self.assertTrue(hasattr(self.state, 'camera_to_field_position'))
        self.assertTrue(callable(getattr(self.state, 'camera_to_field_position')))

if __name__ == '__main__':
    unittest.main()