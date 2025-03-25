import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw
import cv2
import OpenGL.GL as gl
import numpy as np
import json
import os
from cv2_enumerate_cameras import enumerate_cameras
from ultralytics import YOLO
import PySimpleGUI as sg

import sources
from sources import create_opengl_texture, update_opengl_texture

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

        # Field dimensions (aspect ratio)
        self.field_size = [160, 300]  # [width, height]

        # Points of interest information
        self.poi_info = ["" for _ in range(10)]  # Info for 10 POIs

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

    def set_poi_info(self, index, info):
        """Set information for a specific POI"""
        if 0 <= index < len(self.poi_info):
            self.poi_info[index] = info
            self.save_config()

    def save_config(self):
        """Save the current configuration to a JSON file"""
        try:
            config = {
                'selected_camera1': self.selected_camera1,
                'selected_camera2': self.selected_camera2,
                'camera1_points': self.camera1_points,
                'camera2_points': self.camera2_points,
                'field_size': self.field_size,
                'poi_info': self.poi_info,
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

            # Load POI info
            if 'poi_info' in config:
                self.poi_info = config['poi_info']

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

        # Reset POI info
        self.poi_info = ["" for _ in range(10)]

        # Reset POI positions to defaults
        self.poi_positions = [
            (0.2, 0.3), (0.5, 0.5), (0.8, 0.7), (0.3, 0.8), (0.7, 0.2),
            (0.1, 0.9), (0.9, 0.1), (0.4, 0.6), (0.6, 0.4), (0.5, 0.8)
        ]


        # Save the reset configuration
        self.save_config()
        print("Configuration reset to defaults")


class ControlPanel:
    """
    UI component for displaying and manipulating application state.
    """
    def __init__(self, state):
        self.state = state

    def draw(self):
        """Draw the control panel UI and update state values"""
        if imgui.begin("Control Panel", True):
            imgui.text("Cameras")
            changed1, self.state.selected_camera1 = imgui.combo(
                "Camera 1", self.state.selected_camera1, [c[1] for c in self.state.camera_list]
            )
            if changed1:
                self.state.save_config()

            changed2, self.state.selected_camera2 = imgui.combo(
                "Camera 2", self.state.selected_camera2, [c[1] for c in self.state.camera_list]
            )
            if changed2:
                self.state.save_config()
            imgui.separator()

            imgui.text("Field Size")
            changed_width, self.state.field_size[0] = imgui.input_int(
                "Width", self.state.field_size[0]
            )

            if changed_width:
                self.state.save_config()

            changed_height, self.state.field_size[1] = imgui.input_int(
                "Height", self.state.field_size[1]
            )

            if changed_height:
                self.state.save_config()

            imgui.separator()

            imgui.text("Fields")

            # Camera 1 points with display and Set button
            imgui.text("Camera1 Points")
            for i in range(4):
                # Display the current value as (x, y)
                x, y = self.state.camera1_points[i]
                imgui.text(f"Point {i+1}: ({x}, {y})")

                # Indicate if we're waiting for this point to be set
                if self.state.waiting_for_camera1_point == i:
                    imgui.same_line()
                    imgui.text_colored("Waiting for click on image...", 1, 0.5, 0, 1)

                # Add Set button
                imgui.same_line()

                # Change button color/text if this is the active point waiting for selection
                button_text = "Cancel" if self.state.waiting_for_camera1_point == i else "Set"
                if imgui.button(f"{button_text}##cam1_{i}"):
                    if self.state.waiting_for_camera1_point == i:
                        # Cancel selection mode
                        self.state.waiting_for_camera1_point = -1
                    else:
                        # Enter selection mode for this point
                        self.state.waiting_for_camera1_point = i
                        # Reset any other waiting state
                        self.state.waiting_for_camera2_point = -1
                        print(f"Click on the image to set Camera 1 Point {i+1}")

            # Camera 2 points with display and Set button
            imgui.text("Camera2 Points")
            for i in range(4):
                # Display the current value as (x, y)
                x, y = self.state.camera2_points[i]
                imgui.text(f"Point {i+1}: ({x}, {y})")

                # Indicate if we're waiting for this point to be set
                if self.state.waiting_for_camera2_point == i:
                    imgui.same_line()
                    imgui.text_colored("Waiting for click on image...", 1, 0.5, 0, 1)

                # Add Set button
                imgui.same_line()

                # Change button color/text if this is the active point waiting for selection
                button_text = "Cancel" if self.state.waiting_for_camera2_point == i else "Set"
                if imgui.button(f"{button_text}##cam2_{i}"):
                    if self.state.waiting_for_camera2_point == i:
                        # Cancel selection mode
                        self.state.waiting_for_camera2_point = -1
                    else:
                        # Enter selection mode for this point
                        self.state.waiting_for_camera2_point = i
                        # Reset any other waiting state
                        self.state.waiting_for_camera1_point = -1
                        print(f"Click on the image to set Camera 2 Point {i+1}")
            imgui.separator()

            imgui.text("PoI")
            for i in range(10):
                imgui.text(f"Point {i+1}: {self.state.poi_info[i] or 'Information'}")

            imgui.separator()

            # Config Management
            imgui.text("Configuration")
            if imgui.button("Reset to Defaults"):
                if imgui.begin_popup_modal("Confirm Reset", True):
                    imgui.text("Are you sure you want to reset all configuration to defaults?")
                    imgui.text("This action cannot be undone.")
                    imgui.separator()

                    if imgui.button("Yes", 120, 0):
                        self.state.reset_config()
                        imgui.close_current_popup()

                    imgui.same_line()
                    if imgui.button("No", 120, 0):
                        imgui.close_current_popup()

                    imgui.end_popup()
                else:
                    imgui.open_popup("Confirm Reset")

            imgui.end()

    def draw_field_visualization(self):
        """Draw a visualization of the field with POI positions"""
        # Set default window size
        # Field is wider than tall, so width should be greater (swapping dimensions)
        default_width = 400
        field_aspect_ratio = self.state.field_size[0] / self.state.field_size[1]  # Width / Height
        default_height = int(default_width / field_aspect_ratio)
        imgui.set_next_window_size(default_width, default_height, imgui.FIRST_USE_EVER)

        if imgui.begin("Field Visualization"):
            # Get drawing area info
            draw_list = imgui.get_window_draw_list()
            canvas_pos_x, canvas_pos_y = imgui.get_cursor_screen_pos()
            canvas_width = imgui.get_content_region_available_width()
            canvas_height = canvas_width / field_aspect_ratio

            # Draw field outline with thicker border
            draw_list.add_rect(
                canvas_pos_x, canvas_pos_y,
                canvas_pos_x + canvas_width, canvas_pos_y + canvas_height,
                imgui.get_color_u32_rgba(1, 1, 1, 1),  # White color
                0, 2.0  # No rounding, 2px thickness
            )

            # Draw POIs
            for i, (y, x) in enumerate(self.state.poi_positions):
                # Calculate pixel position on the canvas
                poi_x = canvas_pos_x + (x * canvas_width)
                poi_y = canvas_pos_y + (y * canvas_height)

                # Draw X marker
                marker_size = 5.0
                color = imgui.get_color_u32_rgba(1, 0, 0, 1)  # Red color

                # Draw X
                draw_list.add_line(
                    poi_x - marker_size, poi_y - marker_size,
                    poi_x + marker_size, poi_y + marker_size,
                    color, 2.0
                )
                draw_list.add_line(
                    poi_x - marker_size, poi_y + marker_size,
                    poi_x + marker_size, poi_y - marker_size,
                    color, 2.0
                )

                # Draw POI number
                draw_list.add_text(
                    poi_x + marker_size + 2,
                    poi_y - marker_size - 2,
                    imgui.get_color_u32_rgba(1, 1, 0, 1),  # Yellow color
                    f"{i+1}"
                )

            # Add some space for the canvas
            imgui.dummy(0, canvas_height + 10)

            imgui.end()


def create_glfw_window(window_name="Carmine", width=1280, height=720):
    if not glfw.init():
        raise Exception("GLFW initialization failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
    window = glfw.create_window(width, height, window_name, None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")
    glfw.make_context_current(window)
    return window


def ask_for_file():
    layout = [
        [sg.Text("Select a file:")],
        [sg.Input(key="-FILE-"), sg.FileBrowse()],
        [sg.Button("OK"), sg.Button("Cancel")],
    ]

    window = sg.Window("File Browser", layout)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Cancel":
            break
        if event == "OK":
            file_path = values["-FILE-"]
            if file_path:
                print(f"Selected file: {file_path}")
            else:
                print("No file selected")
            break

    window.close()


def main():
    camera_list = []
    for camera_info in enumerate_cameras():
        desc = (camera_info.index, f'{camera_info.index}: {camera_info.name}')
        print(desc[1])
        camera_list.append(desc)

    model=YOLO('yolov8s.pt')

    window = create_glfw_window()
    imgui.create_context()
    impl = GlfwRenderer(window)

    # Initialize application state
    app_state = State(camera_list)

    # Initialize the control panel with the state
    global control_panel
    control_panel = ControlPanel(app_state)

    # Initialize video sources
    source_1 = sources.VideoSource('./AI_angle_1.mov', model)
    source_2 = sources.VideoSource('./AI_angle_2.mov', model)

    # Frame timing variables
    frame_time = 1.0/60.0  # Target 60 FPS
    last_time = glfw.get_time()

    running = True
    while running:
        if glfw.window_should_close(window):
            running = False

        # Calculate frame timing
        current_time = glfw.get_time()
        delta_time = current_time - last_time

        # Only process a new frame if enough time has passed
        if delta_time >= frame_time:
            last_time = current_time

            glfw.poll_events()
            impl.process_inputs()
            imgui.new_frame()

            # Start UI rendering
            if imgui.begin_main_menu_bar():
                if imgui.begin_menu("File", True):
                    clicked_save, selected_save = imgui.menu_item(
                        "Save Config", "Ctrl+S", False, True
                    )

                    if clicked_save:
                        app_state.save_config()

                    imgui.separator()

                    clicked_quit, selected_quit = imgui.menu_item(
                        "Quit", "Ctrl+Q", False, True
                    )

                    if clicked_quit:
                        running = False

                    imgui.end_menu()
                imgui.end_main_menu_bar()

            tex_id = source_1.get_next_frame()

            # Set default window size for OpenCV Image window
            default_width = 640
            # Calculate the correct height based on the video's aspect ratio
            aspect_ratio = source_1.width / source_1.height
            default_height = default_width / aspect_ratio
            imgui.set_next_window_size(default_width, default_height, imgui.FIRST_USE_EVER)

            imgui.begin("OpenCV Image")
            if tex_id:
                # Get window position (needed for mouse position calculation)
                window_pos_x, window_pos_y = imgui.get_window_position()
                # Get cursor position (for content region position)
                cursor_pos_x, cursor_pos_y = imgui.get_cursor_screen_pos()

                # Get available width and height of the ImGui window content area
                avail_width = imgui.get_content_region_available_width()

                # Calculate aspect ratio to maintain proportions
                aspect_ratio = source_1.width / source_1.height

                # Set display dimensions based on available width and aspect ratio
                display_width = avail_width
                display_height = display_width / aspect_ratio

                # Draw the image
                imgui.image(tex_id, display_width, display_height)

                # Draw camera quads if they have points defined
                if True:  # Always draw, not just on hover
                    draw_list = imgui.get_window_draw_list()

                    # Function to draw quad for a camera
                    def draw_camera_quad(points, color):
                        # Only draw if we have valid points
                        if all(isinstance(p, list) and len(p) == 2 for p in points):
                            # Calculate scaling factors to map image coords to screen coords
                            scale_x = display_width / source_1.width
                            scale_y = display_height / source_1.height

                            # Convert image coordinates to screen coordinates
                            screen_points = []
                            for x, y in points:
                                screen_x = cursor_pos_x + (x * scale_x)
                                screen_y = cursor_pos_y + (y * scale_y)
                                screen_points.append((screen_x, screen_y))

                            # Draw the quad as connected lines
                            for i in range(4):
                                next_i = (i + 1) % 4
                                draw_list.add_line(
                                    screen_points[i][0], screen_points[i][1],
                                    screen_points[next_i][0], screen_points[next_i][1],
                                    color, 2.0  # 2px thick
                                )

                    # Draw Camera 1 quad in bright green
                    draw_camera_quad(app_state.camera1_points, imgui.get_color_u32_rgba(0, 1, 0, 0.8))

                    # Draw Camera 2 quad in bright blue
                    draw_camera_quad(app_state.camera2_points, imgui.get_color_u32_rgba(0, 0, 1, 0.8))

                # Draw crosshairs when hovering over the image
                if imgui.is_item_hovered():
                    # Get mouse position
                    mouse_x, mouse_y = imgui.get_io().mouse_pos

                    # Only draw if mouse is inside the image area
                    if (cursor_pos_x <= mouse_x <= cursor_pos_x + display_width and
                        cursor_pos_y <= mouse_y <= cursor_pos_y + display_height):

                        # Draw vertical line
                        draw_list = imgui.get_window_draw_list()
                        draw_list.add_line(
                            mouse_x, cursor_pos_y,
                            mouse_x, cursor_pos_y + display_height,
                            imgui.get_color_u32_rgba(1, 1, 0, 0.5), 1.0
                        )

                        # Draw horizontal line
                        draw_list.add_line(
                            cursor_pos_x, mouse_y,
                            cursor_pos_x + display_width, mouse_y,
                            imgui.get_color_u32_rgba(1, 1, 0, 0.5), 1.0
                        )

                # Check for mouse clicks inside the image
                if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):  # 0 = left mouse button
                    # Get mouse position
                    mouse_x, mouse_y = imgui.get_io().mouse_pos

                    # Calculate relative position within the image
                    rel_x = (mouse_x - cursor_pos_x) / display_width
                    rel_y = (mouse_y - cursor_pos_y) / display_height

                    # Convert to original video frame coordinates
                    frame_x = int(rel_x * source_1.width)
                    frame_y = int(rel_y * source_1.height)

                    # Print to console
                    print(f"Click at video position: x={frame_x}, y={frame_y} (relative: {rel_x:.3f}, {rel_y:.3f})")

                    # Check if we're waiting to set a camera point
                    if app_state.waiting_for_camera1_point >= 0:
                        # Set the camera 1 point
                        point_idx = app_state.waiting_for_camera1_point
                        app_state.set_camera_point(1, point_idx, frame_x, frame_y)
                        print(f"Set Camera 1 Point {point_idx+1} to ({frame_x}, {frame_y})")

                    elif app_state.waiting_for_camera2_point >= 0:
                        # Set the camera 2 point
                        point_idx = app_state.waiting_for_camera2_point
                        app_state.set_camera_point(2, point_idx, frame_x, frame_y)
                        print(f"Set Camera 2 Point {point_idx+1} to ({frame_x}, {frame_y})")
            imgui.end()


            # Draw the control panel and update its values
            control_panel.draw()

            # Draw the field visualization
            control_panel.draw_field_visualization()


            gl.glClearColor(0.1, 0.1, 0.1, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            imgui.render()
            impl.render(imgui.get_draw_data())
            glfw.swap_buffers(window)

    # Save config before exiting
    app_state.save_config()
    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()
