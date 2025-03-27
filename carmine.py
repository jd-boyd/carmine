import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw
import cv2
import OpenGL.GL as gl
import numpy as np
import json
import os
import sys
from cv2_enumerate_cameras import enumerate_cameras
from ultralytics import YOLO

import sources
from sources import create_opengl_texture, update_opengl_texture
from quad import Quad
from state import State


class CameraDisplay:
    """
    UI component for displaying camera view with overlays.
    """

    # The there are several relevent coordinate systems in this window.
    # The is the screen space point in the window. (initially 640x360)
    # There is the image space point in the window (probably 1920x1080)
    # Third, there is the UV space (0,0) to (1,1), for items in the field box.
    # Also drawing in the window is relative to the parent,so:
    #   draw_point = point_in_window + window_position

    def __init__(self, state, source):
        self.state = state
        self.source = source

        self.window_pos_x = 0
        self.window_pos_y = 0

        self.mouse_x = 0
        self.mouse_y = 0


    def get_mouse_in_window_space(self):
        return [self.mouse_x-self.window_pos_x, self.mouse_y-self.window_pos_y]

    def get_mouse_in_image_space(self):
        return (int((self.mouse_x-self.window_pos_x)*self.scale),
                int((self.mouse_y-self.window_pos_y)*self.scale))

    def get_mouse_in_uv_space(self):
        pt = self.get_mouse_in_image_space()
        return self.state.camera1_quad.point_to_uv(pt[0], pt[1])


    def uv_to_window_space(self):
        pass

    def draw(self):
        """
        Draw the camera view with overlays.

        Args:
            source: The video source to display
        """
        # Get texture ID for display
        tex_id = self.source.get_texture_id()

        # Get window position (needed for mouse position calculation)
        self.window_pos_x, self.window_pos_y = imgui.get_window_position()

        # Get cursor position (for content region position)
        cursor_pos_x, cursor_pos_y = imgui.get_cursor_screen_pos()

        self.mouse_x, self.mouse_y = imgui.get_io().mouse_pos
        self.state.set_c1_cursor([self.mouse_x-self.window_pos_x, self.mouse_y-self.window_pos_y])

        # Set default window size for OpenCV Image window
        default_width = 640
        self.scale = 3.0
        # Calculate the correct height based on the video's aspect ratio
        aspect_ratio = self.source.width / self.source.height
        default_height = default_width / aspect_ratio
        imgui.set_next_window_size(default_width, default_height, imgui.FIRST_USE_EVER)

        imgui.begin("Camnera 1")
        if tex_id:

            # Get available width and height of the ImGui window content area
            avail_width = imgui.get_content_region_available_width()

            self.scale = self.source.width / avail_width;

            # Calculate aspect ratio to maintain proportions
            # Set display dimensions based on available width and aspect ratio
            display_width = avail_width
            display_height = avail_width / aspect_ratio

            # Draw the image
            imgui.image(tex_id, display_width, display_height)

            draw_list = imgui.get_window_draw_list()

            if len(self.state.car_detections) > 0:
                hx1, hy1, hx2, hy2, _, _ = self.state.car_detections[0]
                hx1 /= self.scale
                hy1 /= self.scale
                hx2 /= self.scale
                hy2 /= self.scale
                # Check if this is approximately the same detection
                overlap_threshold = 0.7  # Adjust if needed
                # Check that the centers are close to each other
                h_center_x = (hx1 + hx2) // 2
                h_center_y = (hy1 + hy2) // 2

                # Draw field outline with thicker border
                draw_list.add_rect(
                    self.window_pos_x+hx1, self.window_pos_y+hy1,
                    self.window_pos_x+hx2, self.window_pos_y+hy2,

                    imgui.get_color_u32_rgba(0, 1, 1, 1),
                    0, 2.0  # No rounding, 2px thickness
                )

            # Check if centers are within a small distance
            # distance = np.sqrt((center_x - h_center_x)**2 + (center_y - h_center_y)**2)
            # if distance < 30:  # Adjust threshold as needed
            #     is_highlighted = True


            # Function to draw quad for a camera
            def draw_camera_quad(points, color):
                # Only draw if we have valid points
                if all(isinstance(p, list) and len(p) == 2 for p in points):
                    # Calculate scaling factors to map image coords to screen coords
                    scale_x = display_width / self.source.width
                    scale_y = display_height / self.source.height

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
            draw_camera_quad(self.state.camera1_points, imgui.get_color_u32_rgba(0, 1, 0, 0.8))

            # Draw POIs (mines) on the camera view
            if self.state.c1_show_mines and self.state.camera1_points and all(isinstance(p, list) and len(p) == 2 for p in self.state.camera1_points):
                # Calculate scaling factors
                scale_x = display_width / self.source.width
                scale_y = display_height / self.source.height

                # Go through each POI
                for i, (poi_x, poi_y) in enumerate(self.state.poi_positions):
                    # Convert from normalized field coordinates to camera coordinates
                    # Create a quad from the camera points
                    try:
                        quad = self.camera1_quad
                        # Convert from normalized field coordinates to camera coordinates
                        camera_coords = quad.uv_to_point(poi_x, poi_y)

                        if camera_coords:
                            cam_x, cam_y = camera_coords
                            # Scale to display coordinates
                            screen_x = cursor_pos_x + (cam_x * scale_x)
                            screen_y = cursor_pos_y + (cam_y * scale_y)

                            # Draw a red X marker for each POI
                            marker_size = 10.0
                            mine_color = imgui.get_color_u32_rgba(1, 0, 0, 1)  # Red

                            # Draw X
                            draw_list.add_line(
                                screen_x - marker_size, screen_y - marker_size,
                                screen_x + marker_size, screen_y + marker_size,
                                mine_color, 2.0
                            )
                            draw_list.add_line(
                                screen_x - marker_size, screen_y + marker_size,
                                screen_x + marker_size, screen_y - marker_size,
                                mine_color, 2.0
                            )

                            # Draw POI number
                            draw_list.add_text(
                                screen_x + marker_size + 2,
                                screen_y - marker_size - 2,
                                mine_color,
                                f"Mine {i+1}"
                            )
                    except Exception as e:
                        # Silently fail if coordinate transformation doesn't work
                        pass

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
                frame_x = int(rel_x * self.source.width)
                frame_y = int(rel_y * self.source.height)

                # Print to console
                print(f"Click at video position: x={frame_x}, y={frame_y} (relative: {rel_x:.3f}, {rel_y:.3f})")

                # Check if we're waiting to set a camera point
                if self.state.waiting_for_camera1_point >= 0:
                    # Set the camera 1 point
                    point_idx = self.state.waiting_for_camera1_point
                    self.state.set_camera_point(1, point_idx, frame_x, frame_y)
                    print(f"Set Camera 1 Point {point_idx+1} to ({frame_x}, {frame_y})")

                elif self.state.waiting_for_camera2_point >= 0:
                    # Set the camera 2 point
                    point_idx = self.state.waiting_for_camera2_point
                    self.state.set_camera_point(2, point_idx, frame_x, frame_y)
                    print(f"Set Camera 2 Point {point_idx+1} to ({frame_x}, {frame_y})")

        imgui.end()


class FieldVisualization:
    """
    UI component for visualizing the field and POI positions.
    """
    def __init__(self, state):
        self.state = state

    def draw(self):
        """Draw a visualization of the field with POI positions"""
        # Set default window size
        # Rotate the field - height is now width, and width is now height
        default_width = 400
        field_aspect_ratio = self.state.field_size[1] / self.state.field_size[0]  # Height / Width (rotated)
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

            # Draw crosshairs when waiting for POI placement
            if self.state.waiting_for_poi_point >= 0 and imgui.is_window_hovered():
                # Get mouse position
                mouse_x, mouse_y = imgui.get_io().mouse_pos

                # Draw crosshairs only if inside canvas
                if (canvas_pos_x <= mouse_x <= canvas_pos_x + canvas_width and
                    canvas_pos_y <= mouse_y <= canvas_pos_y + canvas_height):

                    # Draw vertical line
                    draw_list.add_line(
                        mouse_x, canvas_pos_y,
                        mouse_x, canvas_pos_y + canvas_height,
                        imgui.get_color_u32_rgba(0, 1, 1, 0.7),  # Cyan color
                        1.0
                    )

                    # Draw horizontal line
                    draw_list.add_line(
                        canvas_pos_x, mouse_y,
                        canvas_pos_x + canvas_width, mouse_y,
                        imgui.get_color_u32_rgba(0, 1, 1, 0.7),  # Cyan color
                        1.0
                    )

                    # Show preview of where the POI will be placed (to confirm the coordinate calculation)
                    # Calculate relative position within the canvas (0-1)
                    rel_x = (mouse_x - canvas_pos_x) / canvas_width
                    rel_y = (mouse_y - canvas_pos_y) / canvas_height

                    # For our POI storage format (which is rotated):
                    # - X axis is vertical in our visualization (top to bottom)
                    # - Y axis is horizontal in our visualization (right to left)
                    norm_x = rel_y
                    norm_y = 1.0 - rel_x

                    # Draw a small circle at the exact point that would be set
                    draw_list.add_circle_filled(
                        mouse_x, mouse_y,
                        3.0,  # 3 pixel radius
                        imgui.get_color_u32_rgba(1, 1, 0, 1),  # Yellow color
                        12  # Number of segments (smoothness)
                    )

            # Draw the highlighted car if available
            if self.state.car_field_position is not None:
                # Get car position
                car_x, car_y = self.state.car_field_position

                # Convert to canvas coordinates with rotation
                car_canvas_x = canvas_pos_x + ((1-car_y) * canvas_width)  # 1-y to flip
                car_canvas_y = canvas_pos_y + (car_x * canvas_height)

                # Draw a larger symbol for the car (circle with dot in center)
                car_marker_size = 7.0
                car_color = imgui.get_color_u32_rgba(0, 1, 1, 1)  # Cyan color

                # Draw circle
                draw_list.add_circle(
                    car_canvas_x, car_canvas_y,
                    car_marker_size,
                    car_color, 12, 2.0  # 12 segments, 2px thickness
                )

                # Draw center dot
                draw_list.add_circle_filled(
                    car_canvas_x, car_canvas_y,
                    2.0,  # Small dot
                    car_color, 6  # 6 segments
                )

                # Draw "CAR" label
                draw_list.add_text(
                    car_canvas_x + car_marker_size + 2,
                    car_canvas_y - car_marker_size - 2,
                    car_color,
                    "CAR"
                )

            # Get POI distances if car position is available
            poi_distances = None
            if self.state.car_field_position is not None:
                poi_distances = self.state.calculate_poi_distances()

            # Draw POIs - swap x and y for rotation
            for i, (y, x) in enumerate(self.state.poi_positions):
                # Calculate pixel position on the canvas with swapped coordinates
                # x becomes y and y becomes x to rotate 90 degrees
                poi_x = canvas_pos_x + ((1-y) * canvas_width)  # 1-y to flip
                poi_y = canvas_pos_y + (x * canvas_height)

                # Draw X marker
                marker_size = 5.0
                # Use different color for the POI we're currently setting
                color = imgui.get_color_u32_rgba(1, 1, 0, 1) if i == self.state.waiting_for_poi_point else imgui.get_color_u32_rgba(1, 0, 0, 1)

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

                # Draw distance above the POI if available
                if poi_distances is not None:
                    for poi_idx, distance in poi_distances:
                        if poi_idx == i:
                            draw_list.add_text(
                                poi_x - marker_size - 10,
                                poi_y - marker_size - 15,
                                imgui.get_color_u32_rgba(0, 1, 1, 1),  # Cyan color
                                f"{distance:.1f}"
                            )
                            break

            # Check for mouse clicks inside the field visualization
            if imgui.is_window_hovered() and imgui.is_window_focused() and imgui.is_mouse_clicked(0):
                mouse_x, mouse_y = imgui.get_io().mouse_pos

                print(f"Mouse position: ({mouse_x}, {mouse_y})")
                print(f"Canvas position: ({canvas_pos_x}, {canvas_pos_y}), size: {canvas_width}x{canvas_height}")

                # Check if click is inside the canvas
                if (canvas_pos_x <= mouse_x <= canvas_pos_x + canvas_width and
                    canvas_pos_y <= mouse_y <= canvas_pos_y + canvas_height):

                    # Calculate relative position within the canvas (0-1)
                    rel_x = (mouse_x - canvas_pos_x) / canvas_width
                    rel_y = (mouse_y - canvas_pos_y) / canvas_height

                    # For our POI storage format (which is rotated):
                    # - X axis is vertical in our visualization (top to bottom)
                    # - Y axis is horizontal in our visualization (right to left)
                    norm_x = rel_y  # Y-coordinate in window becomes X in our POI system
                    norm_y = 1.0 - rel_x  # X-coordinate in window (flipped) becomes Y in our POI system

                    # Debug print - show normalized coordinates
                    print(f"Relative position in window: ({rel_x:.2f}, {rel_y:.2f})")
                    print(f"Normalized POI coordinates: ({norm_x:.2f}, {norm_y:.2f})")

                    # Check if we're waiting to set a POI position
                    if self.state.waiting_for_poi_point >= 0:
                        # Set the POI position
                        point_idx = self.state.waiting_for_poi_point
                        self.state.set_poi_position(point_idx, norm_x, norm_y)
                        print(f"Set POI {point_idx+1} to normalized position ({norm_x:.2f}, {norm_y:.2f})")

            # Add some space for the canvas
            imgui.dummy(0, canvas_height + 10)

            imgui.end()


class ControlPanel:
    """
    UI component for displaying and manipulating application state.
    """
    def __init__(self, state, field_viz, camera_display):
        self.state = state
        self.field_viz = field_viz
        self.camera_display = camera_display

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
            imgui.text("Camera1 Points:")
            for i in range(4):
                # Display the current value as (x, y)
                if i % 2:
                    imgui.same_line()
                x, y = self.state.camera1_points[i]
                imgui.text(f"{i+1}: ({x}, {y})")

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


            imgui.separator()

            imgui.text("Camera1 layers:")

            imgui.text(f"Scale: {self.camera_display.scale}")

            imgui.text("Cursor pos WS: ({}, {})".format(*self.camera_display.get_mouse_in_window_space()))

            imgui.text("Cursor pos IS: ({}, {})".format(*self.camera_display.get_mouse_in_image_space()))

            imgui.text("Cursor pos UV: ({}, {})".format(*self.camera_display.get_mouse_in_uv_space()))




            changed, checked = imgui.checkbox("Car box", self.state.c1_show_carbox)
            if changed:
                self.state.c1_show_carbox = checked
                print(f"Checkbox state changed to: {checked}")

            changed, checked = imgui.checkbox("Mines", self.state.c1_show_mines)
            if changed:
                self.state.c1_show_mines = checked
                print(f"Checkbox state changed to: {checked}")




            imgui.separator()

            # # Camera 2 points with display and Set button
            # imgui.text("Camera2 Points:")
            # for i in range(4):
            #     # Display the current value as (x, y)
            #     x, y = self.state.camera2_points[i]

            #     if i % 2:
            #         imgui.same_line()

            #     imgui.text(f"{i+1}: ({x}, {y})")

            #     # Indicate if we're waiting for this point to be set
            #     if self.state.waiting_for_camera2_point == i:
            #         imgui.same_line()
            #         imgui.text_colored("Waiting for click on image...", 1, 0.5, 0, 1)

            #     # Add Set button
            #     imgui.same_line()

            #     # Change button color/text if this is the active point waiting for selection
            #     button_text = "Cancel" if self.state.waiting_for_camera2_point == i else "Set"
            #     if imgui.button(f"{button_text}##cam2_{i}"):
            #         if self.state.waiting_for_camera2_point == i:
            #             # Cancel selection mode
            #             self.state.waiting_for_camera2_point = -1
            #         else:
            #             # Enter selection mode for this point
            #             self.state.waiting_for_camera2_point = i
            #             # Reset any other waiting state
            #             self.state.waiting_for_camera1_point = -1
            #             print(f"Click on the image to set Camera 2 Point {i+1}")
            # imgui.separator()

            imgui.text("Points of Interest")
            for i in range(10):
                # Get position coordinates
                x, y = self.state.poi_positions[i]

                # Display the current value
                imgui.text(f"Point {i+1}: ({x:.2f}, {y:.2f})")

                # Indicate if we're waiting for this point to be set
                if self.state.waiting_for_poi_point == i:
                    imgui.same_line()
                    imgui.text_colored("Waiting for click on field...", 1, 0.5, 0, 1)

                # Add Set button
                imgui.same_line()

                # Change button text if this is the active point waiting for selection
                button_text = "Cancel" if self.state.waiting_for_poi_point == i else "Set"
                if imgui.button(f"{button_text}##poi_{i}"):
                    if self.state.waiting_for_poi_point == i:
                        # Cancel selection mode
                        self.state.waiting_for_poi_point = -1
                    else:
                        # Enter selection mode for this point
                        self.state.waiting_for_poi_point = i
                        # Reset any other waiting state
                        self.state.waiting_for_camera1_point = -1
                        self.state.waiting_for_camera2_point = -1
                        print(f"Click on the field visualization to set POI {i+1}")

            imgui.separator()

            imgui.text("Car Status")
            if self.state.car_field_position:
                car_x, car_y = self.state.car_field_position
                imgui.text(f"Car: ({car_x:.2f}, {car_y:.2f})")
                car_x_ft = car_x * 300
                car_y_ft = car_y * 160
                imgui.text(f"Car F: ({car_x_ft:.2f}, {car_y_ft:.2f})")
            else:
                imgui.text(f"Car: (unset)")

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


def create_glfw_window(window_name="Carmine", width=1920, height=1080):
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


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize an image to a specified width or height while preserving the aspect ratio.

    Args:
        image: Input image
        width: Target width (None to calculate from height)
        height: Target height (None to calculate from width)
        inter: Interpolation method

    Returns:
        Resized image
    """
    dim = None
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def process_frame_with_yolo(frame, model, conf_threshold=0.25, highlighted_car=None):
    """
    Process a single frame with YOLOv8 to detect cars

    Args:
        frame: Input frame
        model: YOLOv8 model
        conf_threshold: Confidence threshold
        highlighted_car: Optional [x1, y1, x2, y2, conf, cls_id] of a car to highlight

    Returns:
        Tuple of (processed frame with detections, list of car detections)
        Car detections are in format [[x1, y1, x2, y2, conf, cls_id], ...]
    """
    # Scale frame to 640px width for YOLO processing (preserving aspect ratio)
    original_frame = frame.copy()  # Keep original for display
    target_width = 640
    yolo_frame = resize_with_aspect_ratio(frame, width=target_width)

    # YOLOv8 class names (COCO dataset)
    class_names = model.names

    # Car class ID in COCO dataset (2: car, 5: bus, 7: truck)
    vehicle_classes = [2, 5, 7]

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if sys.platform == "win32":
        sys.stdout = open('NUL', 'w') # For Windows systems
    else:
        sys.stdout = open('/dev/null', 'w')  # For *nix systems
    sys.stderr = sys.stdout

    # Get model prediction on the resized frame
    results = model.predict(yolo_frame, conf=conf_threshold)[0]

    # Restore stdout
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # Use the original frame for output (full resolution)
    output_frame = original_frame.copy()

    # Scale factor to map detections back to original frame
    scale_x = original_frame.shape[1] / yolo_frame.shape[1]
    scale_y = original_frame.shape[0] / yolo_frame.shape[0]

    # List to store car detections (for click detection later)
    car_detections = []

    # Iterate through detections
    for det in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)

        # Scale the coordinates back to the original image size
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # Calculate center point of the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Check if the detected object is a vehicle
        if cls_id in vehicle_classes:
            # Store detection data for later use
            car_detections.append([x1, y1, x2, y2, conf, cls_id])

            # # Check if this is the highlighted car
            # is_highlighted = False
            # if highlighted_car is not None:
            #     hx1, hy1, hx2, hy2, _, _ = highlighted_car
            #     # Check if this is approximately the same detection
            #     overlap_threshold = 0.7  # Adjust if needed
            #     # Check that the centers are close to each other
            #     h_center_x = (hx1 + hx2) // 2
            #     h_center_y = (hy1 + hy2) // 2

            #     # Check if centers are within a small distance
            #     distance = np.sqrt((center_x - h_center_x)**2 + (center_y - h_center_y)**2)
            #     if distance < 30:  # Adjust threshold as needed
            #         is_highlighted = True

            # # Draw bounding box (yellow if highlighted, green otherwise)
            # box_color = (0, 255, 255) if is_highlighted else (0, 255, 0)
            # cv2.rectangle(output_frame, (x1, y1), (x2, y2), box_color, 2)

            # Display class name and confidence
            vehicle_type = class_names[cls_id]

            # Prepare label with vehicle type and confidence
            label = f"{vehicle_type}: {conf:.2f}"

            # Calculate label position
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y1_label = max(y1, label_size[1])

            # Draw label background
            # bg_color = (0, 255, 255) if is_highlighted else (0, 255, 0)
            # cv2.rectangle(output_frame, (x1, y1_label - label_size[1] - 5),
            #              (x1 + label_size[0], y1_label), bg_color, -1)

            # # Draw label text
            # cv2.putText(output_frame, label, (x1, y1_label - 5),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return output_frame, car_detections



def process_camera_frame(source, model, app_state):
    """
    Reads a frame from the camera source and processes it with YOLO.

    Args:
        source: The video source to read from
        model: The YOLO model for object detection
        app_state: The application state for car highlighting

    Returns:
        Tuple of (processed_frame, car_detections)
    """
    # Get the raw frame
    raw_frame = source.get_frame()

    # Process with YOLO if needed
    processed_frame, car_detections = process_frame_with_yolo(raw_frame, model, highlighted_car=False) #app_state.highlighted_car)

    # Update texture with processed frame
    sources.update_opengl_texture(source.get_texture_id(), processed_frame)

    return processed_frame, car_detections

def main():
    camera_list = []
    for camera_info in enumerate_cameras():
        desc = (camera_info.index, f'{camera_info.index}: {camera_info.name}')
        print(desc[1])
        camera_list.append(desc)

    model=YOLO('yolov9s.pt')

    window = create_glfw_window()
    imgui.create_context()
    impl = GlfwRenderer(window)

    # Initialize application state
    app_state = State(camera_list)

    # Initialize the UI components with the state
    global control_panel, field_viz, camera_display
    field_viz = FieldVisualization(app_state)

    # Initialize video sources
    #source_1 = sources.VideoSource('../AI_angles.mov')
    source_1 = sources.AVFSource(2)
    #source_1 = sources.BMSource()
    try:
        source_2 = sources.VideoSource('../AI_angle_2.mov')
    except Exception as e:
        print("Couldn't open AI_angle_2.mov.")
        source_2 = None

    camera_display = CameraDisplay(app_state, source_1)

    control_panel = ControlPanel(app_state, field_viz, camera_display)

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

            # Process camera frame (will be done even if the camera view is not displayed)
            processed_frame, car_detections = process_camera_frame(source_1, model, app_state)
            app_state.set_car_detections(car_detections)
            # Draw the camera view using the CameraDisplay class
            camera_display.draw()


            # Draw the control panel and update its values
            control_panel.draw()

            # Draw the field visualization
            field_viz.draw()


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
