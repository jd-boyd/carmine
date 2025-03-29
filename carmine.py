import time
import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw
import cv2
import OpenGL.GL as gl
import numpy as np
import json
import os
import sys
from ultralytics import YOLO
import supervision as sv

import sources
from sources import create_opengl_texture, update_opengl_texture
from quad import Quad
from state import State
#from file_dialog import open_file_dialog, save_file_dialog


class CameraDisplay:
    """
    UI component for displaying camera view with overlays.
    """

    # The there are several relevent coordinate systems in this window.
    # The is the screen space point in the window. (initially 640x360)
    # There is the image space point in the window (probably 1920x1080)
    # There is field space, which maps the field box to a known size (initially 100x300)
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
        # Calculate mouse position in window space first
        mouse_window_x = self.mouse_x - self.window_pos_x
        mouse_window_y = self.mouse_y - self.window_pos_y

        # Scale to image space based on current zoom level
        return (int(mouse_window_x * self.scale),
                int(mouse_window_y * self.scale))


    def get_mouse_in_field_space(self):

        point_x, point_y = self.get_mouse_in_image_space()

        f_x, f_y = self.state.camera1_quad.point_to_field(point_x, point_y)

        # Scale to image space based on current zoom level
        return (f_x, f_y)


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
                for i, (field_x, field_y) in enumerate(self.state.poi_positions):
                    # Create a quad from the camera points with field size
                    try:
                        quad = Quad(self.state.camera1_points, field_size=self.state.field_size)
                        # Convert directly from field to camera coordinates
                        camera_coords = quad.field_to_point(field_x, field_y)

                        if camera_coords:
                            cam_x, cam_y = camera_coords
                            # Scale to display coordinates
                            screen_x = cursor_pos_x + (cam_x * scale_x)
                            screen_y = cursor_pos_y + (cam_y * scale_y)

                            # Determine mine color based on nearest car distance
                            marker_size = 10.0

                            # Default color (red) if no cars or can't calculate distance
                            r, g, b = 1.0, 0.0, 0.0  # Default to red

                            # Try to find the minimum distance from any car to this POI
                            min_distance = float('inf')

                            # Calculate distances if we have cars
                            if self.state.car_field_positions:
                                # Get POI position
                                poi_x, poi_y = self.state.poi_positions[i]

                                # Check each car's distance to this POI
                                for car_x, car_y in self.state.car_field_positions:
                                    # Calculate Euclidean distance
                                    dist = ((car_x - poi_x)**2 + (car_y - poi_y)**2)**0.5
                                    min_distance = min(min_distance, dist)

                            # Set color based on POI ranges (use first 3 values if available)
                            # Green: beyond the safe distance
                            # Yellow: in caution zone
                            # Red: in danger zone
                            if min_distance != float('inf'):
                                # Get thresholds from poi_ranges
                                if len(self.state.poi_ranges) >= 3:
                                    safe_distance = self.state.poi_ranges[0]
                                    caution_distance = self.state.poi_ranges[1]
                                    danger_distance = self.state.poi_ranges[2]
                                else:
                                    # Default values if not enough ranges defined
                                    safe_distance = 45
                                    caution_distance = 15
                                    danger_distance = 3

                                # Set color based on distance thresholds
                                if min_distance > safe_distance:
                                    # Green - safe
                                    r, g, b = 0.0, 1.0, 0.0
                                elif min_distance > caution_distance:
                                    # Yellow - caution
                                    r, g, b = 1.0, 1.0, 0.0
                                elif min_distance > danger_distance:
                                    # Orange - approaching danger
                                    r, g, b = 1.0, 0.5, 0.0
                                else:
                                    # Red - danger
                                    r, g, b = 1.0, 0.0, 0.0

                            # Create colors with the determined RGB values
                            white_color = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
                            mine_color = imgui.get_color_u32_rgba(r, g, b, 1.0)
                            fill_color = imgui.get_color_u32_rgba(r, g, b, 0.5)  # Semi-transparent

                            # Draw triangle (pointing upward)
                            draw_list.add_triangle(
                                screen_x, screen_y - marker_size,               # top vertex
                                screen_x - marker_size, screen_y + marker_size,  # bottom left vertex
                                screen_x + marker_size, screen_y + marker_size,  # bottom right vertex
                                mine_color, 2.0  # outline width
                            )

                            # Add filled triangle with semi-transparency
                            draw_list.add_triangle_filled(
                                screen_x, screen_y - marker_size,               # top vertex
                                screen_x - marker_size, screen_y + marker_size,  # bottom left vertex
                                screen_x + marker_size, screen_y + marker_size,  # bottom right vertex
                                fill_color
                            )

                            # Draw POI number
                            draw_list.add_text(
                                screen_x + marker_size + 2,
                                screen_y - marker_size - 2,
                                white_color,
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

                # Check if we're waiting to set a POI position (allow POI setting from camera view)
                elif self.state.waiting_for_poi_point >= 0:
                    # Convert camera coordinates to field coordinates
                    field_position = self.state.camera_to_field_position(frame_x, frame_y)

                    if field_position:
                        # Set the POI position
                        point_idx = self.state.waiting_for_poi_point
                        field_x, field_y = field_position
                        self.state.set_poi_position(point_idx, field_x, field_y)
                        print(f"Set POI {point_idx+1} to field position ({field_x:.1f}, {field_y:.1f}) from camera view")

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

                    # Mapping coordinates with horizontal flipping:
                    # - X axis is horizontal in our visualization (right to left, flipped)
                    # - Y axis is vertical in our visualization (top to bottom)
                    norm_x = 1.0 - rel_x  # Invert X-coordinate to match our flipped display
                    norm_y = rel_y

                    # Draw a small circle at the exact point that would be set
                    draw_list.add_circle_filled(
                        mouse_x, mouse_y,
                        3.0,  # 3 pixel radius
                        imgui.get_color_u32_rgba(1, 1, 0, 1),  # Yellow color
                        12  # Number of segments (smoothness)
                    )

            # Draw cursor position on field if available
            if self.state.c1_cursor_field_position:
                cursor_x, cursor_y = self.state.c1_cursor_field_position

                # Normalize cursor coordinates (since they're in field units)
                norm_cursor_x = cursor_x / self.state.field_size[0]
                norm_cursor_y = cursor_y / self.state.field_size[1]

                # Convert to canvas coordinates with horizontal flipping
                cursor_canvas_x = canvas_pos_x + ((1-norm_cursor_x) * canvas_width)
                cursor_canvas_y = canvas_pos_y + (norm_cursor_y * canvas_height)

                # Draw cursor as a plus symbol
                cursor_marker_size = 8.0
                cursor_color = imgui.get_color_u32_rgba(0, 1, 0.5, 1)  # Teal color

                # Draw plus symbol
                draw_list.add_line(
                    cursor_canvas_x - cursor_marker_size, cursor_canvas_y,
                    cursor_canvas_x + cursor_marker_size, cursor_canvas_y,
                    cursor_color, 1.5
                )
                draw_list.add_line(
                    cursor_canvas_x, cursor_canvas_y - cursor_marker_size,
                    cursor_canvas_x, cursor_canvas_y + cursor_marker_size,
                    cursor_color, 1.5
                )

                # Draw a label
                draw_list.add_text(
                    cursor_canvas_x + cursor_marker_size + 4,
                    cursor_canvas_y - 8,
                    cursor_color,
                    "Cursor"
                )

            # Draw multiple cars if available
            car_positions = self.state.car_field_positions

            # Define colors for multiple cars - create a rainbow of colors
            car_colors = [
                imgui.get_color_u32_rgba(0, 1, 1, 1),      # Cyan
                imgui.get_color_u32_rgba(1, 0.5, 0, 1),    # Orange
                imgui.get_color_u32_rgba(0, 1, 0, 1),      # Green
                imgui.get_color_u32_rgba(1, 0, 1, 1),      # Magenta
                imgui.get_color_u32_rgba(1, 1, 0, 1),      # Yellow
                imgui.get_color_u32_rgba(0, 0, 1, 1),      # Blue
                imgui.get_color_u32_rgba(1, 0, 0, 1),      # Red
                imgui.get_color_u32_rgba(0.5, 0.5, 1, 1),  # Light blue
                imgui.get_color_u32_rgba(0.5, 1, 0.5, 1),  # Light green
                imgui.get_color_u32_rgba(1, 0.5, 0.5, 1)   # Light red
            ]

            # Draw each car with a different color
            for i, car_pos in enumerate(car_positions):
                if i >= len(car_colors):
                    break  # Don't exceed the number of defined colors

                car_x, car_y = car_pos

                # Normalize the car coordinates (since they're now in field units)
                norm_car_x = car_x / self.state.field_size[0]
                norm_car_y = car_y / self.state.field_size[1]

                # Convert to canvas coordinates, flipping horizontally
                car_canvas_x = canvas_pos_x + ((1-norm_car_x) * canvas_width)  # 1-x to flip horizontally
                car_canvas_y = canvas_pos_y + (norm_car_y * canvas_height)

                # Draw a larger symbol for the car (circle with dot in center)
                car_marker_size = 7.0
                car_color = car_colors[i]

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

                # Draw car number label
                draw_list.add_text(
                    car_canvas_x + car_marker_size + 2,
                    car_canvas_y - car_marker_size - 2,
                    car_color,
                    f"CAR {i+1}"
                )

            # Get POI distances for all cars
            all_car_distances = []
            if len(self.state.car_field_positions) > 0:
                all_car_distances = self.state.calculate_all_car_poi_distances()

            # Fall back to legacy behavior for backward compatibility
            elif self.state.car_field_position is not None:
                legacy_distances = self.state.calculate_poi_distances()
                if legacy_distances:
                    all_car_distances = [legacy_distances]

            # Draw POIs with horizontal flipping
            for i, (field_x, field_y) in enumerate(self.state.poi_positions):
                # Normalize the field coordinates to 0-1 range for canvas positioning
                norm_x = field_x / self.state.field_size[0]
                norm_y = field_y / self.state.field_size[1]

                # Calculate pixel position on the canvas, flipping horizontally
                poi_x = canvas_pos_x + ((1-norm_x) * canvas_width)  # 1-norm_x to flip horizontally
                poi_y = canvas_pos_y + (norm_y * canvas_height)

                # Determine marker color based on car proximity
                marker_size = 5.0

                # Default color - red or yellow if waiting for POI point
                r, g, b = 1.0, 1.0 if i == self.state.waiting_for_poi_point else 0.0, 0.0

                # If we're not setting this POI, use distance-based coloring
                if i != self.state.waiting_for_poi_point:
                    # Try to find the minimum distance from any car to this POI
                    min_distance = float('inf')

                    # Calculate distances if we have cars
                    if self.state.car_field_positions:
                        # Check each car's distance to this POI
                        for car_x, car_y in self.state.car_field_positions:
                            # Calculate Euclidean distance
                            dist = ((car_x - field_x)**2 + (car_y - field_y)**2)**0.5
                            min_distance = min(min_distance, dist)

                    # Set color based on POI ranges thresholds
                    if min_distance != float('inf'):
                        # Get thresholds from poi_ranges
                        if len(self.state.poi_ranges) >= 3:
                            safe_distance = self.state.poi_ranges[0]
                            caution_distance = self.state.poi_ranges[1]
                            danger_distance = self.state.poi_ranges[2]
                        else:
                            # Default values if not enough ranges defined
                            safe_distance = 45
                            caution_distance = 15
                            danger_distance = 3

                        # Set color based on distance thresholds (same logic as in camera view)
                        if min_distance > safe_distance:
                            # Green - safe
                            r, g, b = 0.0, 1.0, 0.0
                        elif min_distance > caution_distance:
                            # Yellow - caution
                            r, g, b = 1.0, 1.0, 0.0
                        elif min_distance > danger_distance:
                            # Orange - approaching danger
                            r, g, b = 1.0, 0.5, 0.0
                        else:
                            # Red - danger
                            r, g, b = 1.0, 0.0, 0.0

                # Create colors with the determined RGB values
                color = imgui.get_color_u32_rgba(r, g, b, 1.0)
                fill_color = imgui.get_color_u32_rgba(r, g, b, 0.5)  # Semi-transparent
                draw_list.add_triangle_filled(
                    poi_x, poi_y - marker_size,               # top vertex
                    poi_x - marker_size, poi_y + marker_size,  # bottom left vertex
                    poi_x + marker_size, poi_y + marker_size,  # bottom right vertex
                    fill_color
                )

                # Draw POI number
                draw_list.add_text(
                    poi_x + marker_size + 2,
                    poi_y - marker_size - 2,
                    imgui.get_color_u32_rgba(1, 1, 0, 1),  # Yellow color
                    f"{i+1}"
                )

                # Draw distances from each car to this POI
                if all_car_distances:
                    # Track vertical offset for multiple distances
                    y_offset = 15

                    # Look for this POI in each car's distances
                    for car_idx, car_distances in enumerate(all_car_distances):
                        # Use same colors as cars
                        car_color = car_colors[car_idx] if car_idx < len(car_colors) else imgui.get_color_u32_rgba(1, 1, 1, 1)

                        # Find this POI in the current car's distances
                        for poi_idx, distance in car_distances:
                            if poi_idx == i:
                                # Draw distance with car number
                                draw_list.add_text(
                                    poi_x - marker_size - 25,
                                    poi_y - marker_size - y_offset,
                                    car_color,
                                    f"C{car_idx+1}: {distance:.1f}"
                                )
                                y_offset += 15  # Increment for next car
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

                    # Mapping coordinates with horizontal flipping:
                    # - X axis is horizontal in our visualization (right to left, flipped)
                    # - Y axis is vertical in our visualization (top to bottom)
                    norm_x = 1.0 - rel_x  # Invert X-coordinate to match our flipped display
                    norm_y = rel_y  # Y-coordinate in window stays as Y in our POI system

                    # Debug print - show normalized coordinates
                    print(f"Relative position in window: ({rel_x:.2f}, {rel_y:.2f})")
                    print(f"Normalized POI coordinates: ({norm_x:.2f}, {norm_y:.2f})")

                    # Check if we're waiting to set a POI position
                    if self.state.waiting_for_poi_point >= 0:
                        # Set the POI position in field coordinates
                        point_idx = self.state.waiting_for_poi_point
                        field_x = norm_x * self.state.field_size[0]
                        field_y = norm_y * self.state.field_size[1]
                        self.state.set_poi_position(point_idx, field_x, field_y)
                        print(f"Set POI {point_idx+1} to field position ({field_x:.1f}, {field_y:.1f})")

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
        self.prev_selected_camera1 = state.selected_camera1
        self.video_path = None  # Store the selected video file path

    def draw(self):
        """Draw the control panel UI and update state values"""
        reinit_camera = False

        if imgui.begin("Control Panel", True):
            imgui.text("Cameras")
            changed1, self.state.selected_camera1 = imgui.combo(
                "Camera 1", self.state.selected_camera1, [c[1] for c in self.state.camera_list]
            )
            if changed1:
                self.state.save_config()
                # Signal that camera source needs to be reinitialized
                reinit_camera = True

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

            imgui.text("Cursor pos FS: ({}, {})".format(*self.camera_display.get_mouse_in_field_space()))


            changed, checked = imgui.checkbox("Car box", self.state.c1_show_carbox)
            if changed:
                self.state.c1_show_carbox = checked
                print(f"Checkbox state changed to: {checked}")

            changed, checked = imgui.checkbox("Mines", self.state.c1_show_mines)
            if changed:
                self.state.c1_show_mines = checked
                print(f"Checkbox state changed to: {checked}")

            imgui.separator()


            imgui.text("Points of Interest")
            for i in range(len(self.state.poi_positions)):
                # Get position coordinates (now in field units)
                x, y = self.state.poi_positions[i]

                # Display the current value in field units
                imgui.text(f"Point {i+1}: ({x:.1f}, {y:.1f})")

                # Indicate if we're waiting for this point to be set
                if self.state.waiting_for_poi_point == i:
                    imgui.same_line()
                    imgui.text_colored("Click on camera or field view...", 1, 0.5, 0, 1)

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
                        print(f"Click on either the camera view or field visualization to set POI {i+1}")

            imgui.separator()

            imgui.text("PoI Ranges")
            for i in range(len(self.state.poi_ranges)):

                changed_width, self.state.poi_ranges[i] = imgui.input_int(
                    "Width", self.state.poi_ranges[i]
                )

                if changed_width:
                    self.state.save_config()


            imgui.separator()

            imgui.text("Car Status")

            # Get list of car positions
            car_positions = self.state.car_field_positions

            # Display number of detected cars
            imgui.text(f"Detected cars: {len(car_positions)}")

            # If we have cars, show their positions
            if car_positions:
                # Create a collapsible section for car details
                if imgui.collapsing_header("Car Positions", flags=imgui.TREE_NODE_DEFAULT_OPEN):
                    # Show each car's position
                    for i, (car_x, car_y) in enumerate(car_positions):
                        if i < 10:  # Limit to max cars
                            # Car positions are already in field units now
                            # Use the same colors as in field visualization
                            if i < 10:
                                r, g, b, a = 0, 0, 0, 0
                                if i == 0: r, g, b = 0, 1, 1    # Cyan
                                elif i == 1: r, g, b = 1, 0.5, 0  # Orange
                                elif i == 2: r, g, b = 0, 1, 0    # Green
                                elif i == 3: r, g, b = 1, 0, 1    # Magenta
                                elif i == 4: r, g, b = 1, 1, 0    # Yellow
                                elif i == 5: r, g, b = 0, 0, 1    # Blue
                                elif i == 6: r, g, b = 1, 0, 0    # Red
                                elif i == 7: r, g, b = 0.5, 0.5, 1  # Light blue
                                elif i == 8: r, g, b = 0.5, 1, 0.5  # Light green
                                elif i == 9: r, g, b = 1, 0.5, 0.5  # Light red

                                # Show colored information for each car
                                imgui.text_colored(f"Car {i+1}:", r, g, b, 1)
                                imgui.same_line()
                                imgui.text(f"({car_x:.1f}, {car_y:.1f}) units")
            else:
                imgui.text("No cars detected")

            imgui.separator()

            # Processing Control
            imgui.text("Processing Control")

            # Add the pause processing button
            if self.state.processing_paused:
                if imgui.button("Resume Processing", width=150, height=0):
                    self.state.processing_paused = False
                    self.state.save_config()
                    print("Processing resumed")
                # Add indicator text when paused
                imgui.same_line()
                imgui.text_colored("PAUSED", 1, 0.5, 0, 1)
            else:
                if imgui.button("Pause Processing", width=150, height=0):
                    self.state.processing_paused = True
                    self.state.save_config()
                    print("Processing paused")

            # Add a tooltip to explain what pausing does
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.text("Pause/resume YOLO and detector processing.")
                imgui.text("Video will continue to update when paused.")
                imgui.end_tooltip()

            imgui.separator()

            # Config Management
            imgui.text("Configuration")
            if imgui.button("Reload Config"):
                self.state.load_config()
                print("Configuration reloaded from file")
            
            imgui.same_line()
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

        return reinit_camera


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

tracker = sv.ByteTrack()
smoother = sv.DetectionsSmoother()
mask_annotator = sv.MaskAnnotator()
old_gray = None

def process_frame_with_yolo(source, model, quad, conf_threshold=0.25, highlighted_car=None):
    """
    Process a single frame with YOLOv8 to detect cars

    Args:
        source: Video source
        model: YOLOv8 model
        quad: Points defining the field boundary
        conf_threshold: Confidence threshold
        highlighted_car: Optional [x1, y1, x2, y2, conf, cls_id] of a car to highlight

    Returns:
        Tuple of (processed frame with detections, list of car detections)
        Car detections are in format [[x1, y1, x2, y2, conf, cls_id], ...]
    """
    p_start_time = time.time()
    # Use the frame provided by the caller
    frame = source.get_frame()

    # Skip YOLO processing if this is a PlaceholderSource
    if isinstance(source, sources.PlaceholderSource):
        return frame, []

    global old_gray

    of_frame = frame.copy()
    mask = np.zeros_like(of_frame)

    # Red cars, so red channel instead of normal gray scale conversion.
    _, _, frame_gray = cv2.split(frame)
    if not old_gray is None:
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]
        p0 = good_new.reshape(-1, 1, 2)

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
            of_frame = cv2.circle(of_frame, (int(a), int(b)), 5, (0, 0, 255), -1)
            of_frame = cv2.add(of_frame, mask)

    old_gray = frame_gray

    #frame = cv2.cvtColor(frame_gray,cv2.COLOR_GRAY2RGB)


    # YOLOv8 class names (COCO dataset)
    class_names = model.names

    # Car class ID in COCO dataset (2: car, 5: bus, 7: truck)
    vehicle_classes = [2, 7]

    # Get model prediction on the resized frame
    results = model.predict(frame, imgsz=1920, conf=conf_threshold)[0]


    polygon = np.array(quad)
    polygon_zone = sv.PolygonZone(polygon=polygon)

    #results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    #is_detections_in_zone = polygon_zone.trigger(detections)

    mask = polygon_zone.trigger(detections=detections)
    detections = detections[mask]

    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    annotated_image = mask_annotator.annotate(
        scene=of_frame.copy(), detections=detections)
    output_frame = annotated_image


    # List to store car detections (for click detection later)
    car_detections = []

    # Iterate through detections
#    for det in results.boxes.data.cpu().numpy():
    for i in range(len(detections.confidence)):
        #x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = detections.xyxy[i]
        conf = detections.confidence[i]
        cls_id = detections.class_id[i]
        cls_id = int(cls_id)

        # Scale the coordinates back to the original image size
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

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

            # Draw bounding box (yellow if highlighted, green otherwise)
            box_color = (0, 255, 255) # if is_highlighted else (0, 255, 0)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), box_color, 2)

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

    print("Processing took: ", (time.time() - p_start_time)*1000, "ms")
    return output_frame, car_detections


def main():
    camera_list = []
    for camera_info in sources.enumerate_avf_sources(): #enumerate_cameras():
        # Format: [(index, name), ...]
        print(f'{camera_info[0]}: {camera_info[1]}')
        camera_list.append(camera_info)

    #model=YOLO('yolov9s.pt')
    #model=YOLO('yolov8s.pt')
    #model=YOLO('yolo11n.pt')
    model=YOLO('yolov5nu.pt')

    window = create_glfw_window()
    imgui.create_context()
    impl = GlfwRenderer(window)

    # Initialize application state
    app_state = State(camera_list)

    # Initialize the UI components with the state
    global control_panel, field_viz, camera_display
    field_viz = FieldVisualization(app_state)

    # Initialize video sources with error handling
    camera1_id = app_state.get_camera1_id()
    try:
        source_1 = sources.AVFSource(camera1_id if camera1_id is not None else 0)
    except Exception as e:
        print(f"Error initializing camera: {e}")
        # Create a placeholder source with error message
        source_1 = sources.PlaceholderSource(
            width=1920,
            height=1080,
            message=f"Camera Error: {str(e)}"
        )

    try:
        source_2 = sources.VideoSource('../AI_angle_2.mov')
    except Exception as e:
        print(f"Couldn't open AI_angle_2.mov: {e}")
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

            # Get a fresh frame from the source
            frame = source_1.get_frame()

            # Only run YOLO processing if not paused
            if not app_state.processing_paused:
                processed_frame, car_detections = process_frame_with_yolo(source_1, model,
                                                                          app_state.camera1_points,
                                                                          highlighted_car=False)
                # Update detections only when processing is active
                app_state.set_car_detections(car_detections)
            else:
                # When paused, just use the raw frame without processing
                processed_frame = frame

            # Always update texture with the current frame (processed or raw)
            sources.update_opengl_texture(source_1.get_texture_id(), processed_frame)
            # Draw the camera view using the CameraDisplay class
            camera_display.draw()

            # Draw the control panel and update its values
            reinit_camera = control_panel.draw()

            # Check if we need to reinitialize camera source
            if reinit_camera:
                # Get the updated camera ID
                camera1_id = app_state.get_camera1_id()
                # Reinitialize camera source with the selected camera ID
                try:
                    new_source = sources.AVFSource(camera1_id if camera1_id is not None else 0)
                    # Update the camera display with the new source
                    source_1 = new_source
                    camera_display.source = new_source
                    print(f"Switched to camera {camera1_id}")
                except Exception as e:
                    error_message = f"Camera Error: {str(e)}"
                    print(f"Error switching camera: {e}")
                    # Create a placeholder source with the error message
                    new_source = sources.PlaceholderSource(
                        width=1920,
                        height=1080,
                        message=error_message
                    )
                    # Update the camera display with the placeholder source
                    source_1 = new_source
                    camera_display.source = new_source

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
