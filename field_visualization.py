import imgui


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
        field_aspect_ratio = (
            self.state.field_size[1] / self.state.field_size[0]
        )  # Height / Width (rotated)
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
                canvas_pos_x,
                canvas_pos_y,
                canvas_pos_x + canvas_width,
                canvas_pos_y + canvas_height,
                imgui.get_color_u32_rgba(1, 1, 1, 1),  # White color
                0,
                2.0,  # No rounding, 2px thickness
            )

            # Draw crosshairs when waiting for POI placement
            if self.state.waiting_for_poi_point >= 0 and imgui.is_window_hovered():
                # Get mouse position
                mouse_x, mouse_y = imgui.get_io().mouse_pos

                # Draw crosshairs only if inside canvas
                if (
                    canvas_pos_x <= mouse_x <= canvas_pos_x + canvas_width
                    and canvas_pos_y <= mouse_y <= canvas_pos_y + canvas_height
                ):
                    # Draw vertical line
                    draw_list.add_line(
                        mouse_x,
                        canvas_pos_y,
                        mouse_x,
                        canvas_pos_y + canvas_height,
                        imgui.get_color_u32_rgba(0, 1, 1, 0.7),  # Cyan color
                        1.0,
                    )

                    # Draw horizontal line
                    draw_list.add_line(
                        canvas_pos_x,
                        mouse_y,
                        canvas_pos_x + canvas_width,
                        mouse_y,
                        imgui.get_color_u32_rgba(0, 1, 1, 0.7),  # Cyan color
                        1.0,
                    )

                    # Show preview of where the POI will be placed (to confirm the coordinate calculation)
                    # Calculate relative position within the canvas (0-1)
                    rel_x = (mouse_x - canvas_pos_x) / canvas_width
                    rel_y = (mouse_y - canvas_pos_y) / canvas_height

                    # Mapping coordinates with horizontal flipping:
                    # - X axis is horizontal in our visualization (right to left, flipped)
                    # - Y axis is vertical in our visualization (top to bottom)
                    norm_x = (
                        1.0 - rel_x
                    )  # Invert X-coordinate to match our flipped display
                    norm_y = rel_y

                    # Draw a small circle at the exact point that would be set
                    draw_list.add_circle_filled(
                        mouse_x,
                        mouse_y,
                        3.0,  # 3 pixel radius
                        imgui.get_color_u32_rgba(1, 1, 0, 1),  # Yellow color
                        12,  # Number of segments (smoothness)
                    )

            # Draw cursor position on field if available
            if self.state.c1_cursor_field_position:
                cursor_x, cursor_y = self.state.c1_cursor_field_position

                # Normalize cursor coordinates (since they're in field units)
                norm_cursor_x = cursor_x / self.state.field_size[0]
                norm_cursor_y = cursor_y / self.state.field_size[1]

                # Convert to canvas coordinates with horizontal flipping
                cursor_canvas_x = canvas_pos_x + ((1 - norm_cursor_x) * canvas_width)
                cursor_canvas_y = canvas_pos_y + (norm_cursor_y * canvas_height)

                # Draw cursor as a plus symbol
                cursor_marker_size = 8.0
                cursor_color = imgui.get_color_u32_rgba(0, 1, 0.5, 1)  # Teal color

                # Draw plus symbol
                draw_list.add_line(
                    cursor_canvas_x - cursor_marker_size,
                    cursor_canvas_y,
                    cursor_canvas_x + cursor_marker_size,
                    cursor_canvas_y,
                    cursor_color,
                    1.5,
                )
                draw_list.add_line(
                    cursor_canvas_x,
                    cursor_canvas_y - cursor_marker_size,
                    cursor_canvas_x,
                    cursor_canvas_y + cursor_marker_size,
                    cursor_color,
                    1.5,
                )

                # Draw a label
                draw_list.add_text(
                    cursor_canvas_x + cursor_marker_size + 4,
                    cursor_canvas_y - 8,
                    cursor_color,
                    "Cursor",
                )

            # Draw multiple cars if available
            car_positions = self.state.car_field_positions

            # Define colors for multiple cars - create a rainbow of colors
            car_colors = [
                imgui.get_color_u32_rgba(0, 1, 1, 1),  # Cyan
                imgui.get_color_u32_rgba(1, 0.5, 0, 1),  # Orange
                imgui.get_color_u32_rgba(0, 1, 0, 1),  # Green
                imgui.get_color_u32_rgba(1, 0, 1, 1),  # Magenta
                imgui.get_color_u32_rgba(1, 1, 0, 1),  # Yellow
                imgui.get_color_u32_rgba(0, 0, 1, 1),  # Blue
                imgui.get_color_u32_rgba(1, 0, 0, 1),  # Red
                imgui.get_color_u32_rgba(0.5, 0.5, 1, 1),  # Light blue
                imgui.get_color_u32_rgba(0.5, 1, 0.5, 1),  # Light green
                imgui.get_color_u32_rgba(1, 0.5, 0.5, 1),  # Light red
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
                car_canvas_x = canvas_pos_x + (
                    (1 - norm_car_x) * canvas_width
                )  # 1-x to flip horizontally
                car_canvas_y = canvas_pos_y + (norm_car_y * canvas_height)

                # Draw a larger symbol for the car (circle with dot in center)
                car_marker_size = 7.0
                car_color = car_colors[i]

                # Draw circle
                draw_list.add_circle(
                    car_canvas_x,
                    car_canvas_y,
                    car_marker_size,
                    car_color,
                    12,
                    2.0,  # 12 segments, 2px thickness
                )

                # Draw center dot
                draw_list.add_circle_filled(
                    car_canvas_x,
                    car_canvas_y,
                    2.0,  # Small dot
                    car_color,
                    6,  # 6 segments
                )

                # Draw car number label
                draw_list.add_text(
                    car_canvas_x + car_marker_size + 2,
                    car_canvas_y - car_marker_size - 2,
                    car_color,
                    f"CAR {i + 1}",
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
                poi_x = canvas_pos_x + (
                    (1 - norm_x) * canvas_width
                )  # 1-norm_x to flip horizontally
                poi_y = canvas_pos_y + (norm_y * canvas_height)

                # Determine marker color based on car proximity
                marker_size = 5.0

                # Default color - red or yellow if waiting for POI point
                r, g, b = (
                    1.0,
                    1.0 if i == self.state.waiting_for_poi_point else 0.0,
                    0.0,
                )

                # If we're not setting this POI, use distance-based coloring
                if i != self.state.waiting_for_poi_point:
                    # Get closest car from cached data
                    closest_info = self.state.get_closest_car_to_poi(i)

                    # Get minimum distance if a closest car was found
                    min_distance = float("inf")
                    if closest_info:
                        _, min_distance = closest_info

                    # Set color based on distance if a car was found
                    if min_distance != float("inf"):
                        # Use the state's color calculation method
                        r, g, b = self.state.get_poi_distance_color(min_distance)

                # Create colors with the determined RGB values
                fill_color = imgui.get_color_u32_rgba(r, g, b, 0.5)  # Semi-transparent
                draw_list.add_triangle_filled(
                    poi_x,
                    poi_y - marker_size,  # top vertex
                    poi_x - marker_size,
                    poi_y + marker_size,  # bottom left vertex
                    poi_x + marker_size,
                    poi_y + marker_size,  # bottom right vertex
                    fill_color,
                )

                # Draw POI number
                draw_list.add_text(
                    poi_x + marker_size + 2,
                    poi_y - marker_size - 2,
                    imgui.get_color_u32_rgba(1, 1, 0, 1),  # Yellow color
                    f"{i + 1}",
                )

                # Draw distances from each car to this POI
                if all_car_distances:
                    # Track vertical offset for multiple distances
                    y_offset = 15

                    # Look for this POI in each car's distances
                    for car_idx, car_distances in enumerate(all_car_distances):
                        # Use same colors as cars
                        car_color = (
                            car_colors[car_idx]
                            if car_idx < len(car_colors)
                            else imgui.get_color_u32_rgba(1, 1, 1, 1)
                        )

                        # Find this POI in the current car's distances
                        for poi_idx, distance in car_distances:
                            if poi_idx == i:
                                # Draw distance with car number
                                draw_list.add_text(
                                    poi_x - marker_size - 25,
                                    poi_y - marker_size - y_offset,
                                    car_color,
                                    f"C{car_idx + 1}: {distance:.1f}",
                                )
                                y_offset += 15  # Increment for next car
                                break

            # Check for mouse clicks inside the field visualization
            if (
                imgui.is_window_hovered()
                and imgui.is_window_focused()
                and imgui.is_mouse_clicked(0)
            ):
                mouse_x, mouse_y = imgui.get_io().mouse_pos

                print(f"Mouse position: ({mouse_x}, {mouse_y})")
                print(
                    f"Canvas position: ({canvas_pos_x}, {canvas_pos_y}), size: {canvas_width}x{canvas_height}"
                )

                # Check if click is inside the canvas
                if (
                    canvas_pos_x <= mouse_x <= canvas_pos_x + canvas_width
                    and canvas_pos_y <= mouse_y <= canvas_pos_y + canvas_height
                ):
                    # Calculate relative position within the canvas (0-1)
                    rel_x = (mouse_x - canvas_pos_x) / canvas_width
                    rel_y = (mouse_y - canvas_pos_y) / canvas_height

                    # Mapping coordinates with horizontal flipping:
                    # - X axis is horizontal in our visualization (right to left, flipped)
                    # - Y axis is vertical in our visualization (top to bottom)
                    norm_x = (
                        1.0 - rel_x
                    )  # Invert X-coordinate to match our flipped display
                    norm_y = (
                        rel_y  # Y-coordinate in window stays as Y in our POI system
                    )

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
                        print(
                            f"Set POI {point_idx + 1} to field position ({field_x:.1f}, {field_y:.1f})"
                        )

            # Add some space for the canvas
            imgui.dummy(0, canvas_height + 10)

            imgui.end()
