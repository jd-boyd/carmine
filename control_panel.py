import imgui
import config_db

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

        # Configuration database UI state
        self.config_name_input = ""
        self.import_config_name = ""
        self.import_file_path = ""
        self.export_config_index = 0
        self.export_file_path = ""
        self.delete_config_index = 0

    def draw(self):
        """Draw the control panel UI and update state values"""
        reinit_camera = False

        if imgui.begin("Control Panel", True):
            imgui.text("Cameras")

            # Video source selection (Camera or Video file)
            changed_source, self.state.use_video_file = imgui.checkbox(
                "Use Video File", self.state.use_video_file
            )
            if changed_source:
                self.state.save_config()
                # Signal that source needs to be reinitialized
                reinit_camera = True

            if self.state.use_video_file:
                # Video file selection
                imgui.text("Video File Path:")
                _, self.state.video_file_path = imgui.input_text(
                    "##videopath", self.state.video_file_path, 256
                )

                imgui.same_line()
                if imgui.button("Browse"):
                    # Here you would ideally use a file dialog
                    # Since file_dialog.py is commented out in the imports,
                    # we'll just allow manual path entry for now
                    print("Enter the video file path manually in the text field")

                if imgui.button("Apply Video Source"):
                    if self.state.video_file_path:
                        self.state.save_config()
                        reinit_camera = True
                        print(f"Video source set to: {self.state.video_file_path}")
                    else:
                        print("Please enter a video file path")
            else:
                # Camera selection dropdown
                changed1, self.state.selected_camera1 = imgui.combo(
                    "Camera 1", self.state.selected_camera1, [c[1] for c in self.state.camera_list]
                )
                if changed1:
                    self.state.save_config()
                    # Signal that camera source needs to be reinitialized
                    reinit_camera = True

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

            # Display current config name and camera
            current_camera = ""
            if self.state.selected_camera1 < len(self.state.camera_list):
                current_camera = self.state.camera_list[self.state.selected_camera1][1]

            imgui.text_colored(
                f"Current: {self.state.current_config_name} (Camera: {current_camera})",
                0.2, 0.8, 0.2, 1.0
            )

            # Load configuration section
            if imgui.collapsing_header("Database Configs", flags=0):
                # Get list of configs from database
                configs = config_db.list_configs()

                # Display configs with load button for each
                for config_name, timestamp in configs:
                    # Format timestamp for display
                    timestamp_str = config_db.format_timestamp(timestamp)

                    # Get camera information for this config
                    config_data = config_db.load_config_from_db(config_name)
                    camera_info = ""
                    if config_data and 'camera_name' in config_data and config_data['camera_name']:
                        camera_info = f" - Camera: {config_data['camera_name']}"

                    # Create a button to load this config
                    if imgui.button(f"Load##load_{config_name}", width=60):
                        success = self.state.load_config(config_name)
                        if success:
                            print(f"Loaded configuration '{config_name}'")
                            # Set the config name input field to the loaded config name
                            # (Don't set it if the config name is "current")
                            if config_name != "current":
                                self.config_name_input = config_name

                    imgui.same_line()
                    imgui.text(f"{config_name} ({timestamp_str}){camera_info}")

                # Save new configuration
                imgui.separator()
                _, self.config_name_input = imgui.input_text(
                    "Config Name", self.config_name_input, 64
                )

                imgui.same_line()
                if imgui.button("Save As"):
                    # If no config name is entered, use "current" as the default
                    config_name = self.config_name_input.strip() or "current"

                    # Save the config with the determined name
                    success = self.state.save_config(config_name)
                    if success:
                        if config_name == "current":
                            print("Configuration saved as 'current'")
                        else:
                            print(f"Saved configuration as '{config_name}'")

                        # Only clear the input field if it's not "current"
                        if config_name != "current":
                            self.config_name_input = ""

                # Add a small vertical space
                imgui.dummy(0, 5)

                # Options to import/export from files
                imgui.separator()
                if imgui.button("Import from File"):
                    if imgui.begin_popup_modal("Import Config", True):
                        imgui.text("Enter config name and file path:")

                        _, self.import_config_name = imgui.input_text(
                            "Config Name", self.import_config_name, 64
                        )

                        _, self.import_file_path = imgui.input_text(
                            "File Path", self.import_file_path, 256
                        )

                        if imgui.button("Import", 120, 0):
                            if self.import_config_name.strip() and self.import_file_path.strip():
                                success = config_db.import_config_from_file(
                                    self.import_config_name, self.import_file_path
                                )
                                if success:
                                    print(f"Imported configuration '{self.import_config_name}' from {self.import_file_path}")
                                    self.import_config_name = ""
                                    self.import_file_path = ""
                                    imgui.close_current_popup()
                            else:
                                print("Please enter both config name and file path")

                        imgui.same_line()
                        if imgui.button("Cancel", 120, 0):
                            imgui.close_current_popup()

                        imgui.end_popup()
                    else:
                        imgui.open_popup("Import Config")

                imgui.same_line()
                if imgui.button("Export to File"):
                    if imgui.begin_popup_modal("Export Config", True):
                        imgui.text("Select config and destination file path:")

                        # Simple dropdown for configs
                        config_names = [name for name, _ in configs]
                        _, selected_idx = imgui.combo(
                            "Config", self.export_config_index, config_names
                        )
                        self.export_config_index = selected_idx

                        _, self.export_file_path = imgui.input_text(
                            "File Path", self.export_file_path, 256
                        )

                        if imgui.button("Export", 120, 0):
                            if 0 <= self.export_config_index < len(config_names) and self.export_file_path.strip():
                                selected_config = config_names[self.export_config_index]
                                success = config_db.export_config_to_file(
                                    selected_config, self.export_file_path
                                )
                                if success:
                                    print(f"Exported configuration '{selected_config}' to {self.export_file_path}")
                                    imgui.close_current_popup()
                            else:
                                print("Please select a config and enter a file path")

                        imgui.same_line()
                        if imgui.button("Cancel", 120, 0):
                            imgui.close_current_popup()

                        imgui.end_popup()
                    else:
                        imgui.open_popup("Export Config")

                # Delete config button
                imgui.same_line()
                if imgui.button("Delete Config"):
                    if imgui.begin_popup_modal("Delete Config", True):
                        imgui.text("Select config to delete:")

                        # Simple dropdown for configs
                        config_names = [name for name, _ in configs]
                        _, selected_idx = imgui.combo(
                            "Config", self.delete_config_index, config_names
                        )
                        self.delete_config_index = selected_idx

                        if imgui.button("Delete", 120, 0):
                            if 0 <= self.delete_config_index < len(config_names):
                                selected_config = config_names[self.delete_config_index]
                                if selected_config != "current":  # Prevent deleting default config
                                    success = config_db.delete_config_from_db(selected_config)
                                    if success:
                                        print(f"Deleted configuration '{selected_config}'")
                                        self.delete_config_index = 0
                                        imgui.close_current_popup()
                                else:
                                    print("Cannot delete the 'current' configuration")
                            else:
                                print("Please select a config to delete")

                        imgui.same_line()
                        if imgui.button("Cancel", 120, 0):
                            imgui.close_current_popup()

                        imgui.end_popup()
                    else:
                        imgui.open_popup("Delete Config")

            # Quick-access config buttons
            imgui.separator()
            if imgui.button("Reload Current"):
                self.state.load_config()
                print("Configuration reloaded")

            imgui.same_line()
            if imgui.button("Reset to Defaults"):
                if imgui.begin_popup_modal("Confirm Reset", True):
                    imgui.text("Are you sure you want to reset all configuration to defaults?")
                    imgui.text("This action cannot be undone.")
                    imgui.separator()

                    if imgui.button("Yes", 120, 0):
                        self.state.reset_config()
                        # Save the reset config to the database
                        self.state.save_config()
                        imgui.close_current_popup()

                    imgui.same_line()
                    if imgui.button("No", 120, 0):
                        imgui.close_current_popup()

                    imgui.end_popup()
                else:
                    imgui.open_popup("Confirm Reset")

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

            # Get cursor position in field space and format with 2 decimal places
            field_x, field_y = self.camera_display.get_mouse_in_field_space()
            imgui.text("Cursor pos FS: ({:.2f}, {:.2f})".format(field_x, field_y))


            changed, checked = imgui.checkbox("Car box", self.state.c1_show_carbox)
            if changed:
                self.state.c1_show_carbox = checked
                print(f"Checkbox state changed to: {checked}")

            changed, checked = imgui.checkbox("Points", self.state.c1_show_mines)
            if changed:
                self.state.c1_show_mines = checked
                print(f"Checkbox state changed to: {checked}")

            imgui.separator()


            imgui.text("Points of Interest")
            for i in range(len(self.state.poi_positions)):
                # Get position coordinates (now in field units)
                x, y = self.state.poi_positions[i]

                # Display point number and add input fields for X and Y
                imgui.text(f"Pt {i+1}:")

                # Create input fields for X coordinate
                imgui.same_line()
                imgui.set_next_item_width(60)  # Set width for X input
                changed_x, new_x = imgui.input_float(f"X##poi_x_{i}", x, format="%.1f")

                # Create input fields for Y coordinate
                imgui.same_line()
                imgui.set_next_item_width(60)  # Set width for Y input
                changed_y, new_y = imgui.input_float(f"Y##poi_y_{i}", y, format="%.1f")

                # Update the POI position if either value changed
                if changed_x or changed_y:
                    # Update with new values
                    if changed_x:
                        x = new_x
                    if changed_y:
                        y = new_y

                    # Update the state with the new position
                    self.state.set_poi_position(i, x, y)
                    print(f"Updated Point {i+1} position to ({x:.1f}, {y:.1f})")

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


                # Show closest car distance for this POI
                if self.state.car_field_positions:
                    # Get closest car from cached data
                    closest_info = self.state.get_closest_car_to_poi(i)
                    
                    # Display distance if we found a closest car
                    if closest_info:
                        closest_car, min_distance = closest_info
                        imgui.same_line()
                        
                        # Get color based on distance from cache
                        text_color = self.state.get_poi_distance_color(min_distance)
                        imgui.same_line()
                        imgui.text_colored(f"Car {closest_car+1}: {min_distance:.1f}",
                                         text_color[0], text_color[1], text_color[2], 1.0)

                # Indicate if we're waiting for this point to be set
                if self.state.waiting_for_poi_point == i:
                    imgui.same_line()
                    imgui.text_colored("Click on camera or field view...", 1, 0.5, 0, 1)



            imgui.separator()

            imgui.text("PoI Ranges")

            # Custom labels for each range
            range_labels = ["Safe", "Warn", "!!!"]

            # Check if we have at least 3 values
            while len(self.state.poi_ranges) < 3:
                self.state.poi_ranges.append(0)  # Add default values if needed

            # Display all three range fields on the same line
            imgui.text(f"{range_labels[0]}:")
            imgui.same_line()
            imgui.set_next_item_width(60)
            changed0, self.state.poi_ranges[0] = imgui.input_int(
                f"##range_0", self.state.poi_ranges[0]
            )

            imgui.same_line()
            imgui.text(f"{range_labels[1]}:")
            imgui.same_line()
            imgui.set_next_item_width(60)
            changed1, self.state.poi_ranges[1] = imgui.input_int(
                f"##range_1", self.state.poi_ranges[1]
            )

            imgui.same_line()
            imgui.text(f"{range_labels[2]}:")
            imgui.same_line()
            imgui.set_next_item_width(60)
            changed2, self.state.poi_ranges[2] = imgui.input_int(
                f"##range_2", self.state.poi_ranges[2]
            )

            # Add closest POI detection info if cars are present
            if self.state.car_field_positions:
                # Get the closest POI-car pair from cached data
                closest_pair = self.state.get_closest_poi_car_pair()
                
                # If we found a closest pair, display its information
                if closest_pair:
                    closest_poi, closest_car, min_distance = closest_pair
                    imgui.same_line()
                    
                    # Get color based on distance from cache
                    text_color = self.state.get_poi_distance_color(min_distance)
                    
                    imgui.text_colored(f"Closest: Point {closest_poi+1} to Car {closest_car+1}: {min_distance:.1f}", 
                                      text_color[0], text_color[1], text_color[2], 1.0)

            # Save config if any field changed
            if changed0 or changed1 or changed2:
                self.state.save_config()


            imgui.separator()

            imgui.text("Car Status")

            # Get list of car positions
            car_positions = self.state.car_field_positions

            # Display number of detected cars
            imgui.text(f"Detected cars: {len(car_positions)}")
            
            # Add Optical Flow Scaling slider
            imgui.text("Optical Flow")
            imgui.set_next_item_width(200)
            changed_flow_scale, self.state.optical_flow_scale = imgui.slider_float(
                "Scale", self.state.optical_flow_scale, 0.5, 1.0, "%.2f"
            )
            if changed_flow_scale:
                self.state.save_config()
            
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.text("Adjust the optical flow scaling factor.")
                imgui.text("Lower values process faster but with less precision.")
                imgui.text("Higher values provide more accuracy but slower processing.")
                imgui.text("Recommended: 0.75")
                imgui.end_tooltip()

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


            imgui.end()

        return reinit_camera
