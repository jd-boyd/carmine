import time
import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw
import cv2
import OpenGL.GL as gl
import numpy as np
from ultralytics import YOLO
import supervision as sv

import sources
from quad import Quad
from state import State
from field_visualization import FieldVisualization
from control_panel import ControlPanel
# from file_dialog import open_file_dialog, save_file_dialog


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
        return [self.mouse_x - self.window_pos_x, self.mouse_y - self.window_pos_y]

    def get_mouse_in_image_space(self):
        # Calculate mouse position in window space first
        mouse_window_x = self.mouse_x - self.window_pos_x
        mouse_window_y = self.mouse_y - self.window_pos_y

        # Scale to image space based on current zoom level
        return (int(mouse_window_x * self.scale), int(mouse_window_y * self.scale))

    def get_mouse_in_field_space(self):
        point_x, point_y = self.get_mouse_in_image_space()

        ret = self.state.camera1_quad.point_to_field(point_x, point_y)
        if ret is None:
            return 0, 0
        f_x, f_y = ret

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

        # Get content region position in absolute screen coordinates
        # This is the upper-left corner of where the content starts
        # This automatically updates when the window moves
        cursor_pos_x, cursor_pos_y = imgui.get_cursor_screen_pos()

        self.mouse_x, self.mouse_y = imgui.get_io().mouse_pos
        self.state.set_c1_cursor(
            [self.mouse_x - self.window_pos_x, self.mouse_y - self.window_pos_y]
        )

        # Set default window size for OpenCV Image window
        default_width = 640
        self.scale = 3.0
        # Calculate the correct height based on the video's aspect ratio
        aspect_ratio = self.source.width / self.source.height
        default_height = default_width / aspect_ratio
        imgui.set_next_window_size(default_width, default_height, imgui.FIRST_USE_EVER)

        imgui.begin("Camera")

        # Calculate if the mouse is inside the image and its position in image coordinates
        if tex_id:
            avail_width = imgui.get_content_region_available_width()
            self.scale = self.source.width / avail_width
            aspect_ratio = self.source.width / self.source.height
            display_width = avail_width
            display_height = avail_width / aspect_ratio

            # Check if mouse is inside the image area
            mouse_in_image = (
                self.window_pos_x <= self.mouse_x <= self.window_pos_x + display_width
                and self.window_pos_y
                <= self.mouse_y
                <= self.window_pos_y + display_height
            )

            if mouse_in_image:
                # Calculate relative position in the image (0-1)
                rel_x = (self.mouse_x - self.window_pos_x) * self.scale
                rel_y = (self.mouse_y - self.window_pos_y) * self.scale

                # Convert to image pixel coordinates
                img_x = int(rel_x)  # * self.source.width)
                img_y = int(rel_y)  # * self.source.height)

                # Update cursor position in image space
                self.state.c1_cursor_image_pos = (img_x, img_y)
                self.state.c1_cursor_in_image = True
            #                print(f"Cursor in image: ({img_x}, {img_y})")
            else:
                self.state.c1_cursor_image_pos = None
                self.state.c1_cursor_in_image = False

                # print("Window Pos: ", self.window_pos_x, self.window_pos_y)
                # print("Mouse Pos: ", self.mouse_x, self.mouse_y)
                # print("Display: ", display_width, display_height)
                # print("Window Limit: ", self.window_pos_x + display_width, self.window_pos_y + display_height)

                # print("Cursor not in image")

            # Calculate aspect ratio to maintain proportions
            # Set display dimensions based on available width and aspect ratio
            display_width = avail_width
            display_height = avail_width / aspect_ratio

            # Draw the image
            imgui.image(tex_id, display_width, display_height)

            # Get the window draw list to ensure drawing is relative to the current window
            draw_list = imgui.get_window_draw_list()

            # Draw POIs (points) on the camera view
            if (
                self.state.c1_show_mines
                and self.state.camera1_points
                and all(
                    isinstance(p, list) and len(p) == 2
                    for p in self.state.camera1_points
                )
            ):
                # Calculate scaling factors
                scale_x = display_width / self.source.width
                scale_y = display_height / self.source.height

                # Go through each POI
                for i, (field_x, field_y) in enumerate(self.state.poi_positions):
                    # Create a quad from the camera points with field size
                    try:
                        quad = Quad(
                            self.state.camera1_points, field_size=self.state.field_size
                        )
                        # Convert directly from field to camera coordinates
                        camera_coords = quad.field_to_point(field_x, field_y)

                        if camera_coords:
                            cam_x, cam_y = camera_coords
                            # Scale to display coordinates
                            # cursor_pos_x/y from get_cursor_screen_pos() already includes window position
                            # so this will move correctly when the window moves
                            screen_x = cursor_pos_x + (cam_x * scale_x)
                            screen_y = cursor_pos_y + (cam_y * scale_y)

                            # Determine point color based on nearest car distance
                            marker_size = 10.0

                            # Default color (red) if no cars or can't calculate distance
                            r, g, b = 1.0, 0.0, 0.0  # Default to red

                            # Try to find the minimum distance from any car to this POI
                            min_distance = float("inf")

                            # Calculate distances if we have cars
                            if self.state.car_field_positions:
                                # Get POI position
                                poi_x, poi_y = self.state.poi_positions[i]

                                # Check each car's distance to this POI
                                for car_x, car_y in self.state.car_field_positions:
                                    # Calculate Euclidean distance
                                    dist = (
                                        (car_x - poi_x) ** 2 + (car_y - poi_y) ** 2
                                    ) ** 0.5
                                    min_distance = min(min_distance, dist)

                            # Set color based on POI ranges (use first 3 values if available)
                            # Green: beyond the safe distance
                            # Yellow: in caution zone
                            # Red: in danger zone
                            if min_distance != float("inf"):
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
                                    r, g, b = 0.0, 1.0, 1.0
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
                            point_color = imgui.get_color_u32_rgba(r, g, b, 1.0)
                            fill_color = imgui.get_color_u32_rgba(
                                r, g, b, 0.5
                            )  # Semi-transparent

                            # Draw triangle (pointing upward)
                            draw_list.add_triangle(
                                screen_x,
                                screen_y - marker_size,  # top vertex
                                screen_x - marker_size,
                                screen_y + marker_size,  # bottom left vertex
                                screen_x + marker_size,
                                screen_y + marker_size,  # bottom right vertex
                                point_color,
                                2.0,  # outline width
                            )

                            # Add filled triangle with semi-transparency
                            draw_list.add_triangle_filled(
                                screen_x,
                                screen_y - marker_size,  # top vertex
                                screen_x - marker_size,
                                screen_y + marker_size,  # bottom left vertex
                                screen_x + marker_size,
                                screen_y + marker_size,  # bottom right vertex
                                fill_color,
                            )

                            # Draw POI number
                            draw_list.add_text(
                                screen_x + marker_size + 2,
                                screen_y - marker_size - 2,
                                white_color,
                                f"Point {i + 1}",
                            )
                    except Exception:
                        # Silently fail if coordinate transformation doesn't work
                        pass

            # Crosshairs are now drawn directly on the frame in FrameProcessor.annotate_frame

            # Check for mouse clicks inside the image
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(
                0
            ):  # 0 = left mouse button
                # Get mouse position
                mouse_x, mouse_y = imgui.get_io().mouse_pos

                # Calculate relative position within the image
                rel_x = (mouse_x - cursor_pos_x) / display_width
                rel_y = (mouse_y - cursor_pos_y) / display_height

                # Convert to original video frame coordinates
                frame_x = int(rel_x * self.source.width)
                frame_y = int(rel_y * self.source.height)

                # Print to console
                print(
                    f"Click at video position: x={frame_x}, y={frame_y} (relative: {rel_x:.3f}, {rel_y:.3f})"
                )

                # Check if we're waiting to set a camera point
                if self.state.waiting_for_camera1_point >= 0:
                    # Set the camera 1 point
                    point_idx = self.state.waiting_for_camera1_point
                    self.state.set_camera_point(1, point_idx, frame_x, frame_y)
                    print(
                        f"Set Camera 1 Point {point_idx + 1} to ({frame_x}, {frame_y})"
                    )

                # Check if we're waiting to set a POI position (allow POI setting from camera view)
                elif self.state.waiting_for_poi_point >= 0:
                    # Convert camera coordinates to field coordinates
                    field_position = self.state.camera_to_field_position(
                        frame_x, frame_y
                    )

                    if field_position:
                        # Set the POI position
                        point_idx = self.state.waiting_for_poi_point
                        field_x, field_y = field_position
                        self.state.set_poi_position(point_idx, field_x, field_y)
                        print(
                            f"Set POI {point_idx + 1} to field position ({field_x:.1f}, {field_y:.1f}) from camera view"
                        )

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


class FrameProcessor:
    """
    Class to handle frame processing, detection, and visualization.
    """

    def __init__(self):
        # Initialize tracking and visualization components
        self.tracker = sv.ByteTrack()
        self.smoother = sv.DetectionsSmoother()
        self.mask_annotator = sv.MaskAnnotator()
        self.old_gray = None
        self.old_scaled_gray = (
            None  # Scaled-resolution grayscale for faster optical flow
        )

        # Initialize flow fields
        self.flow = None  # Will store optical flow data
        self.flow_scale = 1.0  # Will be set dynamically

        # Flow analysis settings
        self.flow_verbose = True  # Set to True for detailed flow logging
        # Flow scale will be set dynamically based on state.optical_flow_scale

        # COCO dataset vehicle classes (2: car, 5: bus, 7: truck)
        self.vehicle_classes = [2, 7]

    def process_frame(self, source, model, quad, conf_threshold=0.25):
        """
        Process a single frame to detect cars without visualization

        Args:
            source: Video source
            model: YOLO model
            quad: Points defining the field boundary
            conf_threshold: Confidence threshold

        Returns:
            Tuple of (frame, detections, car_detections)
            - frame: The original frame from the source
            - detections: Supervision Detections object
            - car_detections: List of [x1, y1, x2, y2, conf, cls_id] for vehicles
        """
        p_start_time = time.time()

        # Get the frame from source
        frame = source.get_frame()

        # Skip YOLO processing if this is a PlaceholderSource
        if isinstance(source, sources.PlaceholderSource):
            return frame, None, []

        # Apply optical flow visualization
        of_frame = frame.copy()
        mask = np.zeros_like(of_frame)

        # Red cars, so red channel instead of normal gray scale conversion for better vehicle contrast
        _, _, frame_gray = cv2.split(frame)

        # Use the app_state's optical flow scale factor if available
        flow_scale_factor = 0.75  # Default value
        if hasattr(source, "state") and hasattr(source.state, "optical_flow_scale"):
            flow_scale_factor = source.state.optical_flow_scale

        # Create scaled image for faster optical flow processing
        scaled_width = int(frame.shape[1] * flow_scale_factor)
        scaled_height = int(frame.shape[0] * flow_scale_factor)
        scaled_gray = cv2.resize(
            frame_gray, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA
        )

        # Update flow scale factor
        self.flow_scale = (
            1.0 / flow_scale_factor
        )  # Scale factor to convert from scaled flow to full-res coordinates

        f_start_time = time.time()
        if (
            hasattr(self, "old_scaled_gray")
            and self.old_scaled_gray is not None
            and self.old_gray is not None
        ):
            # Calculate sparse optical flow for visualization (still using full resolution)
            lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
            p0 = cv2.goodFeaturesToTrack(
                self.old_gray,
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7,
            )
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.old_gray, frame_gray, p0, None, **lk_params
            )

            good_new = p1[st == 1]
            good_old = p0[st == 1]
            p0 = good_new.reshape(-1, 1, 2)

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(
                    mask, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2
                )
                of_frame = cv2.circle(of_frame, (int(a), int(b)), 5, (0, 0, 255), -1)
                of_frame = cv2.add(of_frame, mask)

            # Calculate dense optical flow on scaled-size images for better performance
            scaled_flow = cv2.calcOpticalFlowFarneback(
                self.old_scaled_gray, scaled_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # Scale the 3/4-resolution flow to full resolution
            self.flow = cv2.resize(
                scaled_flow,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

            # Multiply flow vectors by scale factor to account for the scaling
            self.flow *= self.flow_scale
        else:
            # Initialize flow with zeros if this is the first frame
            self.flow = np.zeros((frame.shape[0], frame.shape[1], 2), dtype=np.float32)

        # Save both full and scaled resolution grayscale images for next frame
        self.old_gray = frame_gray
        self.old_scaled_gray = scaled_gray

        print("Flow time: ", int((time.time() - f_start_time) * 1000))

        # Get model prediction
        #
        results = model.predict(frame, imgsz=1280, conf=conf_threshold)[0]

        # Filter detections within the defined quadrilateral
        polygon = np.array(quad)
        polygon_zone = sv.PolygonZone(polygon=polygon)

        detections = sv.Detections.from_ultralytics(results)
        mask = polygon_zone.trigger(detections=detections)
        detections = detections[mask]

        # Apply tracking and smoothing
        # detections = self.tracker.update_with_detections(detections)
        # detections = self.smoother.update_with_detections(detections)

        # Extract vehicle detections
        car_detections = []

        for i in range(len(detections.confidence)):
            x1, y1, x2, y2 = detections.xyxy[i]
            conf = detections.confidence[i]
            cls_id = int(detections.class_id[i])

            # Scale the coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Check if the detected object is a vehicle
            if cls_id in self.vehicle_classes:
                car_detections.append([x1, y1, x2, y2, conf, cls_id])

        processing_time = (time.time() - p_start_time) * 1000
        print(f"Processing took: {processing_time:.2f} ms")

        # Return the original frame, detections, and car_detections
        return of_frame, detections, car_detections

    def annotate_frame(
        self,
        frame,
        model,
        detections,
        car_detections,
        quad_points=None,
        cursor_pos=None,
    ):
        """
        Annotate a processed frame with detection visualizations and optional quad overlay

        Args:
            frame: The frame to annotate
            model: YOLO model (for class names)
            detections: Supervision Detections object
            car_detections: List of car detections
            quad_points: Optional list of quad corner points to draw
            cursor_pos: Optional cursor position in image space for drawing crosshairs

        Returns:
            Annotated frame
        """
        # Start with a copy of the frame
        output_frame = frame.copy()

        # If we have detections, add them to the frame
        if detections is not None:
            # Annotate with mask annotator
            output_frame = self.mask_annotator.annotate(
                scene=output_frame, detections=detections
            )

        # Get class names for car detections if model is provided
        class_names = model.names if model is not None else {}

        # Draw quad overlay if provided
        if quad_points is not None:
            # Use OpenCV to draw the quad on the frame
            if all(isinstance(p, list) and len(p) == 2 for p in quad_points):
                # Draw the quad as connected lines
                for i in range(4):
                    next_i = (i + 1) % 4
                    x1, y1 = quad_points[i]
                    x2, y2 = quad_points[next_i]

                    # Convert to integers
                    x1, y1 = int(x1), int(y1)
                    x2, y2 = int(x2), int(y2)

                    # Draw line on the frame (bright green)
                    cv2.line(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add numbers to corners for reference
                for i, (x, y) in enumerate(quad_points):
                    # Convert to integers
                    x, y = int(x), int(y)

                    # Draw number (white text)
                    cv2.putText(
                        output_frame,
                        f"{i + 1}",
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

        # Draw crosshairs if cursor position is provided
        if cursor_pos is not None:
            print(f"Drawing crosshairs at {cursor_pos}")
            img_x, img_y = cursor_pos

            # Make sure coordinates are within image bounds
            h, w = output_frame.shape[:2]
            if 0 <= img_x < w and 0 <= img_y < h:
                # Draw clear bright yellow crosshairs that will be visible over any background

                # Vertical line (bright yellow)
                cv2.line(
                    output_frame,
                    (img_x, 0),
                    (img_x, h),
                    (0, 255, 255),  # Yellow (BGR format)
                    1,  # Even thicker line for better visibility
                )

                # Horizontal line (bright yellow)
                cv2.line(
                    output_frame,
                    (0, img_y),
                    (w, img_y),
                    (0, 255, 255),  # Yellow (BGR format)
                    1,  # Even thicker line for better visibility
                )

                # Add a white dot at the center of the crosshairs
                cv2.circle(
                    output_frame,
                    (img_x, img_y),
                    5,  # 5 pixel radius
                    (255, 255, 255),  # White
                    -1,  # Filled circle
                )

                # Add a black outline to make it stand out
                cv2.circle(
                    output_frame,
                    (img_x, img_y),
                    5,  # 5 pixel radius
                    (0, 0, 0),  # Black
                    1,  # 1 pixel outline
                )

        # Add car bounding boxes if available
        for detection in car_detections:
            # Check if this detection includes flow data
            if len(detection) >= 8:
                x1, y1, x2, y2, conf, cls_id, flow_x, flow_y = detection
                has_flow = True
            else:
                x1, y1, x2, y2, conf, cls_id = detection
                has_flow = False
                flow_x, flow_y = 0, 0

            # Draw bounding box (yellow)
            box_color = (0, 255, 255)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), box_color, 2)

            # If we have flow data, draw the flow vector
            if has_flow and (abs(flow_x) > 0.5 or abs(flow_y) > 0.5):
                center_x = x1 + (x2 - x1) // 2
                center_y = y1 + (y2 - y1) // 2

                # Draw arrow indicating direction and magnitude of flow
                end_x = int(center_x + flow_x * 10)  # Scale for visibility
                end_y = int(center_y + flow_y * 10)

                cv2.arrowedLine(
                    output_frame,
                    (center_x, center_y),
                    (end_x, end_y),
                    (255, 0, 0),  # Blue color for flow vectors
                    2,
                )

            # Get vehicle type
            vehicle_type = class_names[cls_id]

            # Prepare label with vehicle type and confidence
            label = f"{vehicle_type}: {conf:.2f}"

            # Calculate label position
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        return output_frame

    def combine_detections_with_flow(self, car_detections, flow=None, verbose=None):
        """
        Combine car detections with optical flow data to enhance tracking

        Args:
            car_detections: List of [x1, y1, x2, y2, conf, cls_id] for detected vehicles
            flow: Optional optical flow data. If None, uses self.flow calculated during processing
            verbose: Override default verbosity setting (self.flow_verbose)

        Returns:
            List of enhanced car detections with flow information
            Format: [x1, y1, x2, y2, conf, cls_id, mean_flow_x, mean_flow_y]
        """
        # Set verbosity
        if verbose is None:
            verbose = self.flow_verbose

        if flow is None:
            # Use self.flow if it exists, otherwise set to None
            flow = self.flow if hasattr(self, "flow") else None

        if flow is None or not car_detections:
            # Return original detections with zero flow if no flow data is available
            if verbose:
                print("No flow data available or no detections to process")
            return [
                (x1, y1, x2, y2, conf, cls, 0, 0)
                for x1, y1, x2, y2, conf, cls in car_detections
            ]

        enhanced_detections = []

        if verbose:
            print(f"\n--- Flow Analysis for {len(car_detections)} Detections ---")

        # Process each detection
        for i, det in enumerate(car_detections):
            x1, y1, x2, y2, conf, cls = det

            # Ensure coordinates are integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Make sure the ROI is within image bounds
            h, w = flow.shape[0:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Skip if ROI is invalid
            if x1 >= x2 or y1 >= y2:
                if verbose:
                    print(
                        f"Detection #{i + 1}: Invalid ROI dimensions, skipping flow analysis"
                    )
                enhanced_detections.append((x1, y1, x2, y2, conf, cls, 0, 0))
                continue

            # Extract ROI from flow field
            try:
                roi_flow = flow[y1:y2, x1:x2]

                # Calculate average flow in ROI
                mean_flow_x = np.mean(roi_flow[:, :, 0])
                mean_flow_y = np.mean(roi_flow[:, :, 1])

                # Calculate flow magnitude and direction
                flow_magnitude = np.sqrt(mean_flow_x**2 + mean_flow_y**2)
                flow_direction = np.arctan2(mean_flow_y, mean_flow_x) * 180 / np.pi

                # Calculate predicted position
                new_center_x = (x1 + x2) // 2 + mean_flow_x
                new_center_y = (y1 + y2) // 2 + mean_flow_y

                # Log flow information if verbose
                if verbose:
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    vehicle_type = (
                        "car" if cls == 2 else "truck" if cls == 7 else f"class_{cls}"
                    )

                    print(f"Detection #{i + 1} ({vehicle_type}, conf: {conf:.2f}): ")
                    if flow_magnitude > 0.5:
                        print(f"  Flow: ({mean_flow_x:.2f}, {mean_flow_y:.2f}) pixels")
                        print(
                            f"  Magnitude: {flow_magnitude:.2f} pixels, Direction: {flow_direction:.1f}°"
                        )
                        print(
                            f"  Predicted center: ({center_x}, {center_y}) → ({new_center_x:.1f}, {new_center_y:.1f})"
                        )
                    else:
                        print(f"  Minimal flow detected ({flow_magnitude:.2f} pixels)")

                # Add to enhanced detections with flow values
                enhanced_detections.append(
                    (x1, y1, x2, y2, conf, cls, mean_flow_x, mean_flow_y)
                )
            except Exception as e:
                if verbose:
                    print(f"Detection #{i + 1}: Error processing flow: {e}")
                enhanced_detections.append((x1, y1, x2, y2, conf, cls, 0, 0))

        if verbose:
            print("--- End of Flow Analysis ---\n")

        return enhanced_detections

    def predict_disappeared_detections(
        self, current_detections, previous_detections, flow=None, verbose=None
    ):
        """
        Predict positions of detections that disappeared between frames using optical flow

        Args:
            current_detections: List of current frame detections
            previous_detections: List of detections from previous frame
            flow: Optional optical flow data. If None, uses self.flow calculated during processing
            verbose: Override default verbosity setting

        Returns:
            List of predicted detections from previous frame that are not in current frame
        """
        # Set verbosity
        if verbose is None:
            verbose = self.flow_verbose

        if flow is None:
            # Use self.flow if it exists, otherwise set to None
            flow = self.flow if hasattr(self, "flow") else None

        if flow is None or not previous_detections:
            # Return empty list if no flow data or previous detections
            if verbose:
                print("No flow data or previous detections available for prediction")
            return []

        # Get centers of current detections to check for overlaps
        current_centers = []
        for det in current_detections:
            x1, y1, x2, y2 = det[0:4]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            current_centers.append((center_x, center_y))

        predicted_detections = []

        if verbose:
            print(
                f"\n--- Predicting Disappeared Detections ({len(previous_detections)} previous, {len(current_detections)} current) ---"
            )

        # For each previous detection, check if it disappeared
        for i, det in enumerate(previous_detections):
            try:
                # Handle detections with or without flow data
                if len(det) >= 8:
                    x1, y1, x2, y2, conf, cls_id, prev_flow_x, prev_flow_y = det
                    # Include previous flow information
                    has_previous_flow = True
                else:
                    x1, y1, x2, y2, conf, cls_id = det[0:6]
                    prev_flow_x, prev_flow_y = 0, 0
                    has_previous_flow = False

                # Get center of this previous detection
                prev_center_x = (x1 + x2) // 2
                prev_center_y = (y1 + y2) // 2

                # Check if this detection is already in current detections (within a threshold)
                detection_exists = False
                threshold = 30  # Pixel distance threshold to consider as same detection

                for cur_center_x, cur_center_y in current_centers:
                    dist = (
                        (cur_center_x - prev_center_x) ** 2
                        + (cur_center_y - prev_center_y) ** 2
                    ) ** 0.5
                    if dist < threshold:
                        detection_exists = True
                        break

                # If detection doesn't exist in current frame, use flow to predict new position
                if not detection_exists:
                    # Calculate bounding box width and height
                    width = x2 - x1
                    height = y2 - y1

                    # Sample optical flow at previous center
                    h, w = flow.shape[0:2]
                    if 0 <= prev_center_x < w and 0 <= prev_center_y < h:
                        # Get flow vector at this position
                        # Note: No need to adjust for flow_scale here as the flow values were already
                        # scaled up when we resized the flow field from half resolution to full resolution
                        flow_x = flow[prev_center_y, prev_center_x, 0]
                        flow_y = flow[prev_center_y, prev_center_x, 1]

                        # If previous detection had flow, average with current flow
                        if has_previous_flow:
                            # Weight previous flow less (0.3) than current flow (0.7)
                            flow_x = 0.7 * flow_x + 0.3 * prev_flow_x
                            flow_y = 0.7 * flow_y + 0.3 * prev_flow_y

                        # Check if flow magnitude is significant
                        flow_magnitude = (flow_x**2 + flow_y**2) ** 0.5
                        if flow_magnitude > 0.5:  # Only use if flow is significant
                            # Predict new center
                            new_center_x = int(prev_center_x + flow_x)
                            new_center_y = int(prev_center_y + flow_y)

                            # Calculate new bounding box coordinates
                            new_x1 = int(new_center_x - width / 2)
                            new_y1 = int(new_center_y - height / 2)
                            new_x2 = int(new_center_x + width / 2)
                            new_y2 = int(new_center_y + height / 2)

                            # Check if this predicted detection would overlap with any existing detection
                            predicted_box_overlaps = False

                            # Check against all current detections
                            for cur_det in current_detections:
                                cur_x1, cur_y1, cur_x2, cur_y2 = cur_det[0:4]

                                # Calculate intersection area
                                x_overlap = max(
                                    0, min(new_x2, cur_x2) - max(new_x1, cur_x1)
                                )
                                y_overlap = max(
                                    0, min(new_y2, cur_y2) - max(new_y1, cur_y1)
                                )
                                overlap_area = x_overlap * y_overlap

                                # Calculate predicted detection area
                                pred_area = (new_x2 - new_x1) * (new_y2 - new_y1)

                                # If overlap is significant (>30% of predicted area), don't add this prediction
                                if pred_area > 0 and overlap_area / pred_area > 0.3:
                                    predicted_box_overlaps = True
                                    if verbose:
                                        print(
                                            "  Skipping predicted detection - overlaps with existing detection"
                                        )
                                    break

                            # Only add if it doesn't significantly overlap with any existing detection
                            if not predicted_box_overlaps:
                                # Slightly reduce confidence for predicted detections
                                new_conf = max(
                                    0.1, conf * 0.8
                                )  # Reduce confidence but keep minimum of 0.1

                                # Add to predicted detections with flow data
                                predicted_det = (
                                    new_x1,
                                    new_y1,
                                    new_x2,
                                    new_y2,
                                    new_conf,
                                    cls_id,
                                    flow_x,
                                    flow_y,
                                )
                                predicted_detections.append(predicted_det)

                                if verbose:
                                    vehicle_type = (
                                        "car"
                                        if cls_id == 2
                                        else "truck"
                                        if cls_id == 7
                                        else f"class_{cls_id}"
                                    )
                                    print(
                                        f"Predicted disappeared {vehicle_type} (prev conf: {conf:.2f}, new: {new_conf:.2f})"
                                    )
                                    print(
                                        f"  Moved from ({prev_center_x}, {prev_center_y}) to ({new_center_x}, {new_center_y})"
                                    )
                                    print(
                                        f"  Flow: ({flow_x:.2f}, {flow_y:.2f}) pixels"
                                    )
            except Exception as e:
                if verbose:
                    print(f"Error predicting detection #{i}: {e}")

        if verbose:
            print(f"Added {len(predicted_detections)} predicted detections")
            print("--- End of Prediction Analysis ---\n")

        return predicted_detections

    def process_and_annotate_frame(
        self,
        source,
        model,
        quad,
        conf_threshold=0.25,
        use_flow=True,
        draw_quad=True,
        cursor_pos=None,
        app_state=None,
    ):
        """
        Process a frame and annotate it (convenience method combining the two steps)

        Args:
            source: Video source
            model: YOLO model
            quad: Points defining the field boundary
            conf_threshold: Confidence threshold
            use_flow: Whether to enhance detections with optical flow
            draw_quad: Whether to draw the quad boundary on the frame
            cursor_pos: Optional cursor position in image space for drawing crosshairs
            app_state: Optional application state containing previous detections

        Returns:
            Tuple of (annotated_frame, car_detections)
        """
        frame, detections, car_detections = self.process_frame(
            source, model, quad, conf_threshold
        )

        # Enhance detections with optical flow if requested
        if use_flow and len(car_detections) > 0:
            car_detections = self.combine_detections_with_flow(car_detections)

        # Check for disappeared detections if we have app_state with previous detections
        if (
            use_flow
            and hasattr(self, "flow")
            and self.flow is not None
            and app_state
            and hasattr(app_state, "previous_car_detections")
            and app_state.previous_car_detections
        ):
            # Predict positions of disappeared objects
            predicted_detections = self.predict_disappeared_detections(
                car_detections, app_state.previous_car_detections
            )

            # Append any predicted detections to our list
            if predicted_detections:
                car_detections.extend(predicted_detections)

        # Pass quad points and cursor position to annotate_frame
        quad_points = quad if draw_quad else None
        annotated_frame = self.annotate_frame(
            frame, model, detections, car_detections, quad_points, cursor_pos
        )

        return annotated_frame, car_detections


def main():
    camera_list = []
    for camera_info in sources.enumerate_avf_sources():  # enumerate_cameras():
        # Format: [(index, name), ...]
        print(f"{camera_info[0]}: {camera_info[1]}")
        camera_list.append(camera_info)

    # model=YOLO('yolov9s.pt')
    # model=YOLO('yolov8s.pt')
    # model=YOLO('yolo11n.pt')
    model = YOLO("yolov5nu.pt")

    # Initialize frame processor
    frame_processor = FrameProcessor()

    window = create_glfw_window()
    imgui.create_context()
    impl = GlfwRenderer(window)

    # Initialize application state
    app_state = State(camera_list)

    # Initialize the UI components with the state
    global control_panel, field_viz, camera_display
    field_viz = FieldVisualization(app_state)

    # Initialize video sources with error handling
    # Check if using video file or camera
    if app_state.use_video_file and app_state.video_file_path:
        try:
            source_1 = sources.VideoSource(app_state.video_file_path, state=app_state)
            print(f"Using video file: {app_state.video_file_path}")
        except Exception as e:
            print(f"Error initializing video file: {e}")
            # Create a placeholder source with error message
            source_1 = sources.PlaceholderSource(
                width=1920,
                height=1080,
                message=f"Video File Error: {str(e)}",
                state=app_state,
            )
    else:
        # Use camera source
        camera1_id = app_state.get_camera1_id()
        try:
            source_1 = sources.AVFSource(
                camera1_id if camera1_id is not None else 0, state=app_state
            )
            print(f"Using camera: {camera1_id}")
        except Exception as e:
            print(f"Error initializing camera: {e}")
            # Create a placeholder source with error message
            source_1 = sources.PlaceholderSource(
                width=1920,
                height=1080,
                message=f"Camera Error: {str(e)}",
                state=app_state,
            )

    camera_display = CameraDisplay(app_state, source_1)

    control_panel = ControlPanel(app_state, field_viz, camera_display)

    # Frame timing variables
    frame_time = 1.0 / 60.0  # Target 60 FPS
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
                # Use the new FrameProcessor class with quad drawing enabled
                processed_frame, car_detections = (
                    frame_processor.process_and_annotate_frame(
                        source_1,
                        model,
                        app_state.camera1_points,
                        use_flow=True,
                        draw_quad=True,  # Draw the quad directly on the frame
                        cursor_pos=app_state.c1_cursor_image_pos
                        if app_state.c1_cursor_in_image
                        else None,
                        app_state=app_state,  # Pass app_state for previous detections and optical flow settings
                    )
                )
                # Update detections only when processing is active
                app_state.set_car_detections(car_detections)
            else:
                # When paused, just use the raw frame without processing
                # But still draw the quad for reference
                frame_with_quad = frame.copy()
                processed_frame = frame_processor.annotate_frame(
                    frame_with_quad,
                    model,
                    None,
                    [],
                    quad_points=app_state.camera1_points,
                    cursor_pos=app_state.c1_cursor_image_pos
                    if app_state.c1_cursor_in_image
                    else None,
                )

            # Always update texture with the current frame (processed or raw)
            # Use the processed frame with annotations for display
            sources.update_opengl_texture(source_1.get_texture_id(), processed_frame)

            # Draw the camera view using the CameraDisplay class
            camera_display.draw()

            # Draw the control panel and update its values
            reinit_camera = control_panel.draw()

            # Check if we need to reinitialize camera source
            if reinit_camera:
                # Check whether to use video file or camera
                if app_state.use_video_file and app_state.video_file_path:
                    # Use video file as source
                    try:
                        new_source = sources.VideoSource(
                            app_state.video_file_path, state=app_state
                        )
                        # Update the camera display with the new source
                        source_1 = new_source
                        camera_display.source = new_source
                        print(f"Switched to video file: {app_state.video_file_path}")
                    except Exception as e:
                        error_message = f"Video File Error: {str(e)}"
                        print(f"Error switching to video file: {e}")
                        # Create a placeholder source with the error message
                        new_source = sources.PlaceholderSource(
                            width=1920,
                            height=1080,
                            message=error_message,
                            state=app_state,
                        )
                        # Update the camera display with the placeholder source
                        source_1 = new_source
                        camera_display.source = new_source
                else:
                    # Use camera as source
                    # Get the updated camera ID
                    camera1_id = app_state.get_camera1_id()
                    # Reinitialize camera source with the selected camera ID
                    try:
                        new_source = sources.AVFSource(
                            camera1_id if camera1_id is not None else 0, state=app_state
                        )
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
                            message=error_message,
                            state=app_state,
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
