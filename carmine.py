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
import config_db
from field_visualization import FieldVisualization
from control_panel import ControlPanel
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

        ret = self.state.camera1_quad.point_to_field(point_x, point_y)
        if ret is None:
            return 0,0
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
        self.state.set_c1_cursor([self.mouse_x-self.window_pos_x, self.mouse_y-self.window_pos_y])

        # Set default window size for OpenCV Image window
        default_width = 640
        self.scale = 3.0
        # Calculate the correct height based on the video's aspect ratio
        aspect_ratio = self.source.width / self.source.height
        default_height = default_width / aspect_ratio
        imgui.set_next_window_size(default_width, default_height, imgui.FIRST_USE_EVER)

        # Begin the camera window - title fixed from "Camnera 1"
        imgui.begin("Camera 1")
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

            # Get the window draw list to ensure drawing is relative to the current window
            draw_list = imgui.get_window_draw_list()

            # Draw POIs (points) on the camera view
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
                            # cursor_pos_x/y from get_cursor_screen_pos() already includes window position
                            # so this will move correctly when the window moves
                            screen_x = cursor_pos_x + (cam_x * scale_x)
                            screen_y = cursor_pos_y + (cam_y * scale_y)

                            # Determine point color based on nearest car distance
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
                            fill_color = imgui.get_color_u32_rgba(r, g, b, 0.5)  # Semi-transparent

                            # Draw triangle (pointing upward)
                            draw_list.add_triangle(
                                screen_x, screen_y - marker_size,               # top vertex
                                screen_x - marker_size, screen_y + marker_size,  # bottom left vertex
                                screen_x + marker_size, screen_y + marker_size,  # bottom right vertex
                                point_color, 2.0  # outline width
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
                                f"Point {i+1}"
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
                    # Get a fresh draw list to ensure proper window-relative coordinates
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
        
        # Flow analysis settings
        self.flow_verbose = True  # Set to True for detailed flow logging
        
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

        # Red cars, so red channel instead of normal gray scale conversion
        _, _, frame_gray = cv2.split(frame)
        
        # Initialize flow field 
        self.flow = None
        
        if self.old_gray is not None:
            # Calculate sparse optical flow for visualization
            lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            p0 = cv2.goodFeaturesToTrack(self.old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, p0, None, **lk_params)

            good_new = p1[st == 1]
            good_old = p0[st == 1]
            p0 = good_new.reshape(-1, 1, 2)

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
                of_frame = cv2.circle(of_frame, (int(a), int(b)), 5, (0, 0, 255), -1)
                of_frame = cv2.add(of_frame, mask)
                
            # Calculate dense optical flow for detection integration
            self.flow = cv2.calcOpticalFlowFarneback(
                self.old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
        else:
            # Initialize flow with zeros if this is the first frame
            self.flow = np.zeros((frame.shape[0], frame.shape[1], 2), dtype=np.float32)

        self.old_gray = frame_gray

        # Get model prediction
        results = model.predict(frame, imgsz=1920, conf=conf_threshold)[0]

        # Filter detections within the defined quadrilateral
        polygon = np.array(quad)
        polygon_zone = sv.PolygonZone(polygon=polygon)

        detections = sv.Detections.from_ultralytics(results)
        mask = polygon_zone.trigger(detections=detections)
        detections = detections[mask]

        # Apply tracking and smoothing
        #detections = self.tracker.update_with_detections(detections)
        #detections = self.smoother.update_with_detections(detections)

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

    def annotate_frame(self, frame, model, detections, car_detections, quad_points=None):
        """
        Annotate a processed frame with detection visualizations and optional quad overlay

        Args:
            frame: The frame to annotate
            model: YOLO model (for class names)
            detections: Supervision Detections object
            car_detections: List of car detections
            quad_points: Optional list of quad corner points to draw

        Returns:
            Annotated frame
        """
        if detections is None:
            return frame

        # Annotate with mask annotator
        annotated_image = self.mask_annotator.annotate(
            scene=frame.copy(), detections=detections)

        # Add bounding boxes for vehicle detections
        output_frame = annotated_image
        class_names = model.names
        
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
                        f"{i+1}", 
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 255, 255), 
                        2
                    )

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
            y1_label = max(y1, label_size[1])

            # Uncomment these if you want to add label text
            # Draw label background
            # bg_color = (0, 255, 255)
            # cv2.rectangle(output_frame, (x1, y1_label - label_size[1] - 5),
            #              (x1 + label_size[0], y1_label), bg_color, -1)
            #
            # # Draw label text
            # cv2.putText(output_frame, label, (x1, y1_label - 5),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

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
            flow = self.flow
            
        if flow is None or not car_detections:
            # Return original detections with zero flow if no flow data is available
            if verbose:
                print("No flow data available or no detections to process")
            return [(x1, y1, x2, y2, conf, cls, 0, 0) for x1, y1, x2, y2, conf, cls in car_detections]
            
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
            x2, y2 = min(w-1, x2), min(h-1, y2)
            
            # Skip if ROI is invalid
            if x1 >= x2 or y1 >= y2:
                if verbose:
                    print(f"Detection #{i+1}: Invalid ROI dimensions, skipping flow analysis")
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
                    vehicle_type = "car" if cls == 2 else "truck" if cls == 7 else f"class_{cls}"
                    
                    print(f"Detection #{i+1} ({vehicle_type}, conf: {conf:.2f}): ")
                    if flow_magnitude > 0.5:
                        print(f"  Flow: ({mean_flow_x:.2f}, {mean_flow_y:.2f}) pixels")
                        print(f"  Magnitude: {flow_magnitude:.2f} pixels, Direction: {flow_direction:.1f}°")
                        print(f"  Predicted center: ({center_x}, {center_y}) → ({new_center_x:.1f}, {new_center_y:.1f})")
                    else:
                        print(f"  Minimal flow detected ({flow_magnitude:.2f} pixels)")
                
                # Add to enhanced detections with flow values
                enhanced_detections.append((x1, y1, x2, y2, conf, cls, mean_flow_x, mean_flow_y))
            except Exception as e:
                if verbose:
                    print(f"Detection #{i+1}: Error processing flow: {e}")
                enhanced_detections.append((x1, y1, x2, y2, conf, cls, 0, 0))
        
        if verbose:
            print("--- End of Flow Analysis ---\n")
            
        return enhanced_detections

    def process_and_annotate_frame(self, source, model, quad, conf_threshold=0.25, use_flow=True, draw_quad=True):
        """
        Process a frame and annotate it (convenience method combining the two steps)

        Args:
            source: Video source
            model: YOLO model
            quad: Points defining the field boundary
            conf_threshold: Confidence threshold
            use_flow: Whether to enhance detections with optical flow
            draw_quad: Whether to draw the quad boundary on the frame

        Returns:
            Tuple of (annotated_frame, car_detections)
        """
        frame, detections, car_detections = self.process_frame(source, model, quad, conf_threshold)
        
        # Enhance detections with optical flow if requested
        if use_flow and len(car_detections) > 0:
            car_detections = self.combine_detections_with_flow(car_detections)
        
        # Pass quad points to annotate_frame if requested
        quad_points = quad if draw_quad else None
        annotated_frame = self.annotate_frame(frame, model, detections, car_detections, quad_points)
        
        return annotated_frame, car_detections


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
            source_1 = sources.VideoSource(app_state.video_file_path)
            print(f"Using video file: {app_state.video_file_path}")
        except Exception as e:
            print(f"Error initializing video file: {e}")
            # Create a placeholder source with error message
            source_1 = sources.PlaceholderSource(
                width=1920,
                height=1080,
                message=f"Video File Error: {str(e)}"
            )
    else:
        # Use camera source
        camera1_id = app_state.get_camera1_id()
        try:
            source_1 = sources.AVFSource(camera1_id if camera1_id is not None else 0)
            print(f"Using camera: {camera1_id}")
        except Exception as e:
            print(f"Error initializing camera: {e}")
            # Create a placeholder source with error message
            source_1 = sources.PlaceholderSource(
                width=1920,
                height=1080,
                message=f"Camera Error: {str(e)}"
            )

    # Secondary video source (no longer used actively but kept for reference)
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
                # Use the new FrameProcessor class with quad drawing enabled
                processed_frame, car_detections = frame_processor.process_and_annotate_frame(
                    source_1,
                    model,
                    app_state.camera1_points,
                    use_flow=True,
                    draw_quad=True  # Draw the quad directly on the frame
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
                    quad_points=app_state.camera1_points
                )

            # Always update texture with the current frame (processed or raw)
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
                        new_source = sources.VideoSource(app_state.video_file_path)
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
                            message=error_message
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
