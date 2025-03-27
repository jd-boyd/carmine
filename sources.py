import OpenGL.GL as gl
import cv2
import numpy as np
import bmcapture
import time

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

def create_opengl_texture(image):
    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
#    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, image.shape[1], image.shape[0], 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, image)
    return texture_id


def update_opengl_texture(texture_id, image):
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, image.shape[1], image.shape[0], 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, image)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)


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

    # Get model prediction on the resized frame
    results = model.predict(yolo_frame, conf=conf_threshold)[0]

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
            
            # Check if this is the highlighted car
            is_highlighted = False
            if highlighted_car is not None:
                hx1, hy1, hx2, hy2, _, _ = highlighted_car
                # Check if this is approximately the same detection
                overlap_threshold = 0.7  # Adjust if needed
                # Check that the centers are close to each other
                h_center_x = (hx1 + hx2) // 2
                h_center_y = (hy1 + hy2) // 2
                
                # Check if centers are within a small distance
                distance = np.sqrt((center_x - h_center_x)**2 + (center_y - h_center_y)**2)
                if distance < 30:  # Adjust threshold as needed
                    is_highlighted = True
            
            # Draw bounding box (yellow if highlighted, green otherwise)
            box_color = (0, 255, 255) if is_highlighted else (0, 255, 0)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), box_color, 2)

            # Display class name and confidence
            vehicle_type = class_names[cls_id]

            # Prepare label with vehicle type and confidence
            label = f"{vehicle_type}: {conf:.2f}"

            # Calculate label position
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y1_label = max(y1, label_size[1])

            # Draw label background
            bg_color = (0, 255, 255) if is_highlighted else (0, 255, 0)
            cv2.rectangle(output_frame, (x1, y1_label - label_size[1] - 5),
                         (x1 + label_size[0], y1_label), bg_color, -1)

            # Draw label text
            cv2.putText(output_frame, label, (x1, y1_label - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return output_frame, car_detections


class Source:
    """Base class for all video sources"""

    def get_frame(self):
        """
        Return the current raw frame without any processing

        Returns:
            numpy array: The current video frame
        """
        raise NotImplementedError

    def get_texture_id(self):
        """
        Return the OpenGL texture ID for the current frame

        Returns:
            int: OpenGL texture ID
        """
        raise NotImplementedError

    @property
    def width(self):
        """Width of the source in pixels"""
        raise NotImplementedError

    @property
    def height(self):
        """Height of the source in pixels"""
        raise NotImplementedError


class StillSource(Source):
    """Source that provides a single still image"""

    def __init__(self, filename):
        self.filename = filename
        self.frame = cv2.imread(self.filename)
        if self.frame is None:
            raise FileNotFoundError(f"Image not found at {filename}")

        self._width = self.frame.shape[1]
        self._height = self.frame.shape[0]
        self.texture_id = create_opengl_texture(self.frame)

    def get_frame(self):
        return self.frame

    def get_texture_id(self):
        return self.texture_id

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


class VideoSource(Source):
    """Source that provides frames from a video file"""

    def __init__(self, filename):
        self.frame_counter = 0
        self.video_path = filename
        self.cap = cv2.VideoCapture(self.video_path)

        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise FileNotFoundError(f"Could not read video from {filename}")

        self.frame = frame
        self._width = frame.shape[1]
        self._height = frame.shape[0]
        self.texture_id = create_opengl_texture(frame)
        self.last_frame_time = cv2.getTickCount() / cv2.getTickFrequency()

    def return_to_beginning(self):
        """Reset video to first frame"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get_frame(self):
        """
        Get the next frame from video, respecting frame rate limits

        Returns:
            numpy array: Current video frame
        """
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        # Only grab new frames at ~30fps regardless of UI refresh rate
        if (current_time - self.last_frame_time) > 0.033:
            ret, frame = self.cap.read()

            # If reached end of video, loop back to beginning
            if not ret:
                self.return_to_beginning()
                ret, frame = self.cap.read()
                if not ret:  # Still no frame after reset
                    return self.frame

            # Update current frame
            self.frame = frame
            update_opengl_texture(self.texture_id, self.frame)
            self.frame_counter += 1
            self.last_frame_time = current_time

        return self.frame

    def get_texture_id(self):
        """
        Get the OpenGL texture ID for current frame

        Returns:
            int: OpenGL texture ID
        """
        # Make sure we have the latest frame
        self.get_frame()
        return self.texture_id

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


class CameraSource(Source):
    """Source that provides frames from a camera device"""
    # TODO: Implement camera source
    pass


class BMSource(Source):
    """Source that provides frames from a Blackmagic capture device"""

    def __init__(self, device_index=0, width=1920, height=1080, framerate=24.0, low_latency=True):
        """
        Initialize a Blackmagic capture device

        Args:
            device_index: Index of the Blackmagic device (default: 0)
            width: Frame width (default: 1920)
            height: Frame height (default: 1080)
            framerate: Frame rate (default: 30.0)
            low_latency: Use low-latency mode (default: True)
        """
        self.device_index = device_index
        self._width = width
        self._height = height
        self.framerate = framerate
        self.low_latency = low_latency
        self.frame_count = 0

        # Try to initialize the capture device with the specified parameters
        try:
            self.cap = bmcapture.BMCapture(
                self.device_index,
                self._width,
                self._height,
                self.framerate,
                self.low_latency
            )
            print(f"Initialized Blackmagic capture: {width}x{height} @ {framerate} fps")
        except Exception as e:
            # If initialization fails, try common framerates
            success = False
            for framerate in [29.97, 24.0, 23.98, 25.0, 59.94, 60.0]:
                try:
                    print(f"Trying to initialize with {width}x{height} @ {framerate} fps...")
                    self.cap = bmcapture.BMCapture(
                        self.device_index,
                        self._width,
                        self._height,
                        framerate,
                        self.low_latency
                    )
                    self.framerate = framerate
                    print(f"Success with framerate {framerate}!")
                    success = True
                    break
                except Exception as e:
                    print(f"Failed with framerate {framerate}: {e}")

            # If high resolution fails, try 720p
            if not success:
                try:
                    self._width = 1280
                    self._height = 720
                    print(f"Trying to initialize with 1280x720 @ 59.94 fps...")
                    self.cap = bmcapture.BMCapture(self.device_index, 1280, 720, 59.94, self.low_latency)
                    self.framerate = 59.94
                    print("Success with 1280x720 @ 59.94 fps!")
                    success = True
                except Exception as e:
                    print(f"Failed with 720p: {e}")

            if not success:
                raise RuntimeError("Could not initialize any supported Blackmagic device mode")

        # Create an initial frame
        while not self.cap.update():
            time.sleep(0.1)
#            raise RuntimeError("Failed to get initial frame from Blackmagic device")

        # Get initial frame and create texture
        self.frame = self.cap.get_frame(format='rgb')  # Get frame in BGR format for OpenGL
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        self.texture_id = create_opengl_texture(self.frame)
        self.last_frame_time = time.time()

    def get_frame(self):
        """
        Get the next frame from the Blackmagic device

        Returns:
            numpy array: Current video frame
        """
        current_time = time.time()

        # Try to update frame at appropriate intervals based on framerate
        target_interval = 1.0 / self.framerate

        if (current_time - self.last_frame_time) >= target_interval:
            if self.cap.update():
                self.frame = self.cap.get_frame(format='rgb')  # Get frame in BGR format for OpenGL

                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"frame_dump/frame_{self.frame_count:04d}.jpg", self.frame)
                self.frame_count += 1
                if self.frame_count % 10 == 0:
                    print(f"Captured {self.frame_count} frames")


                update_opengl_texture(self.texture_id, self.frame)
                self.last_frame_time = current_time

        return self.frame

    def get_texture_id(self):
        """
        Get the OpenGL texture ID for current frame

        Returns:
            int: OpenGL texture ID
        """
        self.get_frame()  # Make sure we have the latest frame
        return self.texture_id

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def close(self):
        """
        Close the capture device and release resources
        """
        if hasattr(self, 'cap'):
            self.cap.close()
