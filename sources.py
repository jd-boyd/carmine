import OpenGL.GL as gl
import cv2
import numpy as np

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


def process_frame_with_yolo(frame, model, conf_threshold=0.25):
    """
    Process a single frame with YOLOv8 to detect cars

    Args:
        frame: Input frame
        model: YOLOv8 model
        conf_threshold: Confidence threshold

    Returns:
        Processed frame with detections
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
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display class name and confidence
            vehicle_type = class_names[cls_id]

            # Prepare label with vehicle type and confidence
            label = f"{vehicle_type}: {conf:.2f}"

            # Calculate label position
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y1 = max(y1, label_size[1])

            # Draw label background
            cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 5),
                         (x1 + label_size[0], y1), (0, 255, 0), -1)

            # Draw label text
            cv2.putText(output_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return output_frame


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