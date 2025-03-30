import OpenGL.GL as gl
import sys
import cv2
import numpy as np
import subprocess
import re

import time


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



class Source:
    """Base class for all video sources"""

    def __init__(self):
        raise NotImplementedError

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


class PlaceholderSource(Source):
    """Source that provides a placeholder image with text when no other source is available"""

    def __init__(self, width=640, height=480, message="No video source available"):
        self._width = width
        self._height = height
        self.message = message

        # Create a blank image with dark gray background
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.frame[:] = (50, 50, 50)  # Dark gray background

        # Add text message to the image
        self._add_text_to_frame()

        # Create OpenGL texture from the frame
        self.texture_id = create_opengl_texture(self.frame)

    def _add_text_to_frame(self):
        """Add text message to the center of the frame"""
        # Define font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_color = (255, 255, 255)  # White
        font_thickness = 2

        # Get text size to center it
        text_size = cv2.getTextSize(self.message, font, font_scale, font_thickness)[0]

        # Calculate text position to center it
        text_x = int((self._width - text_size[0]) / 2)
        text_y = int((self._height + text_size[1]) / 2)

        # Put text on the image
        cv2.putText(
            self.frame,
            self.message,
            (text_x, text_y),
            font,
            font_scale,
            font_color,
            font_thickness
        )

        # Add helper text below main message
        helper_message = "Select a camera or video file to begin"
        helper_font_scale = 0.7
        helper_text_size = cv2.getTextSize(helper_message, font, helper_font_scale, font_thickness)[0]
        helper_x = int((self._width - helper_text_size[0]) / 2)
        helper_y = text_y + 40

        cv2.putText(
            self.frame,
            helper_message,
            (helper_x, helper_y),
            font,
            helper_font_scale,
            font_color,
            font_thickness
        )


    def set_message(self, message):
        """Update the placeholder message and redraw the frame"""
        self.message = message
        # Reset frame to dark gray
        self.frame[:] = (50, 50, 50)
        # Add new text
        self._add_text_to_frame()
        # Update the texture
        update_opengl_texture(self.texture_id, self.frame)


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

            # Update current frame (but don't update texture here, let the main loop do it)
            self.frame = frame
            # Don't update texture here - let the main loop do it with the processed frame
            # update_opengl_texture(self.texture_id, self.frame)
            self.frame_counter += 1
            self.last_frame_time = current_time

        return self.frame


def enumerate_avf_sources():
    """
    Use ffmpeg to list available video capture devices on the system.

    Returns:
        list: List of tuples (device_index, device_name) for available video devices.
    """
    # Run ffmpeg command to list devices
    try:
        result = subprocess.run(
            ["ffmpeg", "-list_devices", "true", "-f", "avfoundation", "-i", "dummy"],
            capture_output=True,
            text=True,
            check=False  # Don't raise exception on non-zero exit code
        )

        # The command will have an error code since it doesn't actually open a device,
        # but it will still output the device list to stderr
        output = result.stderr

        # Find the video devices section
        video_section = False
        video_devices = []

        # Process each line of the output
        for line in output.splitlines():
            # Check if we've reached the video devices section
            if "AVFoundation video devices:" in line:
                video_section = True
                continue
            # Check if we've reached the audio devices section (end of video section)
            elif "AVFoundation audio devices:" in line:
                video_section = False
                break

            # If we're in the video section, parse the device information
            if video_section:
                # Match pattern like [0] Device Name
                match = re.match(r'.*\[(\d+)\]\s+(.*)', line)
                if match:
                    device_index = int(match.group(1))
                    device_name = match.group(2).strip()
                    video_devices.append((device_index, device_name))

        return video_devices

    except Exception as e:
        print(f"Error enumerating cameras: {e}")
        return []


class AVFSource(VideoSource):

    def __init__(self, idx):
        self.frame_counter = 0
        self.cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)

        self.last_frame_time = cv2.getTickCount() / cv2.getTickFrequency()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise FileNotFoundError(f"Could not read video source {idx}")

        self.frame = frame
        self._width = frame.shape[1]
        self._height = frame.shape[0]
        self.texture_id = create_opengl_texture(frame)


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

            # Update current frame
            self.frame = frame
            # Don't update texture here - let the main loop do it with the processed frame
            # update_opengl_texture(self.texture_id, self.frame)
            self.frame_counter += 1
            self.last_frame_time = current_time

        return self.frame
