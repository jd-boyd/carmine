import OpenGL.GL as gl
import sys
import cv2
import numpy as np
try:
    import bmcapture
except ModuleNotFoundError:
    print("BMCapture not available.")

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

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

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

            # If reached end of video, loop back to beginning
            # Update current frame
            self.frame = frame
            update_opengl_texture(self.texture_id, self.frame)
            self.frame_counter += 1
            self.last_frame_time = current_time

        return self.frame

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height
