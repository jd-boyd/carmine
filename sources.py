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

class AVFSource(VideoSource):

    def __init__(self, idx):
        self.frame_counter = 0
        self.cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)

        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise FileNotFoundError(f"Could not read video from {filename}")

        self.frame = frame
        self._width = frame.shape[1]
        self._height = frame.shape[0]
        self.texture_id = create_opengl_texture(frame)
        self.last_frame_time = cv2.getTickCount() / cv2.getTickFrequency()




# class BMSource(Source):
#     """Source that provides frames from a Blackmagic capture device"""

#     def __init__(self, device_index=0, width=1920, height=1080, framerate=24.0, low_latency=True):
#         """
#         Initialize a Blackmagic capture device

#         Args:
#             device_index: Index of the Blackmagic device (default: 0)
#             width: Frame width (default: 1920)
#             height: Frame height (default: 1080)
#             framerate: Frame rate (default: 30.0)
#             low_latency: Use low-latency mode (default: True)
#         """
#         self.device_index = device_index
#         self._width = width
#         self._height = height
#         self.framerate = framerate
#         self.low_latency = low_latency
#         self.frame_count = 0

#         # Try to initialize the capture device with the specified parameters
#         try:
#             self.cap = bmcapture.BMCapture(
#                 self.device_index,
#                 self._width,
#                 self._height,
#                 self.framerate,
#                 self.low_latency
#             )
#             print(f"Initialized Blackmagic capture: {width}x{height} @ {framerate} fps")
#         except Exception as e:
#             # If initialization fails, try common framerates
#             success = False
#             for framerate in [24, 29.97, 24.0, 23.98, 25.0, 59.94, 60.0]:
#                 try:
#                     print(f"Trying to initialize with {width}x{height} @ {framerate} fps...")
#                     self.cap = bmcapture.BMCapture(
#                         self.device_index,
#                         self._width,
#                         self._height,
#                         framerate,
#                         self.low_latency
#                     )
#                     self.framerate = framerate
#                     print(f"Success with framerate {framerate}!")
#                     success = True
#                     break
#                 except Exception as e:
#                     print(f"Failed with framerate {framerate}: {e}")

#             # If high resolution fails, try 720p
#             # if not success:
#             #     try:
#             #         self._width = 1280
#             #         self._height = 720
#             #         print(f"Trying to initialize with 1280x720 @ 59.94 fps...")
#             #         self.cap = bmcapture.BMCapture(self.device_index, 1280, 720, 59.94, self.low_latency)
#             #         self.framerate = 59.94
#             #         print("Success with 1280x720 @ 59.94 fps!")
#             #         success = True
#             #     except Exception as e:
#             #         print(f"Failed with 720p: {e}")

#             if not success:
#                 raise RuntimeError("Could not initialize any supported Blackmagic device mode")

#         # Create an initial frame
#         while not self.cap.update():
#             time.sleep(0.1)
# #            raise RuntimeError("Failed to get initial frame from Blackmagic device")

#         # Get initial frame and create texture
#         self.frame = self.cap.get_frame(format='rgb')  # Get frame in BGR format for OpenGL
#         self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
#         self.texture_id = create_opengl_texture(self.frame)
#         self.last_frame_time = time.time()

#     def get_frame(self):
#         """
#         Get the next frame from the Blackmagic device

#         Returns:
#             numpy array: Current video frame
#         """
#         current_time = time.time()

#         # Try to update frame at appropriate intervals based on framerate
#         target_interval = 1.0 / self.framerate

#         if (current_time - self.last_frame_time) >= target_interval:
#             if self.cap.update():
#                 self.frame = self.cap.get_frame(format='rgb')  # Get frame in BGR format for OpenGL

#                 self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
#                 cv2.imwrite(f"frame_dump/frame_{self.frame_count:04d}.jpg", self.frame)
#                 self.frame_count += 1
#                 if self.frame_count % 10 == 0:
#                     print(f"Captured {self.frame_count} frames")


#                 update_opengl_texture(self.texture_id, self.frame)
#                 self.last_frame_time = current_time

#         return self.frame

#     def get_texture_id(self):
#         """
#         Get the OpenGL texture ID for current frame

#         Returns:
#             int: OpenGL texture ID
#         """
#         self.get_frame()  # Make sure we have the latest frame
#         return self.texture_id

#     @property
#     def width(self):
#         return self._width

#     @property
#     def height(self):
#         return self._height

#     def close(self):
#         """
#         Close the capture device and release resources
#         """
#         if hasattr(self, 'cap'):
#             self.cap.close()
