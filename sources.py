import OpenGL.GL as gl
import cv2

def create_opengl_texture(image):
    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, image.shape[1], image.shape[0], 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, image)
    return texture_id


def update_opengl_texture(texture_id, image):
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, image.shape[1], image.shape[0], 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, image)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)


class Source:
    def get_next_frame(self):
        raise NotImplemented


class StillSource(Source):
    def __init__(self, filename):
        self.filename = filename
        #image = cv2.imread("frame_1.jpg") # Replace with your image path

    def get_next_frame(self):
        pass


class VideoSource(Source):
    def __init__(self, filename):
        self.frame_counter = 0

        self.video_path = filename
        self.cap = cv2.VideoCapture(self.video_path)
        ret, self.frame = self.cap.read()

        if self.frame is None:
            raise FileNotFoundError("Image not found. Please make sure 'image.jpg' exists in the same directory or provide the correct path.")
        texture_id = create_opengl_texture(self.frame)


    def get_next_frame(self):
        pass



class CameraSource(Source):
    pass
