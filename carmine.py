from imgui.integrations.pygame import PygameRenderer
import OpenGL.GL as gl
import imgui
import pygame
import sys
import cv2
import numpy as np
from ultralytics import YOLO
#from tracker import *
from cv2_enumerate_cameras import enumerate_cameras

camera_list = []
for camera_info in enumerate_cameras():
    desc = (camera_info.index, f'{camera_info.index}: {camera_info.name}')
    print(desc[1])
    camera_list.append(desc)





model=YOLO('yolov8s.pt')

# Initialize Pygame
pygame.init()

vid_scale_area = (640, 360)

# Set window dimensions
width, height = size = 960, 540
screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)

pygame.display.set_caption("Carmine")

# Initialize OpenCV video capture
cap = cv2.VideoCapture(1404)

def load_texture_from_file(filename):
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Could not load image: {filename}")
        return None, (0, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, image)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture_id, (width, height)

def main():
    imgui.create_context()
    impl = PygameRenderer()

    io = imgui.get_io()
    io.display_size = size
    texture1, size1 = load_texture_from_file("frame_1.jpg")
    texture2, size2 = load_texture_from_file("frame_2.jpg")

    show_custom_window = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            impl.process_event(event)
        impl.process_inputs()
        current_window_size = pygame.display.get_surface().get_size()
        io.display_size = current_window_size

        imgui.new_frame()
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(io.display_size[0], io.display_size[1])
        flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_BACKGROUND
        if imgui.begin("Main Working Area", flags):
            imgui.end()

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", "Ctrl+Q", False, True
                )

                if clicked_quit:
                    sys.exit(0)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        if imgui.begin("Control Panel", True):
            imgui.text("Cameras")
            _, selected_camera1 = imgui.combo("", 0, [c[1] for c in camera_list])
            _, selected_camera2 = imgui.combo("", 1, [c[1] for c in camera_list])
            imgui.separator()

            imgui.text("Fields")
            imgui.text("Camera1 Numerical Points")
            for i in range(1, 5):
                changed, value = imgui.input_int(f"Camera1 Point {i}", 0)
            imgui.text("Camera2 Numerical Points")
            for i in range(1, 5):
                changed, value = imgui.input_int(f"Camera2 Point {i}", 0)
            imgui.separator()

            imgui.text("PoI")
            for i in range(1, 11):
                imgui.text(f"Point {i}: Information")
            imgui.end()

        if imgui.begin("Camera 1", True):
            if texture1 is not None:
                avail_w, avail_h = imgui.get_content_region_available()
                imgui.image(texture1, avail_w, avail_h)
            else:
                imgui.text("No image loaded")
            imgui.end()

        if imgui.begin("Camera 2", True):
            if texture2 is not None:
                avail_w, avail_h = imgui.get_content_region_available()
                imgui.image(texture2, avail_w, avail_h)
            else:
                imgui.text("No image loaded")
            imgui.end()

        # note: cannot use screen.fill((1, 1, 1)) because pygame's screen
        #       does not support fill() on OpenGL sufraces
        gl.glClearColor(1, 1, 1, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        imgui.render()
        impl.render(imgui.get_draw_data())

        # ret, frame = cap.read()
        # if not ret:
        #     print("No Frame.")
        #     continue

        # # Process frame using pygame to render in a custom window region
        # frame = cv2.flip(frame, 1)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = np.rot90(frame)
        # video_surface = pygame.surfarray.make_surface(frame)

        # # Render the video frame in a custom Pygame window region
        # screen.blit(video_surface, (10, 10))


        pygame.display.flip()




main()
