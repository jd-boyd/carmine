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

for camera_info in enumerate_cameras():
    print(f'{camera_info.index}: {camera_info.name}')


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

# running = True
# while running:
#     print("R:")
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     # Capture frame from webcam
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     # Convert OpenCV BGR image to RGB
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # Flip the frame horizontally
#     frame = cv2.flip(frame, 1)

#     # Create Pygame surface from the frame
#     frame = np.rot90(frame)
#     frame = pygame.surfarray.make_surface(frame)

#     # Blit the frame onto the screen
#     screen.blit(frame, (0, 0))

#     # Update the display
#     pygame.display.update()

#     # Release resources
#     cap.release()
#     pygame.quit()

def main():
    imgui.create_context()
    impl = PygameRenderer()

    io = imgui.get_io()
    io.display_size = size

    show_custom_window = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            impl.process_event(event)
        impl.process_inputs()

        imgui.new_frame()

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", "Cmd+Q", False, True
                )

                if clicked_quit:
                    sys.exit(0)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.show_test_window()

        if show_custom_window:
            is_expand, show_custom_window = imgui.begin("Custom window", True)
            if is_expand:
                imgui.text("Bar")
                imgui.text_colored("Eggs", 0.2, 1.0, 0.0)
            imgui.end()

        # note: cannot use screen.fill((1, 1, 1)) because pygame's screen
        #       does not support fill() on OpenGL sufraces
        gl.glClearColor(1, 1, 1, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        imgui.render()
        impl.render(imgui.get_draw_data())

        ret, frame = cap.read()
        if not ret:
            print("No Frame.")
            continue

        # Process OpenCV frame to create an OpenGL texture
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        frame_rgba = cv2.flip(frame_rgba, 1)
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, frame_rgba.shape[1], frame_rgba.shape[0],
                        0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, frame_rgba)

        # Render the OpenGL texture in an ImGui window
        imgui.begin("Video")
        imgui.image(texture_id, frame_rgba.shape[1], frame_rgba.shape[0])
        imgui.end()

        gl.glDeleteTextures([texture_id])


        pygame.display.flip()




main()
