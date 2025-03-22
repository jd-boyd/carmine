import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw
import cv2
import OpenGL.GL as gl
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras
from ultralytics import YOLO

import sources
from sources import create_opengl_texture, update_opengl_texture


camera_list = []
for camera_info in enumerate_cameras():
    desc = (camera_info.index, f'{camera_info.index}: {camera_info.name}')
    print(desc[1])
    camera_list.append(desc)


def create_glfw_window(window_name="Carmine", width=1280, height=720):
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


#model=YOLO('yolov8s.pt')


def main():
    window = create_glfw_window()
    imgui.create_context()
    impl = GlfwRenderer(window)

    source_1 = sources.VideoSource('./AI_angles.MOV')

    running = True
    while running:
        if glfw.window_should_close(window):
            running = False

        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", "Ctrl+Q", False, True
                )

                if clicked_quit:
                    running = False

                imgui.end_menu()
            imgui.end_main_menu_bar()

        tex_id = source_1.get_next_frame()
        
        imgui.begin("OpenCV Image")
        if tex_id:
            imgui.image(tex_id, source_1.width, source_1.height)
        imgui.end()


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


        gl.glClearColor(0.1, 0.1, 0.1, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()
