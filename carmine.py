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


class ControlPanel:
    def __init__(self, camera_list):
        self.camera_list = camera_list
        
        # Camera selection
        self.selected_camera1 = 0
        self.selected_camera2 = 0
        
        # Camera points
        self.camera1_points = [0, 0, 0, 0]  # 4 points for camera 1
        self.camera2_points = [0, 0, 0, 0]  # 4 points for camera 2

        self.field_size = [160, 300]
        
        # Points of interest information
        self.poi_info = ["" for _ in range(10)]  # Info for 10 POIs
    
    def draw(self):
        """Draw the control panel UI and update its values"""
        if imgui.begin("Control Panel", True):
            imgui.text("Cameras")
            changed1, self.selected_camera1 = imgui.combo(
                "Camera 1", self.selected_camera1, [c[1] for c in self.camera_list]
            )
            changed2, self.selected_camera2 = imgui.combo(
                "Camera 2", self.selected_camera2, [c[1] for c in self.camera_list]
            )
            imgui.separator()

            imgui.text("Field Size")
            changed, self.field_size[0] = imgui.input_int(
                "Width", self.field_size[0]
            )

            changed, self.field_size[1] = imgui.input_int(
                "Height", self.field_size[1]
            )

            
            imgui.separator()
            
            imgui.text("Fields")
            imgui.text("Camera1 Numerical Points")
            for i in range(4):
                changed, self.camera1_points[i] = imgui.input_int(
                    f"Camera1 Point {i+1}", self.camera1_points[i]
                )
                
            imgui.text("Camera2 Numerical Points")
            for i in range(4):
                changed, self.camera2_points[i] = imgui.input_int(
                    f"Camera2 Point {i+1}", self.camera2_points[i]
                )
            imgui.separator()

            imgui.text("PoI")
            for i in range(10):
                imgui.text(f"Point {i+1}: {self.poi_info[i] or 'Information'}")
            
            imgui.end()
    
    def get_camera1_id(self):
        """Get the selected camera1 ID"""
        if self.selected_camera1 < len(self.camera_list):
            return self.camera_list[self.selected_camera1][0]
        return None
    
    def get_camera2_id(self):
        """Get the selected camera2 ID"""
        if self.selected_camera2 < len(self.camera_list):
            return self.camera_list[self.selected_camera2][0]
        return None
    
    def set_poi_info(self, index, info):
        """Set information for a specific POI"""
        if 0 <= index < len(self.poi_info):
            self.poi_info[index] = info


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


model=YOLO('yolov8s.pt')


def main():
    window = create_glfw_window()
    imgui.create_context()
    impl = GlfwRenderer(window)
    
    # Initialize the control panel
    global control_panel
    control_panel = ControlPanel(camera_list)

    source_1 = sources.VideoSource('./AI_angles.MOV', model)

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
        
        # Set default window size for OpenCV Image window
        default_width = 640
        # Calculate the correct height based on the video's aspect ratio
        aspect_ratio = source_1.width / source_1.height
        default_height = default_width / aspect_ratio
        imgui.set_next_window_size(default_width, default_height, imgui.FIRST_USE_EVER)
        
        imgui.begin("OpenCV Image")
        if tex_id:
            # Get window position (needed for mouse position calculation)
            window_pos_x, window_pos_y = imgui.get_window_position()
            # Get cursor position (for content region position)
            cursor_pos_x, cursor_pos_y = imgui.get_cursor_screen_pos()
            
            # Get available width and height of the ImGui window content area
            avail_width = imgui.get_content_region_available_width()
            
            # Calculate aspect ratio to maintain proportions
            aspect_ratio = source_1.width / source_1.height
            
            # Set display dimensions based on available width and aspect ratio
            display_width = avail_width
            display_height = display_width / aspect_ratio
            
            # Draw the image
            imgui.image(tex_id, display_width, display_height)
            
            # Draw crosshairs when hovering over the image
            if imgui.is_item_hovered():
                # Get mouse position
                mouse_x, mouse_y = imgui.get_io().mouse_pos
                
                # Only draw if mouse is inside the image area
                if (cursor_pos_x <= mouse_x <= cursor_pos_x + display_width and
                    cursor_pos_y <= mouse_y <= cursor_pos_y + display_height):
                    
                    # Draw vertical line
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
                frame_x = int(rel_x * source_1.width)
                frame_y = int(rel_y * source_1.height)
                
                # Print to console
                print(f"Click at video position: x={frame_x}, y={frame_y} (relative: {rel_x:.3f}, {rel_y:.3f})")
        imgui.end()


        # Draw the control panel and update its values
        control_panel.draw()


        gl.glClearColor(0.1, 0.1, 0.1, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()
