import OpenGL.GL as gl
import cv2

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


def process_frame(frame, model, conf_threshold):
    """
    Process a single frame with YOLOv8 to detect cars
    
    Args:
        frame: Input frame
        model: YOLOv8 model
        conf_threshold: Confidence threshold
    
    Returns:
        Processed frame with detections
    """
    # YOLOv8 class names (COCO dataset)
    class_names = model.names
    
    # Car class ID in COCO dataset (2: car, 5: bus, 7: truck)
    vehicle_classes = [2, 5, 7]
    
    # Get model prediction
    results = model.predict(frame, conf=conf_threshold)[0]
    
    # Create a copy of the frame
    output_frame = frame.copy()
    
    # Iterate through detections
    for det in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)
        
        # Check if the detected object is a vehicle
        if cls_id in vehicle_classes:
            # Draw bounding box
            cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Display class name and confidence
            vehicle_type = class_names[cls_id]
            label = f"{vehicle_type}: {conf:.2f}"
            
            # Calculate label position
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y1 = max(y1, label_size[1])
            
            # Draw label background
            cv2.rectangle(output_frame, (int(x1), int(y1) - label_size[1] - 5), 
                         (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(output_frame, label, (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return output_frame
    

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
    def __init__(self, filename, model):
        self.frame_counter = 0
        self.model = model
        self.video_path = filename
        self.cap = cv2.VideoCapture(self.video_path)
        ret, frame = self.cap.read()
        self.frame = frame
        if frame is None:
            raise FileNotFoundError("Image not found. Please make sure 'image.jpg' exists in the same directory or provide the correct path.")

        self.height = frame.shape[0]
        self.width = frame.shape[1]

        self.texture_id = create_opengl_texture(frame)

    def return_to_beginning(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        

    def get_next_frame(self):
        ret, frame = self.cap.read()
        
        # If we've reached the end of the video
        if not ret:
            # Reset to the beginning
            self.return_to_beginning()
            # Try reading again
            ret, frame = self.cap.read()
            # If still no frame, there's a problem with the video
            if not ret:
                return None

        processed_frame = process_frame(frame, self.model, 0.25)
        
        self.frame = processed_frame
        update_opengl_texture(self.texture_id, self.frame)
        self.frame_counter += 1
        
        return self.texture_id




class CameraSource(Source):
    pass
