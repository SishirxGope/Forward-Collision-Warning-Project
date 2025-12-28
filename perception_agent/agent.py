import cv2
import numpy as np
from ultralytics import YOLO

class PerceptionAgent:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        """
        Initialize the Perception Agent with a YOLO model.
        
        Args:
            model_path (str): Path to the YOLO weight file. Defaults to 'yolov8n.pt'.
            conf_threshold (float): Confidence threshold for detections.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # COCO class names to strictly include
        self.target_classes = ['car', 'bus', 'truck', 'motorcycle']

    def detect(self, image):
        """
        Detect vehicles in the input image.

        Args:
            image (numpy.ndarray): Input image in BGR format.

        Returns:
            list: A list of dictionaries representing detected objects.
                  Each dict contains: 'label', 'box' (x1, y1, x2, y2), 'conf', 'id' (optional)
        """
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        result = results[0]
        
        detections = []
        
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            conf = float(box.conf[0])
            
            if label in self.target_classes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    'label': label,
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': conf,
                    'raw_box': box # Keep raw box for tracking if needed
                })
                
        return detections
