import numpy as np

class DistanceEstimationAgent:
    def __init__(self, focal_length=1000, known_widths=None):
        """
        Initialize the Distance Estimation Agent.
        
        Args:
            focal_length (float): Approximate focal length in pixels. 
                                  Can be calibrated or estimated.
            known_widths (dict): Dictionary mapping class labels to average real-world widths (in meters).
        """
        self.focal_length = focal_length
        if known_widths is None:
            self.known_widths = {
                'car': 1.8,
                'bus': 2.5,
                'truck': 2.5,
                'motorcycle': 0.8
            }
        else:
            self.known_widths = known_widths

    def estimate(self, detections, image_width):
        """
        Estimate distance for each detection.

        Args:
            detections (list): List of detection dictionaries.
            image_width (int): Width of the image in pixels.

        Returns:
            list: Detections with an added 'distance' key (in meters).
        """
        for det in detections:
            label = det['label']
            x1, y1, x2, y2 = det['box']
            pixel_width = x2 - x1
            
            real_width = self.known_widths.get(label, 1.8) # Default to car width
            
            # Distance = (Focal_Length * Real_Width) / Pixel_Width
            # Note: This implies the object is facing us. 
            if pixel_width > 0:
                distance = (self.focal_length * real_width) / pixel_width
            else:
                distance = 0.0
            
            det['distance'] = round(distance, 2)
            
        return detections
