import numpy as np

class DistanceEstimationAgent:
    def __init__(self, focal_length=1000, known_widths=None, known_heights=None):
        """
        Initialize the Distance Estimation Agent.
        
        Args:
            focal_length (float): Approximate focal length in pixels. 
                                  Can be calibrated or estimated.
            known_widths (dict): Dictionary mapping class labels to average real-world widths (in meters).
            known_heights (dict): Dictionary mapping class labels to average real-world heights (in meters).
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

        if known_heights is None:
            self.known_heights = {
                'car': 1.5,
                'bus': 3.2,
                'truck': 3.5,
                'motorcycle': 1.2
            }
        else:
            self.known_heights = known_heights

    def estimate(self, detections, image_width, focal_length=None):
        """
        Estimate distance for each detection using both width and height.

        Args:
            detections (list): List of detection dictionaries.
            image_width (int): Width of the image in pixels.
            focal_length (float): Optional focal length to override default.

        Returns:
            list: Detections with an added 'distance' key (in meters).
        """
        fl = focal_length if focal_length is not None else self.focal_length
        
        for det in detections:
            label = det['label']
            x1, y1, x2, y2 = det['box']
            pixel_width = x2 - x1
            pixel_height = y2 - y1
            
            real_width = self.known_widths.get(label, 1.8) # Default to car width
            real_height = self.known_heights.get(label, 1.5) # Default to car height
            
            # Distance = (Focal_Length * Real_Dimension) / Pixel_Dimension
            
            dist_w = 0.0
            dist_h = 0.0
            valid_estimates = 0
            
            if pixel_width > 0:
                dist_w = (fl * real_width) / pixel_width
                valid_estimates += 1
                
            if pixel_height > 0:
                dist_h = (fl * real_height) / pixel_height
                valid_estimates += 1
            
            if valid_estimates > 0:
                # Average the valid estimates
                distance = (dist_w + dist_h) / valid_estimates
            else:
                distance = 0.0
            
            det['distance'] = round(distance, 2)
            # Store components for debugging/analysis
            det['dist_w'] = round(dist_w, 2)
            det['dist_h'] = round(dist_h, 2)
            
        return detections
