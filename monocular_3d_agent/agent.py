import numpy as np
import cv2

class Monocular3DBoxAgent:
    def __init__(self, focal_length=1000, known_dims=None):
        """
        Initialize the Monocular 3D Box Agent.
        
        Args:
            focal_length (float): Approximate focal length in pixels.
            known_dims (dict): Dictionary mapping class labels to average real-world dimensions (meters).
                               Format: 'label': (width, height, length)
        """
        self.focal_length = focal_length
        # Dims: (width, height, length) in meters
        # Standard average vehicle dimensions
        if known_dims is None:
            self.known_dims = {
                'car': (1.8, 1.5, 4.5),      
                'bus': (2.5, 3.2, 12.0),     
                'truck': (2.6, 3.5, 10.0),   
                'motorcycle': (0.8, 1.2, 2.0) 
            }
        else:
            self.known_dims = known_dims

    def compute_3d_center(self, detection, image_shape, focal_length=None):
        """
        Compute the 3D center (X, Y, Z) of the object.
        
        Args:
            detection (dict): Detection dictionary containing 'box' and 'distance'.
            image_shape (tuple): (height, width) of the image.
            focal_length (float): Optional focal length to override default.
            
        Returns:
            tuple: (X, Y, Z) coordinates in meters. Returns None if invalid.
        """
        h_img, w_img = image_shape[:2]
        cx, cy = w_img / 2.0, h_img / 2.0
        
        fl = focal_length if focal_length is not None else self.focal_length
        
        # 1. 2D Box Center
        x1, y1, x2, y2 = detection['box']
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0 
        
        # 2. Retrieve Distance (Z)
        Z = detection.get('distance', float('inf'))
        if Z == float('inf') or Z <= 0:
            return None

        # 3. Reconstruct 3D Center (X, Y, Z) via Pinhole Model
        # coordinate system: X right, Y down, Z forward
        X = (u - cx) * Z / fl
        Y = (v - cy) * Z / fl
        
        return (X, Y, Z)

    def compute_3d_box(self, detection, image_shape, focal_length=None):
        """
        Compute the 8 corners of the 3D bounding box projected onto the image.
        
        Args:
            detection (dict): Detection dictionary containing 'label', 'box', and 'distance'.
            image_shape (tuple): (height, width) of the image.
            focal_length (float): Optional focal length to override default.
            
        Returns:
            list: List of 8 (x, y) tuples representing the 3D box corners on the 2D image.
                  Returns None if box cannot be computed.
        """
        label = detection['label']
        # Skip if we don't have dimensions for this class
        if label not in self.known_dims:
            return None 
            
        fl = focal_length if focal_length is not None else self.focal_length

        center_3d = self.compute_3d_center(detection, image_shape, focal_length=fl)
        if center_3d is None:
            return None
            
        X, Y, Z = center_3d
        
        h_img, w_img = image_shape[:2]
        cx, cy = w_img / 2.0, h_img / 2.0
        
        # 4. Define 3D Box Corners relative to Center
        W, H, L = self.known_dims[label]
        
        # We assume the object is axis-aligned with the camera (simple bounding box)
        dx = W / 2.0
        dy = H / 2.0
        dz = L / 2.0
        
        # 8 corners: (X, Y, Z) +/- (dx, dy, dz)
        # Order convention: Front face (Z-dz), then Rear face (Z+dz)
        # Standard counter-clockwise or defined order for drawing
        # Let's map coordinates:
        # 0: top-left-front
        # ...
        
        # Let's just iterate combinations signs
        # X signs: -1, 1
        # Y signs: -1, 1
        # Z signs: -1, 1
        
        # To make drawing easier, let's define two faces:
        # Front Face (closer to camera? No, "Front" of car vs "Front" facing camera. 
        # Let's just do "Near Z" and "Far Z" relative to object center)
        
        corners_3d = np.array([
            # Front Face (Z - dz) -> Closer to camera if object is facing away? 
            # Actually, if Z is positive forward, Z-dz is closer to camera.
            [X - dx, Y - dy, Z - dz], # Top Left
            [X + dx, Y - dy, Z - dz], # Top Right
            [X + dx, Y + dy, Z - dz], # Bottom Right
            [X - dx, Y + dy, Z - dz], # Bottom Left
            
            # Back Face (Z + dz)
            [X - dx, Y - dy, Z + dz], # Top Left
            [X + dx, Y - dy, Z + dz], # Top Right
            [X + dx, Y + dy, Z + dz], # Bottom Right
            [X - dx, Y + dy, Z + dz], # Bottom Left
        ])
        
        # 5. Project 3D Corners back to 2D
        corners_2d = []
        for p in corners_3d:
            px, py, pz = p
            
            # Safety check for points behind camera (shouldn't happen given Z > 0 and standard dims)
            if pz <= 0.1: 
                continue 
            
            u_p = (px * fl) / pz + cx
            v_p = (py * fl) / pz + cy
            corners_2d.append((int(u_p), int(v_p)))
            
        if len(corners_2d) < 8:
            return None
            
        return corners_2d

    def draw_3d_box(self, image, corners_2d, color=(0, 255, 255), thickness=2):
        """
        Draws the projected 3D box on the image.
        
        Args:
            image (numpy.ndarray): The image to draw on.
            corners_2d (list): List of 8 (x, y) tuples.
            color (tuple): BGR color.
            thickness (int): Line thickness.
            
        Returns:
            numpy.ndarray: Annotated image.
        """
        if corners_2d is None or len(corners_2d) != 8:
            return image
        
        pts = np.array(corners_2d, dtype=np.int32)
        
        # Front Face (first 4 points)
        cv2.polylines(image, [pts[:4]], True, color, thickness)
        
        # Back Face (last 4 points)
        cv2.polylines(image, [pts[4:]], True, color, thickness)
        
        # Connecting Lines (Edges)
        for i in range(4):
            pt1 = tuple(pts[i])
            pt2 = tuple(pts[i+4])
            cv2.line(image, pt1, pt2, color, thickness)
            
        return image
