import math
import numpy as np

class PurePursuitController:
    def __init__(self, wheelbase=2.8):
        """
        Initialize Pure Pursuit Controller.
        
        Args:
            wheelbase (float): Vehicle wheelbase in meters (approx 2.8m for Tesla Model 3).
        """
        self.wheelbase = wheelbase
        self.lookahead_dist = 4.0 # Base lookahead distance (m)
        self.k_speed = 0.5 # Lookahead gain based on speed

    def compute_control(self, ego_pose, ego_speed, path, max_speed=20.0):
        """
        Computes steering and throttle for path following.
        
        Args:
            ego_pose (dict): {'x', 'y', 'yaw'}
            ego_speed (float): Current speed in m/s.
            path (list): List of (x,y) tuples [GLOBAL FRAME].
            max_speed (float): Speed limit during avoidance (km/h).
            
        Returns:
            dict: {'steer': float (-1 to 1), 'throttle': float, 'brake': float}
        """
        if not path or len(path) < 2:
            return {'steer': 0.0, 'throttle': 0.0, 'brake': 1.0} # Emergency stop if invalid path
            
        # 1. Target Point Selection
        # Dynamic lookahead
        # current speed in m/s
        L = self.lookahead_dist + self.k_speed * ego_speed
        
        target_point = None
        
        # Search for first point on path that is further than L
        # Simple search: assumes path is ordered ahead of us
        # We should find the closest point first, then look ahead?
        # For simplicity in this localized planner: Search from beginning
        
        min_d = float('inf')
        closest_idx = 0
        
        # Find closest point to track progress
        for i, (px, py) in enumerate(path):
            d = math.hypot(px - ego_pose['x'], py - ego_pose['y'])
            if d < min_d:
                min_d = d
                closest_idx = i
                
        # Look ahead from closest point
        found_target = False
        for i in range(closest_idx, len(path)):
            px, py = path[i]
            d = math.hypot(px - ego_pose['x'], py - ego_pose['y'])
            if d > L:
                target_point = (px, py)
                found_target = True
                break
                
        if not found_target:
             # End of path? Aim at last point
             target_point = path[-1]
             
        # 2. Pure Pursuit Steering Calculation
        tx, ty = target_point
        
        # Transform target to vehicle frame
        dx = tx - ego_pose['x']
        dy = ty - ego_pose['y']
        
        # Rotation magnitude to align with ego yaw
        # Rx = dx * cos(-yaw) - dy * sin(-yaw)
        # Ry = dx * sin(-yaw) + dy * cos(-yaw)
        yaw = ego_pose['yaw']
        local_x = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        local_y = dx * math.sin(-yaw) + dy * math.cos(-yaw)
        
        # Curvature = 2 * y / L^2
        # Lookahead dist Ld is actually distance to target point
        Ld = math.hypot(local_x, local_y)
        
        if Ld < 0.5: # Too close, avoid singularity
             # Try to steer towards heading of path? 
             steer = 0.0 
        else:
             curvature = 2.0 * local_y / (Ld**2)
             # Steer angle = atan(curvature * wheelbase)
             steer_rad = math.atan(curvature * self.wheelbase)
             
             # Normalized steer for CARLA (-1.0 to 1.0)
             # Max steer usually ~70 deg? No, carla is normalized. 
             # Physical max steer is usually ~30-40 deg (0.5-0.7 rad).
             max_steer_rad = 0.7
             steer = np.clip(steer_rad / max_steer_rad, -1.0, 1.0)
        
        # 3. Speed Control
        # Convert max_speed (km/h) to m/s
        target_v = max_speed / 3.6 
        
        throttle = 0.0
        brake = 0.0
        
        if ego_speed < target_v:
            throttle = 0.5 # Accelerate gently
            brake = 0.0
        else:
            throttle = 0.0
            brake = 0.0 # Coast
            if ego_speed > target_v + 1.0: # Hysteresis
                brake = 0.2 # Gentle brake
                
        # Hard Brake on Sharp turns?
        # If steering is extreme, slow down
        if abs(steer) > 0.5:
             throttle *= 0.5 # Reduce throttle
        
        return {
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake)
        }
