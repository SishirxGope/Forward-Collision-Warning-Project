import numpy as np
from scipy.interpolate import splprep, splev
import math

import carla

class BSplinePlanner:
    def __init__(self, step_size=0.5):
        """
        Initialize the B-Spline Planner.
        
        Args:
            step_size (float): Distance between sampled waypoints in meters.
        """
        self.step_size = step_size
        self.failure_reason = ""

    def generate_avoidance_path(self, start_pose, obstacle_info, carla_map, side='left', lookahead=20.0, offset_width=3.5):
        """
        Generates a smooth B-Spline path to avoid an obstacle.
        
        Args:
            start_pose (dict): {'x': float, 'y': float, 'yaw': float(radians)}
            obstacle_info (dict): {'dist': float, 'width': float, ...} for the detected obstacle.
                                  Assumes obstacle is directly ahead for now.
            side (str): 'left' or 'right' avoidance.
            lookahead (float): Total length of the maneuver (meters).
            offset_width (float): Lateral offset to shift (meters).
            
        Returns:
            list: List of waypoints [(x, y, v_target), ...] or None if generation failed.
        """
        # Unpack start pose
        sx, sy, syaw = start_pose['x'], start_pose['y'], start_pose['yaw']
        
        # Coordinate Transformation: Global to Local (Ego-centric)
        # It's easier to plan in local frame (x-forward, y-left) then transform back.
        
        # Control Points in Local Frame
        # 1. Start Point (0,0) with tangent along X-axis
        p0 = (0.0, 0.0)
        
        # 2. Apex Point (Beside Obstacle)
        obs_dist = obstacle_info.get('dist', 15.0)
        apex_dist = obs_dist + 5.0 
        
        if offset_width > 4.0: # Limit to realistic lane change width
            offset_width = 4.0

        # Adaptive Offset Search
        # Try finding a valid path by reducing offset if map constraints are violated
        # Start with requested offset, reduce down to minimum useful offset (2.0m)
        offsets_to_try = np.arange(offset_width, 1.9, -0.5)
        if len(offsets_to_try) == 0: offsets_to_try = [2.0]
        
        best_path = None
        best_valid = False
        
        c_p0 = (0.0, 0.0) # Start Point
        
        for trial_offset in offsets_to_try:
            # Generate Control Points for this offset
            lat_offset = trial_offset if side == 'left' else -trial_offset
            p1 = (apex_dist, lat_offset)
            
            # Recalculate re-entry for robustness
            min_reentry_len = 15.0
            reentry_x = max(obs_dist * 2.0, obs_dist + min_reentry_len)
            
            # Lift P1 to help curve reach apex
            c_p1 = (obs_dist * 0.4, lat_offset * 0.6) 
            c_p2 = (obs_dist, lat_offset) 
            c_p3 = (reentry_x, 0.0) 
            
            x_pts = [c_p0[0], c_p1[0], c_p2[0], c_p3[0]]
            y_pts = [c_p0[1], c_p1[1], c_p2[1], c_p3[1]]
            
            try:
                tck, u = splprep([x_pts, y_pts], k=3, s=0.0)
                total_len = c_p3[0] 
                num_samples = int(total_len / self.step_size)
                u_new = np.linspace(0, 1, num_samples)
                x_new, y_new = splev(u_new, tck)
                local_path = list(zip(x_new, y_new))
                
                # Check Clearance/Curvature (Local)
                if not self.validate_path(local_path, obstacle_info):
                     print(f"Offset {trial_offset} failed local check: {self.failure_reason}")
                     continue
                
                # Global Transform
                global_path = self.transform_to_global(local_path, sx, sy, syaw)
                
                # Check Map (Global)
                if carla_map:
                    if not self.validate_on_road(global_path, carla_map):
                        print(f"Offset {trial_offset} failed map check: {self.failure_reason}")
                        continue
                
                # Use this path!
                return global_path, True
                
            except Exception as e:
                print(f"[PLANNER] Spline error on offset {trial_offset}: {e}")
                continue
                
        # If we exit loop, no valid path found
        self.failure_reason = "No valid path found (Adaptive Search Failed)"
        return None, False

    def validate_on_road(self, global_path, carla_map):
        """
        Checks if the path stays within drivable road.
        """
        off_road_count = 0
        total_points = len(global_path)
        
        for i, (gx, gy) in enumerate(global_path):
            # Sample every few points to save time?
            if i % 2 != 0: continue 
            
            loc = carla.Location(x=gx, y=gy, z=0.5)
            waypoint = carla_map.get_waypoint(loc, project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder))
            
            # Check if projected point is close enough to actual point (meaning we are on road)
            # If we are far from the projection, we are off road.
            # But get_waypoint projects to centerline usually.
            
            # Better check: Is the point inside the lane info?
            # waypoint.transform.location is center of lane.
            # lane_width = waypoint.lane_width
            
            wp_loc = waypoint.transform.location
            dist_to_center = math.hypot(gx - wp_loc.x, gy - wp_loc.y)
            
            # Allow staying within lane + small margin?
            # User requirement: "Never go off-road".
            
            # Strict Check: Must be within lane width / 2
            # But waypoint could be 'Shoulder' if we are swerving hard?
            # Let's check lane_type.
            
            # Strict Check: "Never leave the road" (Sidewalks are forbidden)
            # We allow Driving and Shoulder (often needed for avoidance)
            # But strictly reject Sidewalk/off-road.
            allowed_types = [carla.LaneType.Driving, carla.LaneType.Shoulder, carla.LaneType.Bidirectional]
            
            if waypoint.lane_type not in allowed_types:
                 self.failure_reason = f"Invalid Lane Type: {waypoint.lane_type} at step {i}"
                 return False
                 
            # Lane Width Check?
            # If we are in Driving lane, are we comfortably inside?
            # waypoint loc is center.
            # If we are > lane_width/2 + margin, we are "outside" this lane. 
            # BUT we could be in the NEXT lane. get_waypoint snaps to closest lane center.
            # So if we are in next lane, get_waypoint returns next lane. 
            # So dist_to_center should always be small IF we are in a lane.
            
            # If we are OFF boundaries, get_waypoint might snap to nearest lane but distance is large?
            # Actually get_waypoint has project_to_road=True.
            # We can check lane_id change? No.
            
            # If dist to center > lane_width / 2, we are technically outside THAT lane.
            # But if we are in between lanes?
            # Let's use a relaxed check inside the Driving Lane.
            # We trust get_waypoint to return a Driving lane if we are on road.
            # If we are on sidewalk, get_waypoint(project_to_road=True) returns nearest road point.
            # So we check distance between Query Point and Waypoint Point.
            
            dist_to_wp = math.hypot(gx - wp_loc.x, gy - wp_loc.y)
            if dist_to_wp > waypoint.lane_width / 2.0:
                 # We are further from the center than half-width.
                 # This implies we are NOT in the lane returned by get_waypoint.
                 # Since get_waypoint snaps to NEAREST lane, this means we are off-road.
                 self.failure_reason = f"Off-Road Violation. Dist to Lane: {dist_to_wp:.2f} > {waypoint.lane_width/2.0:.2f}"
                 return False
            
        return True

    def validate_path(self, local_path, obstacle_info):
        """
        Validates the generated path for safety.
        1. Curvature checks.
        2. Obstacle clearance.
        """
        self.failure_reason = ""
        # 1. Curvature Check
        # Approximate curvature by angle changes between segments
        max_curvature = 0.4 # Relaxed slightly for close range maneuvers
        
        for i in range(1, len(local_path)-1):
            p0 = local_path[i-1]
            p1 = local_path[i]
            p2 = local_path[i+1]
            
            # Vector 1
            v1 = (p1[0]-p0[0], p1[1]-p0[1])
            # Vector 2
            v2 = (p2[0]-p1[0], p2[1]-p1[1])
            
            # Angles
            ang1 = math.atan2(v1[1], v1[0])
            ang2 = math.atan2(v2[1], v2[0])
            
            diff = abs(ang2 - ang1)
            dist = math.hypot(v1[0], v1[1])
            if dist < 0.01: continue
            
            curvature = diff / dist
            if curvature > max_curvature:
                # print(f"[PLANNER] Curvature Too High: {curvature:.3f} at step {i}")
                self.failure_reason = f"Curvature Too High: {curvature:.2f}"
                self.last_curvature = curvature 
                return False
            
            if curvature > getattr(self, 'last_curvature', 0.0):
                self.last_curvature = curvature

        # 2. Obstacle Clearance Check
        obs_dist = obstacle_info.get('dist', 10.0)
        obs_x = obs_dist
        obs_y = 0.0
        safe_radius = 1.9 # Reduced to 1.9m (10cm margin + car width)
        
        for px, py in local_path:
             # Check distance to obstacle center
             # Note: This is simplified. Ideally check against box.
             d = math.hypot(px - obs_x, py - obs_y)
             if d < safe_radius:
                 # But we might start close? 
                 # Only check if we are parallel to it?
                 # If we are effectively "behind" it, d could be large.
                 # If we are "passing" it (px approx equal obs_x)
                 if obs_x - 5.0 < px < obs_x + 5.0:
                     if d < safe_radius:
                         self.failure_reason = f"Obstacle Clearance Violation: {d:.2f}m < {safe_radius}m"
                         return False

        return True

    def transform_to_global(self, local_path, sx, sy, syaw):
        """Transforms list of (x,y) from local to global frame."""
        global_path = []
        cos_yaw = math.cos(syaw)
        sin_yaw = math.sin(syaw)
        
        for lx, ly in local_path:
            # Rotate
            gx = lx * cos_yaw - ly * sin_yaw
            gy = lx * sin_yaw + ly * cos_yaw
            # Translate
            gx += sx
            gy += sy
            global_path.append((gx, gy))
            
        return global_path
