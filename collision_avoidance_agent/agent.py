from .planner import BSplinePlanner
from .controller import PurePursuitController
import math

class CollisionAvoidanceAgent:
    def __init__(self, thresholds=None):
        """
        Initialize the Collision Avoidance Agent with safety thresholds.
        
        Args:
            thresholds (dict, optional): TTC thresholds for actions.
        """
        if thresholds is None:
            self.thresholds = {
                'warning': 4.0, # Increased for avoidance planning time
                'critical': 1.5,
                'emergency': 0.8
            }
        else:
            self.thresholds = thresholds
        
        self.emergency_latch = False 
        self.state = "NORMAL" # States: NORMAL, AVOIDANCE, EMERGENCY_BRAKE, STOP
        
        self.planner = BSplinePlanner()
        self.controller = PurePursuitController()
        
        self.avoidance_path = None
        self.target_obstacle_id = None
        self.avoidance_side = 'left' # Default side
        self.avoidance_start_pose = None
        
        self.path_curvature = 0.0 # Log param
        self.debug_path = None
        self.path_valid = False

    def reset(self):
        """Resets the agent state."""
        self.emergency_latch = False
        self.state = "NORMAL"
        self.avoidance_path = None
        self.target_obstacle_id = None
        self.debug_path = None
        self.path_valid = False

    def get_control_action(self, detections, overall_status, ego_speed=0.0, ego_pose=None, carla_map=None):
        """
        Decide appropriate control action based on TTC and safety thresholds.
        
        Args:
            detections (list): List of detections.
            overall_status (str): Current decision status.
            ego_speed (float): Current vehicle speed in m/s.
            ego_pose (dict): {'x', 'y', 'yaw', 'start_x'} # Added start_x to track completion?
                             Actually ego_pose needs to be global x,y,yaw for path following.
            carla_map (carla.Map): Map object for validation.

        Returns:
            dict or str: Control action.
                         String for simple actions ("Maintain Speed").
                         Dict for complex control {'steer': ..., 'throttle': ..., 'brake': ...}
        """
        # 0. Check Terminal State
        if self.state == "STOP":
            return "Apply Full Braking"

        if self.state == "EMERGENCY_BRAKE":
             if ego_speed < 0.1:
                 self.state = "STOP"
             return "Apply Full Braking"

        if self.emergency_latch:
            self.state = "EMERGENCY_BRAKE"
            return "Apply Full Braking"

        # 1. Analyze Threats
        min_ttc = float('inf')
        min_dist = float('inf')
        threat_det = None
        
        for det in detections:
            # Use Radar/GT primarily
            ttc = det.get('gt_ttc', det.get('ttc', float('inf')))
            dist = det.get('gt_dist', det.get('distance', float('inf')))
            
            # Filter valid threats
            if dist > 60.0: continue 

            if ttc < min_ttc and ttc > 0:
                min_ttc = ttc
                threat_det = det
            
            if dist < min_dist:
                min_dist = dist

        # print(f"[DEBUG] CA: State={self.state} MinTTC={min_ttc:.2f} MinDist={min_dist:.2f}", flush=True)

        # 2. State Machine Transition Logic
        
        # TRANSITION: NORMAL -> AVOIDANCE
        if self.state == "NORMAL":
            # Check for Avoidance Trigger (TTC or Proximity)
            # Trigger if TTC is within warning range OR if we are getting close (distance based)
            # Low speed approach might have high TTC but we need to act if close.
            should_plan = False
            if (min_ttc < self.thresholds['warning'] and min_ttc > self.thresholds['critical']):
                should_plan = True
            elif min_dist < 15.0 and min_dist > 4.0: # Close range planning
                should_plan = True
                
            if should_plan:
                # Potential Avoidance Candidate
                if ego_pose and threat_det:
                    # --- OPTIMAL SIDE SELECTION ---
                    # Check both Left and Right for the best valid path
                    
                    obs_info = {'dist': threat_det.get('gt_dist', threat_det.get('distance')), 'width': 2.0}
                    candidates = []
                    
                    for side in ['left', 'right']:
                        # Generate path for this side with default max offset (planner adapts internally)
                        path, is_valid = self.planner.generate_avoidance_path(
                            ego_pose, obs_info, carla_map, side=side, offset_width=4.0
                        )
                        candidates.append({'side': side, 'path': path, 'is_valid': is_valid})

                    # Decision Logic
                    best_candidate = None
                    valid_candidates = [c for c in candidates if c['is_valid']]
                    
                    if valid_candidates:
                        # Heuristic: Prefer Left if valid, else pick the first valid one
                        # If both are valid, Left is standard passing side.
                        left_cand = next((c for c in valid_candidates if c['side'] == 'left'), None)
                        best_candidate = left_cand if left_cand else valid_candidates[0]
                    else:
                        # Fallback: Just take the first one (will be invalid/red) for debug
                         if candidates: best_candidate = candidates[0]

                    if best_candidate:
                        path = best_candidate['path']
                        is_valid = best_candidate['is_valid']
                        selected_side = best_candidate['side']
                        if is_valid: print(f"[CA] Optimal Side Selected: {selected_side.upper()}")
                    else:
                        path, is_valid = None, False
                        selected_side = 'left'

                    # Store for visualization
                    self.debug_path = path
                    self.path_valid = is_valid
                    
                    if path and is_valid:
                        print(f"[CA] Path Generated ({selected_side}). Switching to AVOIDANCE.")
                        self.state = "AVOIDANCE"
                        self.avoidance_path = path
                        self.avoidance_side = selected_side
                        self.target_obstacle_id = threat_det.get('id', 0) 
                        
                        # Store obstacle location for completion check
                        cox = math.cos(ego_pose['yaw'])
                        sox = math.sin(ego_pose['yaw'])
                        ox = ego_pose['x'] + obs_info['dist'] * cox
                        oy = ego_pose['y'] + obs_info['dist'] * sox
                        self.obstacle_global_pos = (ox, oy)
                    
                    elif path and not is_valid:
                         # Path rejected. Will be visualized as RED in main.py
                         pass
                        
                    else:
                        # Path planning failed
                        pass
            
            # TRANSITION: Any -> EMERGENCY
            if min_ttc < self.thresholds['critical']:
                print(f"[CA] Critical Threat! Emergency Braking. TTC={min_ttc}")
                self.state = "EMERGENCY_BRAKE"
                self.emergency_latch = True
                return "Apply Full Braking"
                
            if min_dist < 3.0: # LOWERED FROM 7.0 per user request
                 print(f"[CA] Proximity Violation! Emergency Braking. Dist={min_dist}")
                 self.state = "EMERGENCY_BRAKE"
                 self.emergency_latch = True
                 return "Apply Full Braking"

        # 3. State Execution
        
        if self.state == "NORMAL":
            if min_ttc < self.thresholds['warning']:
                return "Warning Only"
            else:
                return "Maintain Speed"

        elif self.state == "AVOIDANCE":
            # Safety Check 1: Dynamic Path Validation
            # Instead of raw TTC (which drops as we approach), check if the PATH is still valid.
            # And check if we are significantly deviating from it.
            
            # Check proximity for "Crash Imminent" fallback
            if min_ttc < 0.5: # Extremely critical
                  print("[CA] CRITICAL IMPACT IMMINENT. Emergency Braking.")
                  self.state = "EMERGENCY_BRAKE"
                  self.emergency_latch = True
                  return "Apply Full Braking"
                  
            # Re-validate path against simplified obstacle model
            # We assume the obstacle is static or moving linearly.
            # If we had a dynamic map, we'd update `obs_info` and call `planner.validate_path`
            # For now, let's trust the generated path unless we have reason to doubt.
            
            # TODO: Improve this by updating obstacle position
            pass

            # Completion Condition
            # If we passed the obstacle?
            # Dot product check? Or simple distance check?
            # "Ego X > Obstacle X + Margin"
            # We need to project ego pos onto the track?
            # Simple Euclidean check: dist to obstacle > safe AND we are past it?
            
            # Let's use the local frame distance.
            # If radar says object is BEHIND us? (dist is negative? Radar doesn't usually report back)
            # If radar loses object, and we have travelled the path length?
            
            # Check if we reached end of path
            if self.avoidance_path:
                last_pt = self.avoidance_path[-1]
                d_end = math.hypot(ego_pose['x'] - last_pt[0], ego_pose['y'] - last_pt[1])
                if d_end < 2.0:
                    print("[CA] Avoidance Maneuver Complete (End of Path).")
                    self.state = "NORMAL"
                    self.avoidance_path = None
                    return "Maintain Speed"
            
            # Check if we passed the obstacle (using stored global pos)
            # Vector from Obs to Ego
            if hasattr(self, 'obstacle_global_pos'):
                 ox, oy = self.obstacle_global_pos
                 # Dot with path direction? 
                 # Assume path is generally forward (X-axis in local initial frame)
                 # Dist check: if we are far enough from it?
                 pass

            # Execute Control
            control = self.controller.compute_control(ego_pose, ego_speed, self.avoidance_path)
            control['action_type'] = 'trajectory_follow' # Flag for interface
            control['path'] = self.avoidance_path # For visualization
            control['side'] = self.avoidance_side
            control['curvature'] = self.planner.last_curvature if hasattr(self.planner, 'last_curvature') else 0.0
            return control
            
        return "Maintain Speed"
