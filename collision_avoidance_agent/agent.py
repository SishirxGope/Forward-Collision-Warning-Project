class CollisionAvoidanceAgent:
    def __init__(self, thresholds=None):
        """
        Initialize the Collision Avoidance Agent with safety thresholds.
        
        Args:
            thresholds (dict, optional): TTC thresholds for actions.
                                         Defaults to:
                                         - warning: 2.5s (Issue warning only)
                                         - critical: 1.5s (Apply braking)
                                         - emergency: 0.8s (Apply full braking)
        """
        if thresholds is None:
            self.thresholds = {
                'warning': 2.5,
                'critical': 1.5,
                'emergency': 0.8
            }
        else:
            self.thresholds = thresholds
        
        self.emergency_latch = False # Latch for emergency braking
        self.state = "NORMAL" # States: NORMAL, STOP

    def reset(self):
        """Resets the agent state (e.g., clears emergency latch)."""
        self.emergency_latch = False
        self.state = "NORMAL"

    def get_control_action(self, detections, overall_status, ego_speed=0.0):
        """
        Decide appropriate control action based on TTC and safety thresholds.
        
        Logic:
        1. Check Latch: If emergency braking was triggered previously, maintain it.
        2. Find minimum valid TTC from potential threats.
        3. Apply deterministic thresholds:
           - TTC > Warning: Maintain Speed
           - Warning > TTC > Critical: Warning Only (No Brake)
           - Critical > TTC > Emergency: Apply Braking
           - TTC < Emergency: Apply Full Braking
           
        Args:
            detections (list): List of detections.
            overall_status (str): Current decision status.
            ego_speed (float, optional): Current vehicle speed in m/s. Defaults to 0.0.

        Returns:
            str: Action description for control system.
        """
        # 0. Check Terminal State & Latch
        if getattr(self, 'state', 'NORMAL') == "STOP":
            return "Apply Full Braking"

        if self.emergency_latch:
             # Check for transition to Terminal STOP
            if ego_speed < 0.1: # Stopped after emergency braking
                self.state = "STOP"
            return "Apply Full Braking"

        min_ttc = float('inf')
        min_dist = float('inf')
        
        # 1. Identify the most critical TTC
        for det in detections:
            # Avoid duplicating logic: Only consider objects flagged as Unsafe by DecisionAgent
            if det.get('risk') != 'Unsafe':
                continue

            # Prioritize Ground Truth if available
            ttc = det.get('gt_ttc', det.get('ttc', float('inf')))
            
            if ttc != float('inf') and ttc > 0:
                 if ttc < min_ttc:
                     min_ttc = ttc
            
            # Track minimum distance
            dist = det.get('gt_dist', det.get('distance', float('inf')))
            if dist < min_dist:
                min_dist = dist
                
            # print(f"[DEBUG] CA checking: Risk={det.get('risk')} TTC={ttc} MinTTC={min_ttc}", flush=True)
        
        # 2. Apply Deterministic Logic
        action = "Maintain Speed"
        
        # print(f"[DEBUG] CA Decision: MinTTC={min_ttc} Warning={self.thresholds['warning']} Critical={self.thresholds['critical']}", flush=True)
        
        print(f"[DEBUG] CA Decision: MinTTC={min_ttc} Warning={self.thresholds['warning']} Critical={self.thresholds['critical']}", flush=True)
        
        if min_ttc == float('inf'):
            action = "Maintain Speed"
        elif min_ttc > self.thresholds['warning']:
            action = "Maintain Speed"
        elif min_ttc > self.thresholds['critical']:
            action = "Warning Only"
        elif min_ttc > self.thresholds['emergency']:
            action = "Apply Braking"
        else:
            action = "Apply Full Braking"
            self.emergency_latch = True # Trigger Latch
            
        # 3. Collision / Near-Collision Override
        # If distance is extremely small (< 2.0m), assume collision or imminent impact
        if min_dist < 2.0:
            action = "Apply Full Braking"
            self.emergency_latch = True
            
        # 4. Proximity Safety Curtain (Anti-Oscillation)
        # Force braking if within safety distance, even if TTC is technically "safe" (e.g. slow approach)
        # or just "Warning" level.
        if min_dist < 7.0: # User requested 7m override
             action = "Apply Full Braking" # Be aggressive at 7m
             self.emergency_latch = True # Latch it to ensure we stop
        elif min_dist < 12.0 and action in ["Maintain Speed", "Warning Only"]:
             action = "Apply Braking" # Force braking if close
            
        # Check if we should transition to STOP state (any braking stop)
        if action in ["Apply Braking", "Apply Full Braking"] and ego_speed < 0.1:
            self.state = "STOP"
            action = "Apply Full Braking" # Enforce full stop
            
        return action
