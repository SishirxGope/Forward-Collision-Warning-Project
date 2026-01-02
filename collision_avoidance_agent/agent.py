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

    def reset(self):
        """Resets the agent state (e.g., clears emergency latch)."""
        self.emergency_latch = False

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
        # 0. Check Latch
        if self.emergency_latch:
            return "Apply Full Braking"

        min_ttc = float('inf')
        
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
        
        # 2. Apply Deterministic Logic
        action = "Maintain Speed"
        
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
            
        return action
