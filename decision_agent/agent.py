import numpy as np

class DecisionAgent:
    def __init__(self, fps=30, ttc_threshold=2.5):
        """
        Initialize the Decision Agent.

        Args:
            fps (int): Frames per second of the video stream.
            ttc_threshold (float): Time-to-collision threshold in seconds for unsafe classification.
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.ttc_threshold = ttc_threshold
        
        # Track objects: {id: {'history': [distances...], 'last_box': box}}
        # For simplicity in this project without a deepsort tracker, 
        # we will use simple centroid matching or assume processed sequentially.
        # Here we will just calculate speed based on matching current boxes to previous boxes using IoU.
        self.prev_detections = [] 

    def analyze(self, detections):
        """
        Analyze the scene to determine risk.

        Args:
            detections (list): List of detections with 'distance'.

        Returns:
            list: Detections updated with 'speed', 'ttc', 'risk_level'.
            str: Overall scene status ('Safe', 'Warning', 'Critical').
        """
        overall_status = "Safe"
        
        # Match current detections with previous to estimate speed
        # Simple matching: Find closest box in prev_detections
        
        for det in detections:
            # Default values
            det['speed'] = 0.0 # m/s (approaching speed)
            det['ttc'] = float('inf')
            det['risk'] = 'Safe'
            
            matched_prev = self._match_detection(det, self.prev_detections)
            
            if matched_prev:
                prev_dist = matched_prev['distance']
                curr_dist = det['distance']
                
                # Speed = (Prev - Curr) / Time
                # Positive speed means approaching
                speed = (prev_dist - curr_dist) / self.dt
                
                # Apply some smoothing or thresholding to ignore noise
                if speed < 0.1: speed = 0.0
                
                det['speed'] = round(speed, 2)
                
                if speed > 0 and curr_dist > 0:
                    ttc = curr_dist / speed
                    det['ttc'] = round(ttc, 2)
                    
                    if ttc < self.ttc_threshold:
                        det['risk'] = 'Unsafe'
                        overall_status = "Unsafe"
                else:
                     det['ttc'] = float('inf')

                # Proximity Override (Vision)
                if curr_dist < 7.0:
                    det['risk'] = 'Unsafe'
                    overall_status = "Unsafe"
            
            # Ground Truth Analysis
            if 'ground_truth' in det:
                gt = det['ground_truth']
                det['gt_dist'] = gt['distance']
                det['gt_speed'] = gt['speed']
                
                # Use GT speed directly if available, otherwise fallback to derivative speed
                approaching_speed = det.get('gt_speed', det.get('speed', 0.0))
                # Update the main speed field to reflect this (for visualization/alerting)
                det['speed'] = approaching_speed
                
                if approaching_speed > 0 and gt['distance'] > 0:
                     gt_ttc = round(gt['distance'] / approaching_speed, 2)
                     det['gt_ttc'] = gt_ttc
                     # Also update main TTC
                     det['ttc'] = gt_ttc
                     
                     # Override risk with Ground Truth if available
                     if gt_ttc < self.ttc_threshold:
                         det['risk'] = 'Unsafe'
                         overall_status = "Unsafe"
                     # Proximity Override (GT)
                     elif gt['distance'] < 7.0:
                         det['risk'] = 'Unsafe'
                         overall_status = "Unsafe"
                     # If GT says safe but Vision says Unsafe? 
                     # For safety critical, usually logical OR (if either says unsafe -> warning).
                     # But here we want to rely on CARLA GT as the "Source of Truth" for the simulation.
                     # Let's enforce GT decision.
                     elif det['risk'] == 'Unsafe': 
                         # Downgrade if GT says safe? Or keep it?
                         # "Use CARLA based ... to compute TTC ... Ensure that collision warnings are triggered consistently"
                         det['risk'] = 'Safe'
                else:
                     det['gt_ttc'] = float('inf')

            # Radar Analysis (Takes Precedence if Available)
            if det.get('radar_available'):
                # Radar Data Logic
                r_dist = det.get('radar_dist', float('inf'))
                r_ttc = det.get('radar_ttc', float('inf'))
                
                # Trust Radar TTC
                det['ttc'] = r_ttc 
                det['distance'] = r_dist # Ensure visualization uses radar distance
                
                if r_ttc < self.ttc_threshold:
                    det['risk'] = 'Unsafe'
                    overall_status = "Unsafe"
                elif r_dist < 7.0: 
                    # Physical proximity safety net
                    det['risk'] = 'Unsafe' 
                    overall_status = "Unsafe"
                else:
                    det['risk'] = 'Safe'
                    # If previously 'Unsafe' due to vision? Overwrite.
                    # Radar is usually better than monocular vision.
            
        self.prev_detections = detections
        return detections, overall_status

    def _match_detection(self, det, prev_detections):
        """
        Find the best matching detection from the previous frame.
        Using IoU (Intersection over Union).
        """
        if not prev_detections:
            return None
        
        best_match = None
        max_iou = 0.0
        
        boxA = det['box']
        
        for prev in prev_detections:
            boxB = prev['box']
            iou = self._calculate_iou(boxA, boxB)
            if iou > 0.5 and iou > max_iou: # IoU threshold match
                max_iou = iou
                best_match = prev
                
        return best_match

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou
