class AlertAgent:
    def __init__(self):
        pass

    def generate_alert(self, overall_status, detections):
        """
        Generate a human-readable alert message.

        Args:
            overall_status (str): 'Safe' or 'Unsafe'.
            detections (list): List of processed detections.

        Returns:
            str: Alert message.
        """
        if overall_status == "Safe":
            return None
        
        # Find the most critical object
        critical_dets = [d for d in detections if d['risk'] == 'Unsafe']
        
        if not critical_dets:
            return None
            
        # Sort by TTC
        critical_dets.sort(key=lambda x: x['ttc'])
        target = critical_dets[0]
        
        message = f"WARNING: Collision Risk! {target['label'].upper()} at {target['distance']}m. TTC: {target['ttc']}s."
        return message
