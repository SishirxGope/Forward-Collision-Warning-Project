import cv2
import numpy as np

class XAIModule:
    def __init__(self):
        pass
        
    def explain(self, frame, detections, alert_message):
        """
        Generate visual and text explanations.

        Args:
            frame (numpy.ndarray): Original image.
            detections (list): List of detections.
            alert_message (str): The generated alert.

        Returns:
            numpy.ndarray: Annotated image with bounding boxes, risk info, and saliency overlay.
            str: Detailed text explanation.
        """
        explained_frame = frame.copy()
        text_log = []
        
        # Generate Saliency Map (Simulated as Heatmap on risky objects)
        # In a full research project, this would use GradCAM.
        # Here we verify the claim by highlighting the region of interest effectively.
        saliency_layer = np.zeros_like(explained_frame, dtype=np.uint8)
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            dist = det['distance']
            risk = det.get('risk', 'Safe')
            ttc = det.get('ttc', float('inf'))
            speed = det.get('speed', 0)

            # Color coding
            if risk == 'Unsafe':
                color = (0, 0, 255) # Red
                # Add check to saliency
                cv2.rectangle(saliency_layer, (x1, y1), (x2, y2), (0, 0, 255), -1)
                explanation = f"CRITICAL: {label} is {dist}m away, approaching at {speed}m/s. TTC {ttc}s < Threshold."
            else:
                color = (0, 255, 0) # Green
                explanation = f"Safe: {label} at {dist}m."
            
            text_log.append(explanation)

            # Draw Box
            cv2.rectangle(explained_frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label_text = f"{label} {dist}m"
            if risk == 'Unsafe':
                label_text += f" TTC:{ttc}s"
                
            cv2.putText(explained_frame, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Blend saliency
        if alert_message:
            alpha = 0.3
            cv2.addWeighted(saliency_layer, alpha, explained_frame, 1 - alpha, 0, explained_frame)
            
            # Add Alert Banner
            cv2.rectangle(explained_frame, (0, 0), (explained_frame.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(explained_frame, alert_message, (50, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        return explained_frame, "\n".join(text_log)
