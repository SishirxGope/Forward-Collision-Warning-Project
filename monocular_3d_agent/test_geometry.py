import sys
import os
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monocular_3d_agent.agent import Monocular3DBoxAgent

def test_3d_box_computation():
    print("Testing Monocular3DBoxAgent...")
    
    # 1. Initialize
    agent = Monocular3DBoxAgent(focal_length=1000)
    
    # 2. Mock Data
    file_path = os.path.abspath(__file__)
    print(f"Running from: {file_path}")
    
    detection = {
        'label': 'car',
        'box': [280, 300, 360, 360], # Center of image ~320x240. This is slightly lower right.
        'distance': 10.0 # 10 meters away
    }
    image_shape = (480, 640, 3) # H, W, C
    
    # 3. Compute
    center_3d = agent.compute_3d_center(detection, image_shape)
    if center_3d is None:
        print("[FAIL] Center 3D computation returned None.")
        sys.exit(1)
        
    print(f"[PASS] Computed 3D Center: {center_3d}")
    # Verify X is negative (since box is left of center 640/2=320)
    # Box center x = (280+360)/2 = 320. Wait, actually 320 is exactly center.
    # Let's check calculation. u=320. cx=320. X should be 0.
    if abs(center_3d[0]) < 0.1:
         print("[PASS] X coordinate matches expectation (approx 0).")
    else:
         print(f"[FAIL] X coordinate {center_3d[0]} != 0")
         
    box_3d = agent.compute_3d_box(detection, image_shape)
    
    if box_3d is None:
        print("[FAIL] Box 3D computation returned None.")
        sys.exit(1)
        
    print(f"[PASS] Computed 3D Box: {len(box_3d)} points")
    for i, pt in enumerate(box_3d):
        print(f"  Pt {i}: {pt}")
        
    # 4. Draw
    dummy_img = np.zeros(image_shape, dtype=np.uint8)
    drawn_img = agent.draw_3d_box(dummy_img, box_3d)
    
    # Check if any pixel was drawn
    if np.sum(drawn_img) > 0:
        print("[PASS] Drawing logic modified the image.")
    else:
        print("[FAIL] Image remains empty after drawing.")
        sys.exit(1)
        
    print("Test Complete. Success!")

if __name__ == "__main__":
    test_3d_box_computation()
