import cv2
import numpy as np
import os

def test_deferred_initialization():
    output_path = "outputs"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    filename = os.path.join(output_path, "test_deferred.mp4")
    
    # Clean up previous run
    if os.path.exists(filename):
        os.remove(filename)
        
    width, height = 640, 480 # Expected dimensions
    fps = 20.0
    
    # Logic mimicking main.py fix
    out = None
    
    frame_count = 0
    print("Starting deferred initialization test...")
    
    # Simulate a stream of frames
    for i in range(50):
        # Create a frame (simulating carla_interface.get_frame())
        # Let's say the frame comes in at 640x480
        frame = np.zeros((height, width, 3), np.uint8)
        cv2.putText(frame, f"Deferred Frame {i}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if frame is None:
            continue
            
        # Initialize VideoWriter if not already done
        if out is None:
            h, w = frame.shape[:2]
            print(f"Initializing VideoWriter with frame size: {w}x{h}")
            out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
            if not out.isOpened():
                print("Failed to open VideoWriter.")
                return
        
        out.write(frame)
        frame_count += 1
        
    if out:
        out.release()
        
    # Check result
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"Test Successful. File created: {filename}, Size: {size} bytes")
    else:
        print("Test Failed. File not found.")

if __name__ == "__main__":
    test_deferred_initialization()
