import cv2
import numpy as np
import os

def test_video_writer():
    output_path = "outputs"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    filename = os.path.join(output_path, "test_video.mp4")
    width, height = 640, 480
    fps = 20.0
    
    # Try the same codec as main.py
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: VideoWriter failed to open.")
        return False
        
    print(f"VideoWriter opened successfully. Writing to {filename}")
    
    # Generate some frames
    for i in range(50):
        # Create a blank image
        img = np.zeros((height, width, 3), np.uint8)
        # Add some moving text
        cv2.putText(img, f"Frame {i}", (50 + i*5, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(img)
        
    out.release()
    print("Video writing complete. Check the file.")
    
    # Check file size
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"File created. Size: {size} bytes")
        if size < 1000:
            print("Warning: File size seems too small.")
            return False
        return True
    else:
        print("Error: File was not created.")
        return False

if __name__ == "__main__":
    test_video_writer()
