import cv2
import os
import glob
from perception_agent.agent import PerceptionAgent
from distance_agent.agent import DistanceEstimationAgent
from decision_agent.agent import DecisionAgent
from alert_agent.agent import AlertAgent
from explainable_ai.xai import XAIModule
from carla_interface.interface import CarlaInterface
import argparse
import sys
import datetime

def main():
    """
    Main entry point for the Forward Collision Warning System.
    Parses arguments to select between Dataset Mode and CARLA Mode.
    """
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Forward Collision Warning System")
    parser.add_argument('--source', type=str, default='dataset', 
                        choices=['dataset', 'carla', 'video'],
                        help="Mode of operation: 'dataset' (default), 'carla' (simulation), or 'video' (file).")
    parser.add_argument('--data_path', type=str, default='data', 
                        help="Path to data folder (for dataset/video mode).")
    parser.add_argument('--output_path', type=str, default='outputs', 
                        help="Path to save outputs.")
    args = parser.parse_args()

    # 2. Setup
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    print(f"Initializing Agents...")
    # Initialize agents common to all modes
    perception = PerceptionAgent() 
    distance_estimator = DistanceEstimationAgent()
    decision_maker = DecisionAgent(fps=10) # Default FPS, updated in specific modes
    alerter = AlertAgent()
    explainer = XAIModule()
    
    # 3. Mode Dispatch
    if args.source == 'carla':
        print("\n--- Starting CARLA Simulation Mode ---")
        run_carla_mode(perception, distance_estimator, decision_maker, alerter, explainer, args.output_path)
        
    elif args.source == 'dataset':
        print("\n--- Starting Dataset Evaluation Mode ---")
        run_dataset_mode(args.data_path, perception, distance_estimator, decision_maker, alerter, explainer, args.output_path)
        
    elif args.source == 'video':
        print("\n--- Starting Video Processing Mode ---")
        # Check for video file
        video_files = glob.glob(os.path.join(args.data_path, '*.mp4'))
        if video_files:
             run_video_mode(video_files[0], perception, distance_estimator, decision_maker, alerter, explainer, args.output_path)
        else:
            print(f"No video files found in {args.data_path}")

def run_dataset_mode(data_path, perception, distance_estimator, decision_maker, alerter, explainer, output_path):
    """
    Runs the FCW pipeline on a dataset of images (e.g., BDD100K).
    """
    print(f"Searching for images in {data_path}...")
    image_files = sorted(glob.glob(os.path.join(data_path, '**', '*.jpg'), recursive=True) + 
                         glob.glob(os.path.join(data_path, '**', '*.png'), recursive=True))
    
    if not image_files:
        print("No images found. Please check your data path.")
        return

    print(f"Found {len(image_files)} images. Processing subset for demonstration.")
    
    limit = 20
    image_files = image_files[:limit]
    
    for i, img_path in enumerate(image_files):
        print(f"Processing {os.path.basename(img_path)}...")
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        # Pipeline execution
        # 1. Perception
        detections = perception.detect(frame)
        
        # 2. Distance
        detections = distance_estimator.estimate(detections, frame.shape[1])
        
        # 3. Decision
        detections, overall_status = decision_maker.analyze(detections)
        
        # 4. Alert
        alert_msg = alerter.generate_alert(overall_status, detections)
        if alert_msg:
            print(f"  [ALERT] {alert_msg}")
            
        # 5. XAI
        annotated_frame, explanation = explainer.explain(frame, detections, alert_msg)
        
        # Save output
        base_name = os.path.basename(img_path)
        out_name = os.path.join(output_path, f"processed_{base_name}")
        cv2.imwrite(out_name, annotated_frame)
        
        # Simple logging
        with open(os.path.join(output_path, "log.txt"), "a") as f:
            f.write(f"--- Frame {base_name} ---\n{explanation}\n\n")

    print("Dataset processing complete.")

def run_carla_mode(perception, distance_estimator, decision_maker, alerter, explainer, output_path):
    """
    Runs the FCW pipeline connected to the CARLA simulator.
    Handles:
    - Connection to CARLA
    - Scenario Setup (Deterministic)
    - Ground Truth Extraction
    - Event Logging
    - Alert Snapshots
    """
    out = None
    try:
        carla_interface = CarlaInterface()
        carla_interface.setup()
        carla_interface.setup_fcw_scenario()
        carla_interface.attach_camera()
        
        # Wait for simulation to settle
        import time
        time.sleep(2)
        
        print("Processing CARLA stream. Press Ctrl+C to stop.")
        
        # Setup Video Writer
        width, height = 640, 480 
        fps = 20.0 
        
        # specific settings for CARLA
        decision_maker.fps = fps
        decision_maker.dt = 1.0 / fps
        
        out_video_path = os.path.join(output_path, "carla_output.mp4")
        out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        frame_idx = 0
        while True:
            frame = carla_interface.get_frame()
            if frame is None:
                continue
            
            # --- Pipeline ---
            
            # 1. Perception
            detections = perception.detect(frame)
            
            # Ground Truth Injection (CARLA Specific)
            gt = carla_interface.get_ground_truth()
            if gt:
                for det in detections:
                    if det['label'] in ['car', 'truck', 'bus']:
                        det['ground_truth'] = gt
            
            # 2. Distance Estimation
            detections = distance_estimator.estimate(detections, frame.shape[1])
            
            # 3. Decision
            detections, overall_status = decision_maker.analyze(detections)
            
            # 4. Alert
            alert_msg = alerter.generate_alert(overall_status, detections)
            
            if alert_msg:
                print(f"  [ALERT] {alert_msg}")
                _log_carla_event(detections, output_path)
            
            # 5. XAI
            annotated_frame, explanation = explainer.explain(frame, detections, alert_msg)
            
            if alert_msg:
                _save_alert_snapshot(annotated_frame, explanation, alert_msg, output_path)

            # Show live feed
            cv2.imshow("CARLA FCW", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            out.write(annotated_frame)
            frame_idx += 1
            if frame_idx % 50 == 0:
                 print(f"Processed {frame_idx} CARLA frames")
                 
    except KeyboardInterrupt:
        print("Stopping CARLA stream...")
    except Exception as e:
        print(f"CARLA Error: {e}")
    finally:
        if 'carla_interface' in locals():
            carla_interface.cleanup()
        if out:
            out.release()
        cv2.destroyAllWindows()

def run_video_mode(video_path, perception, distance_estimator, decision_maker, alerter, explainer, output_path):
    """
    Runs the pipeline on a video file.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    decision_maker.fps = fps
    decision_maker.dt = 1.0 / fps
    
    out_video_path = os.path.join(output_path, "output_video.mp4")
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        detections = perception.detect(frame)
        detections = distance_estimator.estimate(detections, frame.shape[1])
        detections, overall_status = decision_maker.analyze(detections)
        alert_msg = alerter.generate_alert(overall_status, detections)
        annotated_frame, explanation = explainer.explain(frame, detections, alert_msg)
        
        out.write(annotated_frame)
            
    cap.release()
    out.release()
    print("Video processing complete.")

def _log_carla_event(detections, output_path):
    """Helper to log CARLA events to CSV."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for det in detections:
        # Log relevant detections
        if det.get('risk') == 'Unsafe' or 'ground_truth' in det:
             dist = det.get('distance', -1)
             speed = det.get('speed', -1)
             ttc = det.get('ttc', -1)
             gt_dist = det.get('gt_dist', -1)
             gt_speed = det.get('gt_speed', -1)
             gt_ttc = det.get('gt_ttc', -1)
             
             log_line = f"{timestamp},{dist},{speed},{ttc},{gt_dist},{gt_speed},{gt_ttc}\n"
             
             csv_path = os.path.join(output_path, "carla_events.csv")
             if not os.path.exists(csv_path):
                 with open(csv_path, 'w') as f:
                     f.write("Timestamp,Distance,Speed,TTC,GT_Distance,GT_Speed,GT_TTC\n")
             
             with open(csv_path, 'a') as f:
                 f.write(log_line)
             break 

def _save_alert_snapshot(frame, explanation, alert_msg, output_path):
    """Helper to save XAI snapshots."""
    ts_obj = datetime.datetime.now()
    file_ts = ts_obj.strftime("%Y%m%d_%H%M%S_%f")
    
    alerts_dir = os.path.join(output_path, "alerts")
    if not os.path.exists(alerts_dir):
        os.makedirs(alerts_dir)
        
    img_name = os.path.join(alerts_dir, f"alert_{file_ts}.jpg")
    cv2.imwrite(img_name, frame)
    
    txt_name = os.path.join(alerts_dir, f"alert_{file_ts}.txt")
    with open(txt_name, "w") as f:
        f.write(f"Alert Timestamp: {ts_obj.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Alert Message: {alert_msg}\n")
        f.write("\n--- Explanation ---\n")
        f.write(explanation)

if __name__ == "__main__":
    main()
