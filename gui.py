import sys
import cv2
import math
import numpy as np
import datetime
import os
import signal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QLabel, QPushButton, QFrame, QProgressBar, QGroupBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QPoint, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QColor, QPen, QBrush

# Import Agents directly (Ensure current directory is in path or run from root)
from perception_agent.agent import PerceptionAgent
from distance_agent.agent import DistanceEstimationAgent
from monocular_3d_agent.agent import Monocular3DBoxAgent
from decision_agent.agent import DecisionAgent
from alert_agent.agent import AlertAgent
from collision_avoidance_agent.agent import CollisionAvoidanceAgent
from explainable_ai.xai import XAIModule
from carla_interface.interface import CarlaInterface

class RadarWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 200)
        self.setStyleSheet("background-color: #111; border: 2px solid #555;")
        self.dist = float('inf')
        # self.azimuth = 0.0 # Future support

    def update_target(self, dist):
        self.dist = dist
        self.update() # Trigger repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # 1. Background (Grid)
        painter.fillRect(0, 0, w, h, QColor(20, 20, 20))
        
        # Draw FOV Cone (Visual flair)
        # Assuming +/- 15 degrees width at top
        fov_poly = [
            QPoint(int(w/2), h),
            QPoint(0, 0),
            QPoint(w, 0)
        ]
        
        # Semi-transparent gradient for beam
        beam_color = QColor(0, 150, 255, 30)
        painter.setBrush(QBrush(beam_color))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(*fov_poly)

        # Grid Lines & Labels
        pen = QPen(QColor(60, 60, 60))
        pen.setStyle(Qt.DotLine)
        painter.setPen(pen)
        
        font = QFont("Arial", 8)
        painter.setFont(font)
        
        # Dimensions for scale calculation
        ego_h = 36
        ego_y_top = h - ego_h - 5 # 0m Reference Point
        available_height = ego_y_top
        
        # Scale: available_height pixels = 50m (Max Range)
        max_range = 50.0
        px_per_m = available_height / max_range
        
        for d in range(10, 60, 10): # 10, 20, 30, 40, 50
            y = ego_y_top - (d * px_per_m)
            if y < 0: continue
            
            painter.setPen(pen)
            painter.drawLine(0, int(y), w, int(y))
            
            # Label
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(5, int(y) - 2, f"{d}m")
            
        # Center line
        painter.setPen(QPen(QColor(60, 60, 60), 1, Qt.SolidLine))
        painter.drawLine(int(w/2), 0, int(w/2), int(ego_y_top)) # Draw only up to bumper

        # 2. Ego Vehicle (Bottom Center)
        painter.setBrush(QBrush(QColor(0, 200, 255))) # Cyan
        painter.setPen(Qt.NoPen)
        ego_w = 24
        # ego_h defined above
        # ego_y_top defined above
        painter.drawRoundedRect(int(w/2 - ego_w/2), int(ego_y_top), ego_w, ego_h, 5, 5)
        
        # 3. Target Object
        if self.dist < max_range:
            # Correct Mapping:
            # pixel_offset = (distance / max_range) * available_height
            pixel_offset = (self.dist / max_range) * available_height
            
            # dot_y = ego_y_top - pixel_offset
            scaled_y = ego_y_top - pixel_offset
            
            # Safety Constraint:
            # "Ensure... dot appears just above the ego vehicle" when dist is small.
            radius = 10
            min_y = radius + 2 # Keep away from top edge
            max_y = ego_y_top - radius - 2 # Keep away from Ego Icon
            
            # Clamp logic
            target_y = max(min_y, min(max_y, scaled_y))
            
            # Color based on distance
            color = QColor(0, 255, 0) # Green
            if self.dist < 15: color = QColor(255, 0, 0) # Red
            elif self.dist < 30: color = QColor(255, 255, 0) # Yellow
            
            # Draw Target with Glow effect
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            
            target_x = w/2 
            
            painter.drawEllipse(QPoint(int(target_x), int(target_y)), radius, radius)
            
            # Draw distance text next to target
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(int(target_x) + 15, int(target_y) + 5, f"{self.dist:.1f}m")

class WorkerThread(QThread):
    # Signals
    frame_signal = pyqtSignal(np.ndarray)
    radar_signal = pyqtSignal(float, float, float) # Dist, Speed, TTC
    status_signal = pyqtSignal(str, str) # Status, Action
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.output_path = "outputs_gui"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def run(self):
        print("Worker Thread Started. Initializing Agents...")
        
        # Initialize Agents
        perception = PerceptionAgent() 
        decision_maker = DecisionAgent(fps=20)
        alerter = AlertAgent()
        collision_avoider = CollisionAvoidanceAgent()
        monocular_3d = Monocular3DBoxAgent()
        explainer = XAIModule(monocular_agent=monocular_3d)
        
        # Initialize CARLA
        try:
            carla_interface = CarlaInterface()
            carla_interface.setup()
            carla_interface.setup_fcw_scenario()
            carla_interface.attach_camera()
            carla_interface.attach_radar(v_fov=5.0, range=50.0) # Use the fixed values
            
            # Allow settle time
            self.msleep(2000)
            collision_avoider.reset()
            
            # Main Loop
            width, height = 640, 480 
            f_carla = (width / 2.0) / math.tan(math.radians(90.0 / 2.0))
            
            while self.running:
                frame = carla_interface.get_frame(timeout=0.5)
                if frame is None:
                    continue
                
                # --- Pipeline Logic (Copied/Adapted from main.py) ---
                detections = perception.detect(frame)
                gt = carla_interface.get_ground_truth()
                radar_obj = carla_interface.get_closest_radar_object()
                
                radar_dist = float('inf')
                radar_speed = 0.0
                radar_ttc = float('inf')
                
                if radar_obj:
                    radar_dist = radar_obj['depth']
                    radar_speed = radar_obj['velocity']
                    if radar_speed < -0.1:
                        radar_ttc = radar_dist / abs(radar_speed)
                    else:
                        radar_ttc = float('inf')
                        
                # Update GUI with Radar Stats immediately
                self.radar_signal.emit(radar_dist, radar_speed, radar_ttc)
                
                # Fusion Logic
                radar_claimed = False
                for det in detections:
                    if radar_obj and det['label'] in ['car', 'truck', 'bus']:
                        det['radar_available'] = True
                        det['radar_dist'] = radar_dist
                        det['radar_speed'] = radar_speed
                        det['radar_ttc'] = radar_ttc
                        det['distance'] = radar_dist
                        det['speed'] = radar_speed
                        det['ttc'] = radar_ttc
                        radar_claimed = True
                    
                    if gt and det['label'] in ['car', 'truck', 'bus']:
                        det['ground_truth'] = gt

                # Ghost Object
                if radar_obj and not radar_claimed:
                    ghost_det = {
                        'label': 'obstacle (radar)',
                        'box': [0, 0, width, height],
                        'confidence': 1.0,
                        'distance': radar_dist,
                        'speed': radar_speed,
                        'ttc': radar_ttc,
                        'radar_available': True,
                        'radar_dist': radar_dist,
                        'radar_speed': radar_speed,
                        'radar_ttc': radar_ttc
                    }
                    detections.append(ghost_det)
                
                # Decision & Control
                detections, overall_status = decision_maker.analyze(detections)
                alert_msg = alerter.generate_alert(overall_status, detections)
                ego_speed = carla_interface.get_ego_speed()
                action = collision_avoider.get_control_action(detections, overall_status, ego_speed)
                
                carla_interface.apply_control(action)
                
                # Update GUI Status
                self.status_signal.emit(overall_status, action)
                
                # XAI & Visualization
                for det in detections:
                    det['box_3d'] = monocular_3d.compute_3d_box(det, frame.shape, focal_length=f_carla)
                
                annotated_frame, _ = explainer.explain(frame, detections, alert_msg, action, ego_speed, collision_avoider.emergency_latch)
                
                # Emit Frame
                self.frame_signal.emit(annotated_frame)
                
        except Exception as e:
            print(f"Worker Error: {e}")
        finally:
            print("Cleaning up Worker...")
            if 'carla_interface' in locals():
                carla_interface.cleanup()

    def stop(self):
        self.running = False
        self.wait()

class FCWGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CARLA Forward Collision Warning System")
        self.resize(1000, 600)
        
        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main Layout (Horizontal)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # --- Left Panel: Camera Feed ---
        self.camera_label = QLabel("Camera Feed Loading...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: #000; color: #FFF; border: 2px solid #333;")
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setSizePolicy(1, 1) # Expanding
        
        self.main_layout.addWidget(self.camera_label, stretch=2)
        
        # --- Right Panel: Radar & Status ---
        self.right_panel = QFrame()
        self.right_panel.setFixedWidth(350)
        self.right_panel.setStyleSheet("""
            QFrame { background-color: #222; color: #EEE; border: none; }
            QGroupBox { 
                border: 2px solid #555; 
                border-radius: 8px; 
                margin-top: 20px; 
                font-weight: bold; 
                color: #AAA; 
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }
            QLabel { color: #EEE; }
            QPushButton { 
                background-color: #AA0000; 
                color: white; 
                border-radius: 5px; 
                padding: 10px; 
                font-weight: bold; 
                font-size: 14px;
            }
            QPushButton:hover { background-color: #CC0000; }
        """)
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(15, 15, 15, 15)
        self.right_layout.setSpacing(15)
        
        # Font Styles
        data_font = QFont("Consolas", 11)
        
        # --- 1. System Status Group ---
        self.grp_status = QGroupBox("SYSTEM STATUS")
        self.layout_status = QVBoxLayout()
        
        self.lbl_status = QLabel("SAFE")
        self.lbl_status.setFont(QFont("Arial", 28, QFont.Bold))
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #FFFFFF; background-color: #00AA00; padding: 10px; border-radius: 8px;")
        
        self.lbl_action = QLabel("Action: Maintain Speed")
        self.lbl_action.setFont(data_font)
        self.lbl_action.setAlignment(Qt.AlignCenter)
        self.lbl_action.setWordWrap(True)
        self.lbl_action.setStyleSheet("color: #DDD; margin-top: 5px;")
        
        self.layout_status.addWidget(self.lbl_status)
        self.layout_status.addWidget(self.lbl_action)
        self.grp_status.setLayout(self.layout_status)
        self.right_layout.addWidget(self.grp_status)
        
        # --- 2. Radar Data Group ---
        self.grp_radar = QGroupBox("RADAR SENSOR")
        self.layout_radar = QVBoxLayout()
        self.layout_radar.setSpacing(8)
        
        # Distance
        self.lbl_dist = QLabel("Distance:  -- m")
        self.lbl_dist.setFont(data_font)
        self.bar_dist = QProgressBar()
        self.bar_dist.setRange(0, 50)
        self.bar_dist.setTextVisible(False)
        self.bar_dist.setFixedHeight(8)
        self.bar_dist.setStyleSheet("QProgressBar { background: #333; border: 1px solid #555; border-radius: 4px; } QProgressBar::chunk { background-color: green; border-radius: 3px; }")
        
        # TTC
        self.lbl_ttc = QLabel("TTC:       -- s")
        self.lbl_ttc.setFont(data_font)
        self.bar_ttc = QProgressBar()
        self.bar_ttc.setRange(0, 60)
        self.bar_ttc.setTextVisible(False)
        self.bar_ttc.setFixedHeight(8)
        self.bar_ttc.setStyleSheet("QProgressBar { background: #333; border: 1px solid #555; border-radius: 4px; } QProgressBar::chunk { background-color: green; border-radius: 3px; }")
        
        # Speed
        self.lbl_speed = QLabel("Rel Speed: -- m/s")
        self.lbl_speed.setFont(data_font)

        self.layout_radar.addWidget(self.lbl_dist)
        self.layout_radar.addWidget(self.bar_dist)
        self.layout_radar.addWidget(self.lbl_ttc)
        self.layout_radar.addWidget(self.bar_ttc)
        self.layout_radar.addWidget(self.lbl_speed)
        self.grp_radar.setLayout(self.layout_radar)
        self.right_layout.addWidget(self.grp_radar)
        
        # --- 3. Visualization Group ---
        self.grp_vis = QGroupBox("RADAR MAP")
        self.layout_vis = QVBoxLayout()
        
        self.radar_view = RadarWidget()
        hbox_radar = QHBoxLayout()
        hbox_radar.addStretch()
        hbox_radar.addWidget(self.radar_view)
        hbox_radar.addStretch()
        
        self.layout_vis.addLayout(hbox_radar)
        self.grp_vis.setLayout(self.layout_vis)
        self.right_layout.addWidget(self.grp_vis)
        
        self.right_layout.addStretch()
        
        # Controls
        self.btn_quit = QPushButton("QUIT SYSTEM")
        self.btn_quit.clicked.connect(self.close)
        self.right_layout.addWidget(self.btn_quit)
        
        self.main_layout.addWidget(self.right_panel, stretch=0)
        
        # --- Start Worker Thread ---
        self.worker = WorkerThread()
        self.worker.frame_signal.connect(self.update_frame)
        self.worker.radar_signal.connect(self.update_radar_data)
        self.worker.status_signal.connect(self.update_status)
        self.worker.start()

    def closeEvent(self, event):
        """Handle cleanup on window close."""
        self.cleanup()
        event.accept()

    def closeEvent_explicit(self):
        """Called by app.aboutToQuit."""
        self.cleanup()

    def cleanup(self):
        """Stop worker and clean up resources."""
        if hasattr(self, 'worker') and self.worker.isRunning():
            print("Stopping Worker Thread...")
            self.worker.stop()
            print("Worker Stopped.")

    @pyqtSlot(np.ndarray)
    def update_frame(self, cv_img):
        """
        Updates the camera label with a new OpenCV image.
        Args:
            cv_img (np.ndarray): Image in BGR format.
        """
        if cv_img is None:
            return
            
        try:
            # Convert BGR (OpenCV) to RGB (Qt)
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            
            # Create QImage
            qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale to fit label (keeping aspect ratio)
            scaled_pixmap = QPixmap.fromImage(qt_img).scaled(
                self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            self.camera_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"GUI Error: {e}")

    @pyqtSlot(float, float, float)
    def update_radar_data(self, dist, speed, ttc):
        """
        Updates the Radar Data fields and bars.
        """
        self.lbl_dist.setText(f"Distance:  {dist:.2f} m")
        
        # Speed Display Logic
        # Negative speed = Closing (Getting closer)
        # Positive speed = Opening (Moving away)
        abs_speed = abs(speed)
        if speed < -0.1:
            self.lbl_speed.setText(f"Closing:   {abs_speed:.2f} m/s")
            self.lbl_speed.setStyleSheet("color: #FF5555;") # Red-ish for closing
        elif speed > 0.1:
            self.lbl_speed.setText(f"Opening:   {abs_speed:.2f} m/s")
            self.lbl_speed.setStyleSheet("color: #55FF55;") # Green-ish for opening
        else:
            self.lbl_speed.setText(f"Speed:     {abs_speed:.2f} m/s")
            self.lbl_speed.setStyleSheet("color: #EEE;") # Neutral
            
        # Update Distance Bar (0-50m)
        d_val = int(max(0, min(50, dist)))
        self.bar_dist.setValue(d_val)
        
        # Update Radar Visualization
        self.radar_view.update_target(dist)
        
        # Distance Color
        
        # Distance Color
        if dist < 15:
            self.bar_dist.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        elif dist < 30:
            self.bar_dist.setStyleSheet("QProgressBar::chunk { background-color: yellow; }")
        else:
            self.bar_dist.setStyleSheet("QProgressBar::chunk { background-color: green; }")

        
        if ttc == float('inf'):
            self.lbl_ttc.setText(f"TTC:       Inf s")
            self.bar_ttc.setValue(60) # Max
            self.bar_ttc.setStyleSheet("QProgressBar::chunk { background-color: green; }")
        else:
            self.lbl_ttc.setText(f"TTC:       {ttc:.2f} s")
            # Update TTC Bar (0-6s -> 0-60)
            t_val = int(max(0, min(60, ttc * 10)))
            self.bar_ttc.setValue(t_val)
            
            # TTC Color
            if ttc < 2.5:
                self.bar_ttc.setStyleSheet("QProgressBar::chunk { background-color: red; }")
            elif ttc < 4.5:
                self.bar_ttc.setStyleSheet("QProgressBar::chunk { background-color: yellow; }")
            else:
                self.bar_ttc.setStyleSheet("QProgressBar::chunk { background-color: green; }")

    @pyqtSlot(str, str)
    def update_status(self, status, action):
        """
        Updates the System Status and Action.
        """
        self.lbl_status.setText(status)
        self.lbl_action.setText(f"Action: {action}")
        
        # Color Coding
        # Green = SAFE
        if status == "Safe":
            self.lbl_status.setStyleSheet("color: #FFFFFF; background-color: #00AA00; padding: 15px; border-radius: 8px;")
        # Yellow = WARNING
        elif status == "Warning":
            self.lbl_status.setStyleSheet("color: #000000; background-color: #FFDD00; padding: 15px; border-radius: 8px;")
        # Red = BRAKING / Unsafe
        elif status == "Unsafe" or "Braking" in action:
             # Regular Red
            self.lbl_status.setStyleSheet("color: #FFFFFF; background-color: #FF0000; padding: 15px; border-radius: 8px;")
        # Dark Red = STOP
        elif status == "STOP":
            self.lbl_status.setStyleSheet("color: #FFFFFF; background-color: #880000; border: 3px solid #FF0000; padding: 15px; border-radius: 8px;")
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Allow Ctrl+C to terminate
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    window = FCWGui()
    window.show()
    
    # Robust cleanup on app exit
    app.aboutToQuit.connect(window.closeEvent_explicit)
    
    sys.exit(app.exec_())
