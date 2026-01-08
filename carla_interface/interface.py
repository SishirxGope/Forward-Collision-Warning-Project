import carla
import random
import time
import numpy as np
import cv2
from queue import Queue
from queue import Empty

class CarlaInterface:
    def __init__(self, host='localhost', port=2000):
        self.host = host
        self.port = port
        self.client = None
        self.world = None
        self.vehicle = None
        self.lead_vehicle = None # Track lead vehicle
        self.camera = None
        self.radar = None
        self.image_queue = Queue()
        self.radar_queue = Queue()
        
    def setup(self):
        """Connect to CARLA and setup the world."""
        print(f"Connecting to CARLA at {self.host}:{self.port}...")
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            
            # Initialize Traffic Manager with a specific port
            self.tm_port = 8001
            try:
                self.tm = self.client.get_trafficmanager(self.tm_port)
                # Check if it works by setting a global parameter (optional but good for validation)
                self.tm.set_global_distance_to_leading_vehicle(2.5)
                print(f"Traffic Manager initialized on port {self.tm_port}.")
            except Exception as e:
                print(f"Failed to initialize Traffic Manager on port {self.tm_port}: {e}")
                # We should exit or raise because autopilot depends on it
                raise
                
            print("Connected to CARLA.")
        except Exception as e:
            print(f"Error connecting to CARLA: {e}")
            raise

    def setup_fcw_scenario(self, ego_filter='vehicle.tesla.model3', lead_filter='vehicle.nissan.patrol', lead_distance=20.0):
        """
        Sets up a deterministic Forward Collision Warning scenario.
        
        Scenario:
        1. Finds a spawn point.
        2. Spawns the ego vehicle.
        3. Spawns a lead vehicle directly ahead of the ego vehicle at a fixed distance.
        4. Sets both vehicles to autopilot (or could set constant velocity).
        """
        try:
           # Get Blueprint Library
            bp_lib = self.world.get_blueprint_library()
            ego_bp = bp_lib.find(ego_filter)
            ego_bp.set_attribute('role_name', 'ego_vehicle')
            
            lead_bp = bp_lib.find(lead_filter)
            lead_bp.set_attribute('role_name', 'lead_vehicle')
            
            # --- Robust Cleanup ---
            # Search for existing actors with these role names and destroy them
            # This handles cases where previous run didn't cleanup properly
            print("Checking for existing actors to clean up...")
            actor_list = self.world.get_actors()
            for actor in actor_list:
                if 'role_name' in actor.attributes:
                    if actor.attributes['role_name'] in ['ego_vehicle', 'lead_vehicle']:
                        print(f"Destroying stale actor: {actor.id}")
                        actor.destroy()
            
            # 1. Select a spawn point
            # We want a road, preferably straight.
            # get_spawn_points returns a list of Transform objects.
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise ValueError("No spawn points found.")
            
            # Pick a spawn point. For determinism, we could pick a specific index if we knew the map,
            # but picking the first one (index 0) is deterministic for a given map.
            # To be safe against bad points, we can loop until we find a valid pair.
            
            ego_spawn_point = None
            lead_spawn_point = None
            
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise ValueError("No spawn points found.")
            
            spawn_success = False
            
            # Helper to calculate lead position
            import math
            
            # Check a few points (e.g., first 5 or 10) to avoid infinite wait if map is full
            # Randomize start index to avoid stuck on same bad point? Or just iterate.
            # Let's iterate.
            for i in range(min(10, len(spawn_points))):
                 sp = spawn_points[i]
                 
                 ego_spawn_point = sp
                 
                 # Calculate lead position
                 yaw_rad = math.radians(sp.rotation.yaw)
                 lx = sp.location.x + lead_distance * math.cos(yaw_rad)
                 ly = sp.location.y + lead_distance * math.sin(yaw_rad)
                 lz = sp.location.z + 0.5 
                 
                 lead_loc = carla.Location(x=lx, y=ly, z=lz)
                 lead_spawn_point = carla.Transform(lead_loc, sp.rotation)
                 
                 try:
                     print(f"Attempting spawn at index {i}...")
                     self.vehicle = self.world.spawn_actor(ego_bp, ego_spawn_point)
                     self.lead_vehicle = self.world.spawn_actor(lead_bp, lead_spawn_point)
                     spawn_success = True
                     print(f"Spawn successful at {ego_spawn_point.location}")
                     break
                 except Exception as e:
                     print(f"Spawn failed at index {i} ({e}). Retrying...")
                     if self.vehicle: 
                         self.vehicle.destroy()
                         self.vehicle = None
                     if self.lead_vehicle: 
                         self.lead_vehicle.destroy()
                         self.lead_vehicle = None
                     continue

            if not spawn_success:
                 raise RuntimeError("Failed to spawn vehicles after multiple attempts (Collision or Invalid Location).")
            
            # Set Autopilot ONLY for Lead Vehicle
            self.vehicle.set_autopilot(False, self.tm_port) # Ensure Ego is manual
            self.lead_vehicle.set_autopilot(False, self.tm_port) # Stop lead vehicle for braking test
            
            # Apply brakes to lead vehicle to ensure it stays put
            control_lead = carla.VehicleControl()
            control_lead.brake = 1.0
            control_lead.hand_brake = True
            self.lead_vehicle.apply_control(control_lead)
            
            # NOTE: For even more determinism (e.g. forced collision), we could set velocity vectors directly.
            # But autopilot is good for "traffic flow" scenario.
            
            print("Scenario Setup Complete: Ego (Manual) and Lead (Autopilot) vehicles spawned.")
            
        except Exception as e:
            print(f"Failed to setup scenario: {e}")
            self.cleanup() # Cleanup any partial spawns
            raise

    def attach_camera(self, image_width=640, image_height=480, fov=90):
        """Attaches an RGB camera to the ego vehicle."""
        if not self.vehicle:
            raise ValueError("Vehicle not spawned. Call spawn_ego_vehicle() first.")
            
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(image_width))
        camera_bp.set_attribute('image_size_y', str(image_height))
        camera_bp.set_attribute('fov', str(fov))
        
        # Position the camera on the hood/dash of Tesla Model 3 (adjusted approx)
        spawn_point = carla.Transform(carla.Location(x=1.5, z=2.4))
        
        self.camera = self.world.spawn_actor(camera_bp, spawn_point, attach_to=self.vehicle)
        self.camera.listen(self.process_image)
        print("Attached RGB camera.")

    def process_image(self, image):
        """Callback for camera sensor to put images in the queue."""
        try:
            # Convert raw CARLA image to numpy array for OpenCV
            i = np.array(image.raw_data)
            i2 = i.reshape((image.height, image.width, 4))
            # CARLA images are BGRA, opencv needs BGR
            i3 = i2[:, :, :3] 
            self.image_queue.put(i3)
        except Exception as e:
            print(f"Error in process_image: {e}", flush=True)

    def attach_radar(self, range=50.0, h_fov=35.0, v_fov=5.0):
        """Attaches a Radar sensor to the ego vehicle."""
        if not self.vehicle:
            raise ValueError("Vehicle not spawned. Call spawn_ego_vehicle() first.")
            
        bp_lib = self.world.get_blueprint_library()
        radar_bp = bp_lib.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', str(h_fov))
        radar_bp.set_attribute('vertical_fov', str(v_fov))
        radar_bp.set_attribute('range', str(range))
        
        # Position the radar on the front bumper (approx x=2.3, z=0.5 for Model 3)
        # UPDATED: Move forward and up to avoid self-collision/ground hits (x=2.8, z=1.0)
        spawn_point = carla.Transform(carla.Location(x=2.8, z=1.0))
        
        self.radar = self.world.spawn_actor(radar_bp, spawn_point, attach_to=self.vehicle)
        self.radar.listen(self.process_radar)
        print("Attached Radar sensor.")

    def process_radar(self, data):
        """Callback for radar sensor."""
        try:
            # data is a carla.RadarMeasurement
            # Just push it to the queue
            self.radar_queue.put(data)
        except Exception as e:
            print(f"Error in process_radar: {e}", flush=True)

    def get_closest_radar_object(self, timeout=0.05):
        """
        Retrieves the latest radar packet and filters for the closest relevant object.
        Filters:
        - Azimuth within +/- 10 degrees (Focus on lane)
        - Sort by Depth (Closest)
        
        Returns:
            dict: {'depth': float, 'velocity': float, 'azimuth': float} or None
        """
        try:
            # Get latest, flush old
            last_data = None
            while not self.radar_queue.empty():
                try:
                    last_data = self.radar_queue.get(timeout=timeout)
                except Empty:
                    break
            
            if last_data is None:
                return None
                
            # Process detections
            closest_det = None
            min_depth = float('inf')
            
            # Thresholds
            azimuth_limit = 0.20 # ~11 degrees (radians)
            
            for detect in last_data:
                # detect has depth, azimuth, altitude, velocity
                # Check Azimuth (Forward Cone)
                if abs(detect.azimuth) > azimuth_limit:
                    continue
                    
                if detect.depth < min_depth:
                    min_depth = detect.depth
                    closest_det = detect
            
            if closest_det:
                return {
                    'depth': closest_det.depth,
                    'velocity': closest_det.velocity, # m/s
                    'azimuth': closest_det.azimuth
                }
            return None
            
        except Exception as e:
            print(f"Error in get_closest_radar_object: {e}", flush=True)
            return None

    def get_frame(self, timeout=1.0):
        """Retrieve the latest frame from the queue."""
        try:
            return self.image_queue.get(timeout=timeout)
        except Empty:
            return None

    def cleanup(self):
        """Clean up actors."""
        print("Cleaning up CARLA actors...")
        if self.camera:
            try:
                self.camera.stop()
                self.camera.destroy()
            except: pass
        if self.radar:
            try:
                self.radar.stop()
                self.radar.destroy()
            except: pass
        if self.vehicle:
            try:
                self.vehicle.destroy()
            except: pass
        if self.lead_vehicle:
            try:
                self.lead_vehicle.destroy()
            except: pass
            
        print("Cleanup done.")

    def get_ground_truth(self):
        """
        Returns ground truth distance and relative speed to the lead vehicle.
        Returns:
            dict: {'distance': float, 'speed': float} or None if not available.
        """
        if not self.vehicle or not self.lead_vehicle:
            return None
        
        # Location
        loc_ego = self.vehicle.get_location()
        loc_lead = self.lead_vehicle.get_location()
        
        # Euclidean distance
        distance = loc_lead.distance(loc_ego)
        
        # Euclidean distance between transform locations (Center-to-Center)
        distance = loc_lead.distance(loc_ego)
        
        # User requested pure Euclidean distance between transform locations.
        # Removed bumper-to-bumper adjustment (-4.9m).
        
        # Velocity (m/s)
        vel_ego = self.vehicle.get_velocity()
        vel_lead = self.lead_vehicle.get_velocity()
        
        # Calculate speed projected onto ego's forward vector for accuracy
        fwd = self.vehicle.get_transform().get_forward_vector()
        
        speed_ego = vel_ego.x * fwd.x + vel_ego.y * fwd.y + vel_ego.z * fwd.z
        speed_lead = vel_lead.x * fwd.x + vel_lead.y * fwd.y + vel_lead.z * fwd.z
        
        # Relative speed: positive means closing in (ego faster than lead)
        relative_speed = speed_ego - speed_lead
        
        return {
            'distance': round(distance, 2),
            'speed': round(relative_speed, 2)
        }

    def apply_control(self, action):
        """
        Applies longitudinal control (throttle/brake) based on the action string.
        Disables autopilot if intervention is needed.
        
        Args:
            action (str): One of ['Maintain Speed', 'Warning Only', 'Apply Braking', 'Apply Full Braking']
        """
        if not self.vehicle:
            return

        control = carla.VehicleControl()
        # Default: no steering, no reverse
        control.steer = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        # Always disable Autopilot to ensure Manual Control
        try:
            self.vehicle.set_autopilot(False, self.tm_port)
        except Exception:
            pass # Ignore TM errors, control override will handle it

        if action == "Maintain Speed" or action == "Warning Only":
            # Apply constant low throttle to move forward (Initial Motion)
            control.throttle = 0.3 
            control.brake = 0.0
            
        elif action == "Apply Braking":
            control.throttle = 0.0
            control.brake = 1.0 # Increased from 0.5 to 1.0 for stronger response
            
        elif action == "Apply Full Braking":
            # Should be covered by latch, but for completeness
            control.throttle = 0.0
            control.brake = 1.0 # Full emergency braking
            control.hand_brake = True # Force stop
            
        else:
            # Fallback
            control.throttle = 0.0
            control.brake = 0.0

            control.brake = 0.0
            
        print(f"[DEBUG] Applying Control: {action} | Throttle: {control.throttle} | Brake: {control.brake} | Handbrake: {control.hand_brake}", flush=True)

        self.vehicle.apply_control(control)

    def get_ego_speed(self):
        """Returns the current speed of the ego vehicle in m/s."""
        if not self.vehicle:
            return 0.0
        vel = self.vehicle.get_velocity()
        return (vel.x**2 + vel.y**2 + vel.z**2)**0.5

