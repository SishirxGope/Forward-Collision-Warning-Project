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
        self.image_queue = Queue()
        
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
            lead_bp = bp_lib.find(lead_filter)
            
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
            
            # Simple search for a valid pair
            for i in range(len(spawn_points)):
                 sp = spawn_points[i]
                 
                 # Calculate where lead vehicle would be
                 # Simple approximation: move 'lead_distance' forward along yaw
                 import math
                 yaw_rad = math.radians(sp.rotation.yaw)
                 lx = sp.location.x + lead_distance * math.cos(yaw_rad)
                 ly = sp.location.y + lead_distance * math.sin(yaw_rad)
                 lz = sp.location.z + 0.5 # slightly up
                 
                 # Check if this point is drivable is hard without map query, 
                 # but we can try to spawn and see if it collides.
                 # Better approach for simplicity: Just assume the road is straight enough at spawn point 0.
                 # Or use specific points if we knew the map (Town04, etc).
                 # Let's stick to the previous simple arithmetic logic but applied to the object.
                 
                 ego_spawn_point = sp
                 
                 # Create a transform for lead vehicle
                 lead_loc = carla.Location(x=lx, y=ly, z=lz)
                 lead_spawn_point = carla.Transform(lead_loc, sp.rotation)
                 
                 # In a real rigorous setup we would check waypoint connectivity.
                 break
            
            if not ego_spawn_point:
                raise ValueError("Could not find suitable spawn point.")
                
            # 2. Spawn Ego Vehicle
            print(f"Spawning Ego Vehicle at {ego_spawn_point.location}...")
            self.vehicle = self.world.spawn_actor(ego_bp, ego_spawn_point)
            
            # 3. Spawn Lead Vehicle
            print(f"Spawning Lead Vehicle at {lead_spawn_point.location}...")
            self.lead_vehicle = self.world.spawn_actor(lead_bp, lead_spawn_point)
            
            # Set Autopilot for both using the explicit TM instance
            self.vehicle.set_autopilot(True, self.tm_port)
            self.lead_vehicle.set_autopilot(True, self.tm_port)
            
            # NOTE: For even more determinism (e.g. forced collision), we could set velocity vectors directly.
            # But autopilot is good for "traffic flow" scenario.
            
            print("Scenario Setup Complete: Ego and Lead vehicle spawned.")
            
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
        # Convert raw CARLA image to numpy array for OpenCV
        i = np.array(image.raw_data)
        i2 = i.reshape((image.height, image.width, 4))
        # CARLA images are BGRA, opencv needs BGR
        i3 = i2[:, :, :3] 
        self.image_queue.put(i3)

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
        
        # Account for vehicle length roughly (center to center vs bumper to bumper)
        # Model 3 length ~4.7m, Nissan Patrol ~5.1m. Center to center distance includes half lengths.
        # We want bumper-to-bumper.
        # Approx deduction: (4.7/2) + (5.1/2) = 2.35 + 2.55 = 4.9m
        distance = max(0.0, distance - 4.9)
        
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

