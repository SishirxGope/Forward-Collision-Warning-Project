import carla
import time
import sys

def test_connection(host='localhost', port=2000):
    print(f"Attempting to connect to CARLA at {host}:{port}...")
    try:
        client = carla.Client(host, port)
        client.set_timeout(5.0)
        
        print("Client created. Fetching world...")
        world = client.get_world()
        print(f"Successfully connected to World: {world.get_map().name}")
        
        # Check Traffic Manager
        tm_port = 8001
        print(f"Checking Traffic Manager on port {tm_port}...")
        tm = client.get_trafficmanager(tm_port)
        print("Traffic Manager connected.")
        
        # Check Blueprint Library
        bp_lib = world.get_blueprint_library()
        print(f"Blueprint Library accessed. Found {len(bp_lib)} blueprints.")
        
        return True
    except Exception as e:
        print(f"FAILED to connect: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    if success:
        print("\nConnection Test PASSED.")
        sys.exit(0)
    else:
        print("\nConnection Test FAILED. Please ensure CARLA server is running.")
        sys.exit(1)
