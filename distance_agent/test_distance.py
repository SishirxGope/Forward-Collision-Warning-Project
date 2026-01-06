import sys
import os
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from distance_agent.agent import DistanceEstimationAgent

class TestDistanceEstimation(unittest.TestCase):
    def setUp(self):
        self.agent = DistanceEstimationAgent(focal_length=1000)
        
    def test_car_distance(self):
        # Car: Width 1.8m, Height 1.5m
        # Scenario: Object is 10m away
        # Expected Pixel Width = (1000 * 1.8) / 10 = 180
        # Expected Pixel Height = (1000 * 1.5) / 10 = 150
        
        detection = {
            'label': 'car',
            'box': [100, 100, 280, 250] # 180w, 150h
        }
        
        res = self.agent.estimate([detection], 640)
        dist = res[0]['distance']
        print(f"\nCar Test: Est={dist}, Components: W={res[0]['dist_w']}, H={res[0]['dist_h']}")
        
        self.assertAlmostEqual(dist, 10.0, delta=0.1)
        
    def test_mismatch_dimensions(self):
        # Scenario: Car is turned, so width is smaller, but height is same.
        # Distance should be between width-est and height-est if simple average.
        # Real Width used is 1.8. 
        # If effectively 0.9m visible width -> Pixel W = (1000 * 0.9*) / 10 = 90
        # Pixel H stays 150 (since height invariant to yaw rotation mostly)
        
        # Dist W calc: (1000 * 1.8) / 90 = 20m (Incorrectly thinks it's far because it looks narrow)
        # Dist H calc: (1000 * 1.5) / 150 = 10m (Correct)
        # Avg = 15m
        
        detection = {
            'label': 'car',
            'box': [100, 100, 190, 250] # 90w, 150h
        }
        
        res = self.agent.estimate([detection], 640)
        dist = res[0]['distance']
        print(f"Mismatch Test: Est={dist}, Components: W={res[0]['dist_w']}, H={res[0]['dist_h']}")
        
        self.assertAlmostEqual(dist, 15.0, delta=0.1)

if __name__ == '__main__':
    unittest.main()
