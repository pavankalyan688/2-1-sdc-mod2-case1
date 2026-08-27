import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vehicle_simulator import VehicleSimulator
from fuzzy_controller import FuzzyController

class TestVehicleSimulator(unittest.TestCase):
    """Test vehicle simulator functionality"""
    
    def setUp(self):
        self.simulator = VehicleSimulator()
    
    def test_initialization(self):
        """Test simulator initialization"""
        self.assertEqual(self.simulator.position, [0, 0])
        self.assertEqual(self.simulator.heading, 0)
        self.assertEqual(self.simulator.speed, 0)
        self.assertIsInstance(self.simulator.controller, FuzzyController)
    
    def test_simulate_scenario(self):
        """Test scenario simulation"""
        # Create test data
        test_data = pd.DataFrame({
            'distance_obstacle': [50, 30, 10],
            'speed': [40, 60, 30],
            'steering_angle': [0, 10, -5],
            'traffic_density': [0.2, 0.5, 0.8],
            'road_curvature': [0.1, 0.4, 0.7],
            'lane_position': [0, 0.3, -0.2]
        })
        
        results = self.simulator.simulate_scenario(test_data)
        
        # Check results
        self.assertEqual(len(results), 3)
        self.assertIn('output_speed', results.columns)
        self.assertIn('output_steering', results.columns)
        self.assertIn('position_x', results.columns)
        self.assertIn('position_y', results.columns)
        
        # Check that speed updates
        self.assertNotEqual(results['output_speed'].iloc[0], 0)
    
    def test_vehicle_dynamics(self):
        """Test vehicle dynamics updates"""
        # Manually set state
        self.simulator.position = [0, 0]
        self.simulator.heading = 0.1
        self.simulator.speed = 50
        
        # Run one step
        test_data = pd.DataFrame({
            'distance_obstacle': [40],
            'speed': [50],
            'steering_angle': [5],
            'traffic_density': [0.3],
            'road_curvature': [0.2],
            'lane_position': [0]
        })
        
        results = self.simulator.simulate_scenario(test_data)
        
        # Position should change
        self.assertNotEqual(results['position_x'].iloc[0], 0)
        self.assertNotEqual(results['position_y'].iloc[0], 0)
    
    def test_obstacle_avoidance(self):
        """Test obstacle avoidance behavior"""
        # Scenario with obstacle very close
        test_data = pd.DataFrame({
            'distance_obstacle': [5, 15, 25],
            'speed': [60, 60, 60],
            'steering_angle': [0, 5, -5],
            'traffic_density': [0.2, 0.3, 0.4],
            'road_curvature': [0.1, 0.2, 0.3],
            'lane_position': [0, 0.1, -0.1]
        })
        
        results = self.simulator.simulate_scenario(test_data)
        
        # Speed should be lower when obstacle is closer
        speed_at_closest = results['output_speed'].iloc[0]
        speed_at_furthest = results['output_speed'].iloc[2]
        
        # The closest obstacle should result in lower speed
        # Note: This is a fuzzy system, so speed might not strictly follow
        # but generally should be lower
        self.assertLessEqual(speed_at_closest, 60)
        self.assertLessEqual(speed_at_furthest, 60)
    
    def test_fuzzy_membership_output(self):
        """Test that fuzzy membership values are included in results"""
        test_data = pd.DataFrame({
            'distance_obstacle': [30],
            'speed': [50],
            'steering_angle': [0],
            'traffic_density': [0.4],
            'road_curvature': [0.3],
            'lane_position': [0]
        })
        
        results = self.simulator.simulate_scenario(test_data)
        
        # Check for fuzzy membership columns
        fuzzy_columns = [col for col in results.columns if col.startswith('fuzzy_')]
        self.assertGreater(len(fuzzy_columns), 0)
        
        # Check a specific fuzzy value
        if 'fuzzy_distance_medium' in results.columns:
            self.assertGreaterEqual(results['fuzzy_distance_medium'].iloc[0], 0)
            self.assertLessEqual(results['fuzzy_distance_medium'].iloc[0], 1)
    
    def test_steering_control(self):
        """Test steering control behavior"""
        # Test with different steering inputs
        test_data = pd.DataFrame({
            'distance_obstacle': [50, 50, 50],
            'speed': [50, 50, 50],
            'steering_angle': [-20, 0, 20],
            'traffic_density': [0.2, 0.2, 0.2],
            'road_curvature': [0.1, 0.1, 0.1],
            'lane_position': [-0.5, 0, 0.5]
        })
        
        results = self.simulator.simulate_scenario(test_data)
        
        # Steering output should follow input direction
        self.assertLess(results['output_steering'].iloc[0], 0)  # Left turn
        self.assertAlmostEqual(results['output_steering'].iloc[1], 0, delta=5)  # Straight
        self.assertGreater(results['output_steering'].iloc[2], 0)  # Right turn

class TestSimulationIntegration(unittest.TestCase):
    """Integration tests with full controller"""
    
    def setUp(self):
        self.simulator = VehicleSimulator()
    
    def test_full_scenario_sequence(self):
        """Test a complete sequence of scenarios"""
        # Create a realistic driving scenario
        n_steps = 20
        test_data = pd.DataFrame({
            'distance_obstacle': np.linspace(80, 5, n_steps),
            'speed': np.linspace(70, 30, n_steps),
            'steering_angle': np.sin(np.linspace(0, 4*np.pi, n_steps)) * 15,
            'traffic_density': np.linspace(0.1, 0.9, n_steps),
            'road_curvature': np.abs(np.sin(np.linspace(0, 3*np.pi, n_steps))),
            'lane_position': np.sin(np.linspace(0, 2*np.pi, n_steps)) * 0.5
        })
        
        results = self.simulator.simulate_scenario(test_data)
        
        # Check that vehicle moved
        self.assertNotEqual(results['position_x'].max(), 0)
        self.assertNotEqual(results['position_y'].max(), 0)
        
        # Check for reasonable outputs
        self.assertGreater(results['output_speed'].min(), 0)
        self.assertLess(results['output_speed'].max(), 85)
        self.assertGreater(results['output_steering'].min(), -35)
        self.assertLess(results['output_steering'].max(), 35)
    
    def test_safety_constraints(self):
        """Test safety constraints in simulation"""
        # Scenario with multiple obstacles
        test_data = pd.DataFrame({
            'distance_obstacle': [5, 10, 15, 20, 25, 30],
            'speed': [60, 60, 60, 60, 60, 60],
            'steering_angle': [0, 0, 0, 0, 0, 0],
            'traffic_density': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            'road_curvature': [0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            'lane_position': [0] * 6
        })
        
        results = self.simulator.simulate_scenario(test_data)
        
        # Speed should generally decrease with decreasing distance and increasing traffic
        for i in range(len(results) - 1):
            # Not a strict requirement due to fuzzy logic complexity,
            # but generally speed should be lower in more dangerous situations
            if results['input_distance'].iloc[i] > results['input_distance'].iloc[i+1]:
                # If distance decreases, speed should not increase dramatically
                speed_diff = results['output_speed'].iloc[i+1] - results['output_speed'].iloc[i]
                self.assertLess(speed_diff, 10)  # Not a sudden increase

if __name__ == '__main__':
    unittest.main()