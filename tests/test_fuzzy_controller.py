import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fuzzy_controller import FuzzyController
from fuzzy_sets import TriangularFuzzySet, TrapezoidalFuzzySet, GaussianFuzzySet

class TestFuzzySets(unittest.TestCase):
    """Test fuzzy set membership functions"""
    
    def setUp(self):
        self.triangular = TriangularFuzzySet('test', 0, 5, 10)
        self.trapezoidal = TrapezoidalFuzzySet('test', 0, 3, 7, 10)
        self.gaussian = GaussianFuzzySet('test', 5, 2)
    
    def test_triangular_membership(self):
        """Test triangular membership function"""
        self.assertEqual(self.triangular.membership(5), 1.0)
        self.assertEqual(self.triangular.membership(0), 0.0)
        self.assertEqual(self.triangular.membership(10), 0.0)
        self.assertEqual(self.triangular.membership(2.5), 0.5)
        self.assertEqual(self.triangular.membership(7.5), 0.5)
    
    def test_trapezoidal_membership(self):
        """Test trapezoidal membership function"""
        self.assertEqual(self.trapezoidal.membership(0), 0.0)
        self.assertEqual(self.trapezoidal.membership(1.5), 0.5)
        self.assertEqual(self.trapezoidal.membership(3), 1.0)
        self.assertEqual(self.trapezoidal.membership(7), 1.0)
        self.assertEqual(self.trapezoidal.membership(10), 0.0)
        self.assertEqual(self.trapezoidal.membership(8.5), 0.5)
    
    def test_gaussian_membership(self):
        """Test Gaussian membership function"""
        self.assertEqual(self.gaussian.membership(5), 1.0)
        self.assertAlmostEqual(self.gaussian.membership(3), np.exp(-1/2), places=2)
        self.assertAlmostEqual(self.gaussian.membership(7), np.exp(-1/2), places=2)

class TestFuzzyController(unittest.TestCase):
    """Test fuzzy controller functionality"""
    
    def setUp(self):
        self.controller = FuzzyController()
    
    def test_fuzzification(self):
        """Test input fuzzification"""
        inputs = {
            'distance': 25,
            'speed': 50,
            'steering': 5,
            'traffic': 0.5,
            'curvature': 0.3
        }
        
        fuzzy_inputs = self.controller.fuzzify(inputs)
        
        # Check that all input variables are fuzzified
        self.assertIn('distance', fuzzy_inputs)
        self.assertIn('speed', fuzzy_inputs)
        self.assertIn('steering', fuzzy_inputs)
        self.assertIn('traffic', fuzzy_inputs)
        self.assertIn('curvature', fuzzy_inputs)
        
        # Check distance membership values
        self.assertGreater(fuzzy_inputs['distance']['close'], 0)
        self.assertGreater(fuzzy_inputs['distance']['medium'], 0)
        
        # Check speed membership values
        self.assertGreater(fuzzy_inputs['speed']['medium'], 0)
        self.assertGreater(fuzzy_inputs['speed']['fast'], 0)
    
    def test_rule_evaluation(self):
        """Test rule evaluation"""
        inputs = {
            'distance': 10,  # close
            'speed': 60,
            'steering': 0,
            'traffic': 0.8,  # high
            'curvature': 0.1
        }
        
        result = self.controller.evaluate(inputs)
        
        # Should output low speed (brake or slow)
        self.assertLess(result['speed_command'], 30)
        
        # Steering should be reasonable
        self.assertGreaterEqual(result['steering_command'], -30)
        self.assertLessEqual(result['steering_command'], 30)
    
    def test_emergency_braking(self):
        """Test emergency braking scenario"""
        inputs = {
            'distance': 3,  # very close
            'speed': 70,
            'steering': 0,
            'traffic': 0.9,
            'curvature': 0.2
        }
        
        result = self.controller.evaluate(inputs)
        
        # Should apply brakes (very low speed)
        self.assertLess(result['speed_command'], 10)
    
    def test_high_speed_clear_road(self):
        """Test high speed on clear road"""
        inputs = {
            'distance': 80,  # far
            'speed': 65,
            'steering': 0,
            'traffic': 0.1,  # low
            'curvature': 0.05  # straight
        }
        
        result = self.controller.evaluate(inputs)
        
        # Should maintain high speed
        self.assertGreater(result['speed_command'], 60)
        
        # Steering should be straight
        self.assertLess(abs(result['steering_command']), 5)
    
    def test_defuzzification(self):
        """Test defuzzification methods"""
        # Test speed defuzzification
        aggregated_speed = {
            'brake': 0.0,
            'slow': 0.5,
            'medium': 0.8,
            'fast': 0.2
        }
        
        speed_output = self.controller.defuzzify('speed', aggregated_speed)
        self.assertGreater(speed_output, 0)
        self.assertLess(speed_output, 80)
        
        # Test steering defuzzification
        aggregated_steering = {
            'hard_left': 0.0,
            'left': 0.3,
            'straight': 0.9,
            'right': 0.1,
            'hard_right': 0.0
        }
        
        steering_output = self.controller.defuzzify('steering', aggregated_steering)
        self.assertGreater(steering_output, -30)
        self.assertLess(steering_output, 30)

class TestIntegration(unittest.TestCase):
    """Integration tests for the fuzzy controller"""
    
    def setUp(self):
        self.controller = FuzzyController()
    
    def test_end_to_end(self):
        """Test complete control loop"""
        test_cases = [
            {'distance': 5, 'speed': 40, 'steering': 0, 'traffic': 0.8, 'curvature': 0.1},
            {'distance': 30, 'speed': 50, 'steering': 10, 'traffic': 0.4, 'curvature': 0.5},
            {'distance': 60, 'speed': 70, 'steering': -5, 'traffic': 0.2, 'curvature': 0.2},
            {'distance': 90, 'speed': 75, 'steering': 20, 'traffic': 0.1, 'curvature': 0.8},
            {'distance': 15, 'speed': 30, 'steering': -15, 'traffic': 0.6, 'curvature': 0.7},
        ]
        
        for inputs in test_cases:
            result = self.controller.evaluate(inputs)
            
            # Check that outputs are within valid ranges
            self.assertGreaterEqual(result['speed_command'], 0)
            self.assertLessEqual(result['speed_command'], 80)
            self.assertGreaterEqual(result['steering_command'], -30)
            self.assertLessEqual(result['steering_command'], 30)
            
            # Check that all required keys are present
            self.assertIn('fuzzy_inputs', result)
            self.assertIn('aggregated_speed', result)
            self.assertIn('aggregated_steering', result)
    
    def test_consistency(self):
        """Test consistency of controller outputs"""
        inputs = {
            'distance': 50,
            'speed': 55,
            'steering': 5,
            'traffic': 0.3,
            'curvature': 0.2
        }
        
        # Run controller multiple times with same inputs
        results = [self.controller.evaluate(inputs) for _ in range(5)]
        
        # Results should be consistent
        for i in range(1, len(results)):
            self.assertEqual(results[0]['speed_command'], results[i]['speed_command'])
            self.assertEqual(results[0]['steering_command'], results[i]['steering_command'])

if __name__ == '__main__':
    unittest.main()