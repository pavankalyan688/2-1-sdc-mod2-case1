import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rule_engine import RuleEngine, Rule
from fuzzy_controller import FuzzyController

class TestRuleEngine(unittest.TestCase):
    """Test rule engine functionality"""
    
    def setUp(self):
        self.rule_engine = RuleEngine()
        self.controller = FuzzyController()
    
    def test_add_rule(self):
        """Test adding rules to engine"""
        self.rule_engine.add_rule(
            ['distance', 'traffic'],
            ['very_close', 'high'],
            ['brake'],
            'speed'
        )
        
        self.assertEqual(len(self.rule_engine.rules), 1)
        self.assertEqual(self.rule_engine.rules[0].inputs, ['distance', 'traffic'])
        self.assertEqual(self.rule_engine.rules[0].input_sets, ['very_close', 'high'])
        self.assertEqual(self.rule_engine.rules[0].output_set, 'brake')
        self.assertEqual(self.rule_engine.rules[0].output_type, 'speed')
    
    def test_multiple_rules(self):
        """Test adding multiple rules"""
        self.rule_engine.add_rule(
            ['distance'],
            ['close'],
            ['slow'],
            'speed'
        )
        self.rule_engine.add_rule(
            ['steering'],
            ['left'],
            ['left'],
            'steering'
        )
        
        self.assertEqual(len(self.rule_engine.rules), 2)
    
    def test_rule_evaluation(self):
        """Test individual rule evaluation"""
        rule = Rule(
            ['distance', 'traffic'],
            ['close', 'high'],
            'brake',
            'speed'
        )
        
        fuzzy_inputs = {
            'distance': {'very_close': 0, 'close': 0.8, 'medium': 0.2, 'far': 0},
            'traffic': {'low': 0, 'moderate': 0.2, 'high': 0.9}
        }
        
        strength = rule.evaluate(fuzzy_inputs)
        self.assertEqual(strength, min(0.8, 0.9))  # Should be 0.8
    
    def test_rule_aggregation(self):
        """Test rule aggregation"""
        # Add test rules
        self.rule_engine.add_rule(
            ['distance', 'traffic'],
            ['very_close', 'high'],
            ['brake'],
            'speed'
        )
        self.rule_engine.add_rule(
            ['distance', 'traffic'],
            ['close', 'high'],
            ['slow'],
            'speed'
        )
        self.rule_engine.add_rule(
            ['distance', 'traffic'],
            ['medium', 'low'],
            ['medium'],
            'speed'
        )
        
        # Test with inputs
        fuzzy_inputs = {
            'distance': {'very_close': 0.9, 'close': 0.1, 'medium': 0, 'far': 0},
            'traffic': {'low': 0, 'moderate': 0, 'high': 1.0},
            'speed': {'slow': 0, 'medium': 0, 'fast': 0},
            'steering': {'hard_left': 0, 'left': 0, 'straight': 0, 'right': 0, 'hard_right': 0},
            'curvature': {'straight': 0, 'moderate_curve': 0, 'sharp_curve': 0}
        }
        
        speed_agg, steering_agg = self.rule_engine.apply_rules(fuzzy_inputs)
        
        # Should have brake and slow activated
        self.assertIn('brake', speed_agg)
        self.assertIn('slow', speed_agg)
        self.assertEqual(speed_agg['brake'], 0.9)  # Min of very_close and high
        self.assertEqual(speed_agg['slow'], 0.1)  # Min of close and high
    
    def test_empty_rules(self):
        """Test rule engine with no rules"""
        empty_engine = RuleEngine()
        fuzzy_inputs = {
            'distance': {'close': 0.5},
            'traffic': {'high': 0.5}
        }
        
        speed_agg, steering_agg = empty_engine.apply_rules(fuzzy_inputs)
        
        self.assertEqual(len(speed_agg), 0)
        self.assertEqual(len(steering_agg), 0)
    
    def test_max_aggregation(self):
        """Test that aggregation uses max for same output"""
        # Add two rules that could activate the same output
        self.rule_engine.add_rule(
            ['distance'],
            ['close'],
            ['slow'],
            'speed'
        )
        self.rule_engine.add_rule(
            ['traffic'],
            ['high'],
            ['slow'],
            'speed'
        )
        
        fuzzy_inputs = {
            'distance': {'close': 0.8},
            'traffic': {'high': 0.6},
            'speed': {'slow': 0, 'medium': 0, 'fast': 0},
            'steering': {'hard_left': 0, 'left': 0, 'straight': 0, 'right': 0, 'hard_right': 0},
            'curvature': {'straight': 0, 'moderate_curve': 0, 'sharp_curve': 0}
        }
        
        speed_agg, _ = self.rule_engine.apply_rules(fuzzy_inputs)
        
        # Should use max of the two rule strengths
        self.assertEqual(speed_agg['slow'], max(0.8, 0.6))  # Should be 0.8
    
    def test_different_output_types(self):
        """Test rules with different output types"""
        self.rule_engine.add_rule(
            ['steering'],
            ['left'],
            ['left'],
            'steering'
        )
        self.rule_engine.add_rule(
            ['steering'],
            ['right'],
            ['right'],
            'steering'
        )
        self.rule_engine.add_rule(
            ['distance'],
            ['far'],
            ['fast'],
            'speed'
        )
        
        fuzzy_inputs = {
            'steering': {'hard_left': 0, 'left': 0.7, 'straight': 0.3, 'right': 0, 'hard_right': 0},
            'distance': {'far': 0.9},
            'speed': {'slow': 0, 'medium': 0, 'fast': 0},
            'traffic': {'low': 0, 'moderate': 0, 'high': 0},
            'curvature': {'straight': 0, 'moderate_curve': 0, 'sharp_curve': 0}
        }
        
        speed_agg, steering_agg = self.rule_engine.apply_rules(fuzzy_inputs)
        
        self.assertIn('left', steering_agg)
        self.assertEqual(steering_agg['left'], 0.7)
        self.assertIn('fast', speed_agg)
        self.assertEqual(speed_agg['fast'], 0.9)

if __name__ == '__main__':
    unittest.main()