import numpy as np
# Remove relative imports - use absolute imports instead
from fuzzy_sets import TriangularFuzzySet, TrapezoidalFuzzySet, GaussianFuzzySet
from rule_engine import RuleEngine

class FuzzyController:
    """Main fuzzy logic controller for autonomous vehicle"""
    
    def __init__(self):
        # Define input fuzzy sets for speed control
        self.speed_sets = {
            'slow': TriangularFuzzySet('slow', 0, 0, 40),
            'medium': TriangularFuzzySet('medium', 20, 50, 80),
            'fast': TriangularFuzzySet('fast', 50, 80, 100)
        }
        
        # Define input fuzzy sets for distance to obstacle
        self.distance_sets = {
            'very_close': TrapezoidalFuzzySet('very_close', 0, 0, 5, 15),
            'close': TriangularFuzzySet('close', 5, 15, 30),
            'medium': TriangularFuzzySet('medium', 15, 35, 55),
            'far': TrapezoidalFuzzySet('far', 40, 70, 100, 100)
        }
        
        # Define input fuzzy sets for steering angle
        self.steering_sets = {
            'hard_left': TrapezoidalFuzzySet('hard_left', -30, -30, -20, -10),
            'left': TriangularFuzzySet('left', -20, -10, -2),
            'straight': TriangularFuzzySet('straight', -5, 0, 5),
            'right': TriangularFuzzySet('right', 2, 10, 20),
            'hard_right': TrapezoidalFuzzySet('hard_right', 10, 20, 30, 30)
        }
        
        # Define input fuzzy sets for traffic density
        self.traffic_sets = {
            'low': TriangularFuzzySet('low', 0, 0, 0.4),
            'moderate': TriangularFuzzySet('moderate', 0.2, 0.5, 0.8),
            'high': TriangularFuzzySet('high', 0.6, 1, 1)
        }
        
        # Define input fuzzy sets for road curvature
        self.curvature_sets = {
            'straight': TriangularFuzzySet('straight', 0, 0, 0.3),
            'moderate_curve': TriangularFuzzySet('moderate_curve', 0.2, 0.5, 0.8),
            'sharp_curve': TriangularFuzzySet('sharp_curve', 0.6, 1, 1)
        }
        
        # Define output fuzzy sets for speed command
        self.speed_output_sets = {
            'brake': TriangularFuzzySet('brake', 0, 0, 20),
            'slow': TriangularFuzzySet('slow', 5, 25, 45),
            'medium': TriangularFuzzySet('medium', 30, 50, 70),
            'fast': TriangularFuzzySet('fast', 55, 80, 80)
        }
        
        # Define output fuzzy sets for steering command
        self.steering_output_sets = {
            'hard_left': TrapezoidalFuzzySet('hard_left', -30, -30, -25, -15),
            'left': TriangularFuzzySet('left', -20, -10, -5),
            'straight': TriangularFuzzySet('straight', -5, 0, 5),
            'right': TriangularFuzzySet('right', 5, 10, 20),
            'hard_right': TrapezoidalFuzzySet('hard_right', 15, 25, 30, 30)
        }
        
        # Initialize rule engine
        self.rule_engine = RuleEngine()
        self._define_rules()
    
    def _define_rules(self):
        """Define fuzzy rules for speed control"""
        # Speed rules
        speed_rules = [
            # If distance is very_close OR traffic is high THEN brake
            (['distance', 'traffic'], ['very_close', 'high'], ['brake'], 'speed'),
            # If distance is close OR curvature is sharp THEN slow
            (['distance', 'curvature'], ['close', 'sharp_curve'], ['slow'], 'speed'),
            # If distance is medium AND traffic is low THEN medium
            (['distance', 'traffic'], ['medium', 'low'], ['medium'], 'speed'),
            # If distance is far AND curvature is straight THEN fast
            (['distance', 'curvature'], ['far', 'straight'], ['fast'], 'speed'),
        ]
        
        # Steering rules
        steering_rules = [
            # If steering is hard_left THEN hard_left
            (['steering'], ['hard_left'], ['hard_left'], 'steering'),
            # If steering is left THEN left
            (['steering'], ['left'], ['left'], 'steering'),
            # If steering is straight THEN straight
            (['steering'], ['straight'], ['straight'], 'steering'),
            # If steering is right THEN right
            (['steering'], ['right'], ['right'], 'steering'),
            # If steering is hard_right THEN hard_right
            (['steering'], ['hard_right'], ['hard_right'], 'steering'),
            # If curvature is sharp_curve THEN reduce speed and adjust steering
            (['curvature'], ['sharp_curve'], ['left'], 'steering'),
            (['curvature'], ['sharp_curve'], ['slow'], 'speed'),
        ]
        
        for rule in speed_rules:
            self.rule_engine.add_rule(*rule)
        for rule in steering_rules:
            self.rule_engine.add_rule(*rule)
    
    def fuzzify(self, inputs):
        """Convert crisp inputs to fuzzy membership values"""
        fuzzy_inputs = {}
        
        # Fuzzify distance
        fuzzy_inputs['distance'] = {}
        for set_name, fuzzy_set in self.distance_sets.items():
            fuzzy_inputs['distance'][set_name] = fuzzy_set.membership(inputs['distance'])
        
        # Fuzzify speed
        fuzzy_inputs['speed'] = {}
        for set_name, fuzzy_set in self.speed_sets.items():
            fuzzy_inputs['speed'][set_name] = fuzzy_set.membership(inputs['speed'])
        
        # Fuzzify steering
        fuzzy_inputs['steering'] = {}
        for set_name, fuzzy_set in self.steering_sets.items():
            fuzzy_inputs['steering'][set_name] = fuzzy_set.membership(inputs['steering'])
        
        # Fuzzify traffic
        fuzzy_inputs['traffic'] = {}
        for set_name, fuzzy_set in self.traffic_sets.items():
            fuzzy_inputs['traffic'][set_name] = fuzzy_set.membership(inputs['traffic'])
        
        # Fuzzify curvature
        fuzzy_inputs['curvature'] = {}
        for set_name, fuzzy_set in self.curvature_sets.items():
            fuzzy_inputs['curvature'][set_name] = fuzzy_set.membership(inputs['curvature'])
        
        return fuzzy_inputs
    
    def defuzzify(self, output_type, aggregated_output):
        """Defuzzify using centroid method"""
        if output_type == 'speed':
            output_sets = self.speed_output_sets
        else:  # steering
            output_sets = self.steering_output_sets
        
        # Create discrete points for centroid calculation
        if output_type == 'speed':
            x_range = np.linspace(0, 80, 100)
        else:
            x_range = np.linspace(-30, 30, 100)
        
        numerator = 0
        denominator = 0
        
        for x in x_range:
            # Find maximum membership across all activated output sets
            membership_values = []
            for set_name, fuzzy_set in output_sets.items():
                if set_name in aggregated_output:
                    membership = min(aggregated_output[set_name], fuzzy_set.membership(x))
                    membership_values.append(membership)
            
            if membership_values:
                max_membership = max(membership_values)
                numerator += x * max_membership
                denominator += max_membership
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def evaluate(self, inputs):
        """Evaluate fuzzy controller with given inputs"""
        # Fuzzify inputs
        fuzzy_inputs = self.fuzzify(inputs)
        
        # Apply rules
        aggregated_speed, aggregated_steering = self.rule_engine.apply_rules(fuzzy_inputs)
        
        # Defuzzify outputs
        speed_command = self.defuzzify('speed', aggregated_speed)
        steering_command = self.defuzzify('steering', aggregated_steering)
        
        return {
            'speed_command': speed_command,
            'steering_command': steering_command,
            'fuzzy_inputs': fuzzy_inputs,
            'aggregated_speed': aggregated_speed,
            'aggregated_steering': aggregated_steering
        }