import numpy as np

class Rule:
    """Single fuzzy rule"""
    def __init__(self, inputs, input_sets, output_set, output_type):
        self.inputs = inputs
        self.input_sets = input_sets
        self.output_set = output_set
        self.output_type = output_type
    
    def evaluate(self, fuzzy_inputs):
        """Evaluate rule and return membership degree"""
        # Find minimum membership across all input conditions
        membership_degree = 1.0
        
        for idx, input_name in enumerate(self.inputs):
            set_name = self.input_sets[idx]
            membership = fuzzy_inputs[input_name][set_name]
            membership_degree = min(membership_degree, membership)
        
        return membership_degree

class RuleEngine:
    """Manages and applies fuzzy rules"""
    
    def __init__(self):
        self.rules = []
    
    def add_rule(self, inputs, input_sets, output_sets, output_type):
        """Add a new rule to the engine"""
        # For each output set in the rule
        for output_set in output_sets:
            rule = Rule(inputs, input_sets, output_set, output_type)
            self.rules.append(rule)
    
    def apply_rules(self, fuzzy_inputs):
        """Apply all rules and aggregate outputs"""
        aggregated_speed = {}
        aggregated_steering = {}
        
        for rule in self.rules:
            # Evaluate rule
            strength = rule.evaluate(fuzzy_inputs)
            
            if strength > 0:
                # Add to appropriate output aggregation
                if rule.output_type == 'speed':
                    if rule.output_set not in aggregated_speed:
                        aggregated_speed[rule.output_set] = strength
                    else:
                        aggregated_speed[rule.output_set] = max(
                            aggregated_speed[rule.output_set], strength
                        )
                else:  # steering
                    if rule.output_set not in aggregated_steering:
                        aggregated_steering[rule.output_set] = strength
                    else:
                        aggregated_steering[rule.output_set] = max(
                            aggregated_steering[rule.output_set], strength
                        )
        
        return aggregated_speed, aggregated_steering