import numpy as np
import pandas as pd
from fuzzy_controller import FuzzyController

class VehicleSimulator:
    """Simulates vehicle dynamics for testing fuzzy control"""
    
    def __init__(self):
        self.controller = FuzzyController()
        self.position = [0, 0]  # x, y coordinates
        self.heading = 0  # radians
        self.speed = 0  # km/h
        
    def simulate_scenario(self, scenario_data):
        """Simulate a driving scenario"""
        results = []
        
        for idx, row in scenario_data.iterrows():
            # Prepare inputs for controller
            inputs = {
                'distance': row['distance_obstacle'],
                'speed': self.speed,
                'steering': row['steering_angle'],
                'traffic': row['traffic_density'],
                'curvature': row['road_curvature']
            }
            
            # Get fuzzy control output
            control_output = self.controller.evaluate(inputs)
            
            # Update vehicle state
            self.speed = control_output['speed_command']
            
            # Convert steering command to heading change
            steering_rad = np.radians(control_output['steering_command'])
            self.heading += steering_rad * 0.1  # Simple vehicle model
            
            # Update position
            self.position[0] += self.speed * 0.05 * np.cos(self.heading)
            self.position[1] += self.speed * 0.05 * np.sin(self.heading)
            
            # Store results
            result = {
                'scenario_id': idx,
                'input_distance': inputs['distance'],
                'input_speed': inputs['speed'],
                'input_steering': inputs['steering'],
                'input_traffic': inputs['traffic'],
                'input_curvature': inputs['curvature'],
                'output_speed': control_output['speed_command'],
                'output_steering': control_output['steering_command'],
                'position_x': self.position[0],
                'position_y': self.position[1],
                'heading': self.heading
            }
            
            # Add fuzzy membership values for analysis
            for var, sets in control_output['fuzzy_inputs'].items():
                for set_name, membership in sets.items():
                    result[f'fuzzy_{var}_{set_name}'] = membership
            
            results.append(result)
        
        return pd.DataFrame(results)