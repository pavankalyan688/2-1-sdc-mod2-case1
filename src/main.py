import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fuzzy_controller import FuzzyController
from vehicle_simulator import VehicleSimulator

def main():
    """Main execution function"""
    print("Initializing Autonomous Vehicle Fuzzy Control System...")
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_raw_path = os.path.join(project_root, 'data', 'raw', 'sensor_data.csv')
    data_processed_path = os.path.join(project_root, 'data', 'processed', 'training_data.csv')
    results_plots_path = os.path.join(project_root, 'results', 'plots')
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(data_raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(data_processed_path), exist_ok=True)
    os.makedirs(results_plots_path, exist_ok=True)
    
    # Load dataset
    try:
        sensor_data = pd.read_csv(data_raw_path)
        print(f"Loaded {len(sensor_data)} sensor readings from {data_raw_path}")
    except FileNotFoundError:
        print("Sensor data not found. Generating sample data...")
        sensor_data = generate_sample_data(data_raw_path)
    
    # Initialize simulator
    simulator = VehicleSimulator()
    
    # Run simulation
    print("Running simulation...")
    results = simulator.simulate_scenario(sensor_data)
    
    # Save results
    results.to_csv(data_processed_path, index=False)
    print(f"Simulation complete. Results saved to {data_processed_path}")
    
    # Analyze results
    analyze_results(results, results_plots_path)

def generate_sample_data(save_path):
    """Generate sample sensor data if file doesn't exist"""
    np.random.seed(42)
    
    # Generate 50 records
    data = {
        'distance_obstacle': np.concatenate([
            np.random.uniform(0, 15, 10),
            np.random.uniform(15, 30, 10),
            np.random.uniform(30, 55, 15),
            np.random.uniform(55, 100, 15)
        ]),
        'speed': np.random.uniform(0, 80, 50),
        'steering_angle': np.random.uniform(-30, 30, 50),
        'traffic_density': np.random.uniform(0, 1, 50),
        'road_curvature': np.random.uniform(0, 1, 50),
        'lane_position': np.random.uniform(-1, 1, 50),
    }
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Add target outputs
    def compute_recommended_speed(row):
        if row['distance_obstacle'] < 15 or row['traffic_density'] > 0.7:
            return np.random.uniform(5, 20)
        elif row['distance_obstacle'] < 30 or row['traffic_density'] > 0.5:
            return np.random.uniform(20, 40)
        elif row['distance_obstacle'] < 55 or row['road_curvature'] > 0.6:
            return np.random.uniform(40, 60)
        else:
            return np.random.uniform(60, 80)
    
    def compute_recommended_steering(row):
        if abs(row['steering_angle']) > 20:
            return row['steering_angle'] * 0.8
        elif abs(row['steering_angle']) > 10:
            return row['steering_angle'] * 0.6
        elif row['road_curvature'] > 0.7:
            return np.random.uniform(10, 25) * np.random.choice([-1, 1])
        else:
            return row['lane_position'] * 10
    
    df['recommended_speed'] = df.apply(compute_recommended_speed, axis=1)
    df['recommended_steering'] = df.apply(compute_recommended_steering, axis=1)
    
    df.to_csv(save_path, index=False)
    print(f"Generated {len(df)} sample records and saved to {save_path}")
    return df

def analyze_results(results, save_path):
    """Analyze and display simulation results"""
    print("\n--- Simulation Results Analysis ---")
    print(f"Average Speed Command: {results['output_speed'].mean():.2f} km/h")
    print(f"Average Steering Command: {results['output_steering'].mean():.2f} degrees")
    print(f"Max Speed Command: {results['output_speed'].max():.2f} km/h")
    print(f"Min Speed Command: {results['output_speed'].min():.2f} km/h")
    
    # Safety metrics
    dangerous_scenarios = results[
        (results['input_distance'] < 15) & 
        (results['output_speed'] > 20)
    ]
    print(f"\nDangerous scenarios (distance<15m, speed>20km/h): {len(dangerous_scenarios)}")
    
    # Plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Speed control
        axes[0, 0].scatter(results['input_distance'], results['output_speed'], alpha=0.6)
        axes[0, 0].set_xlabel('Distance to Obstacle (m)')
        axes[0, 0].set_ylabel('Speed Command (km/h)')
        axes[0, 0].set_title('Speed Response vs Distance')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Steering control
        axes[0, 1].scatter(results['input_steering'], results['output_steering'], alpha=0.6)
        axes[0, 1].set_xlabel('Input Steering (degrees)')
        axes[0, 1].set_ylabel('Output Steering Command (degrees)')
        axes[0, 1].set_title('Steering Response')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Traffic influence
        axes[1, 0].scatter(results['input_traffic'], results['output_speed'], alpha=0.6)
        axes[1, 0].set_xlabel('Traffic Density')
        axes[1, 0].set_ylabel('Speed Command (km/h)')
        axes[1, 0].set_title('Speed vs Traffic Density')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Vehicle path
        axes[1, 1].plot(results['position_x'], results['position_y'], 'b-', linewidth=2)
        axes[1, 1].set_xlabel('X Position (m)')
        axes[1, 1].set_ylabel('Y Position (m)')
        axes[1, 1].set_title('Vehicle Path')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axis('equal')
        
        plt.tight_layout()
        plot_path = os.path.join(save_path, 'simulation_analysis.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Analysis plots saved to {plot_path}")
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Skipping plots.")

if __name__ == "__main__":
    main()