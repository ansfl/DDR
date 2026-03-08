import pandas as pd
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to import models
sys.path.append(str(Path(__file__).parent.parent))

from models.weinberg import WeinbergStepDetector

def main():
    # Paths (adjust to your local data)
    imu_file = "path/to/your/imu_data.csv"
    config_file = "../config/config_weinberg.xml"
    
    if not os.path.exists(config_file):
        print(f"Error: Config file not found at {config_file}")
        return

    # Initialize detector
    detector = WeinbergStepDetector(config_file)
    
    # Load your IMU data
    # Expected columns: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    try:
        imu_data = pd.read_csv(imu_file)
    except FileNotFoundError:
        print(f"IMU file not found: {imu_file}. Please provide a valid CSV.")
        return

    # 1. Detect steps
    num_steps, peaks, acc_mag = detector.detect_steps(imu_data)
    print(f"Detected {num_steps} steps.")

    # 2. Calculate step lengths
    step_lengths = detector.calculate_step_lengths()
    total_dist = sum(step_lengths)
    print(f"Estimated total distance: {total_dist:.2f} meters.")

    # 3. Calculate headings from IMU (Gyro integration or Madgwick)
    headings = detector.calculate_heading(imu_data)
    print(f"Calculated {len(headings)} heading angles.")

    # 4. Build trajectory
    trajectory = detector.build_trajectory(headings)
    print(f"Final estimated position: North={trajectory[-1, 0]:.2f}, East={trajectory[-1, 1]:.2f}")

if __name__ == "__main__":
    main()
